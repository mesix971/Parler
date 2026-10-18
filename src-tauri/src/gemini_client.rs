use anyhow::Result;
use base64::Engine;
use hound::{WavSpec, WavWriter};
use log::{debug, warn};
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::io::Cursor;
use std::sync::OnceLock;
use std::time::Duration;

const GEMINI_API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta/models";

/// Délai maximal pour établir la connexion avec l'API.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Délai maximal d'une requête complète (envoi de l'audio + réponse). Sans lui,
/// un réseau qui cale laisse la dictée bloquée en « Transcription… » jusqu'au
/// redémarrage de l'application.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

/// Client HTTP partagé : la connexion TLS est réutilisée d'une dictée à l'autre
/// au lieu d'être renégociée à chaque fois.
pub(crate) fn http_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(REQUEST_TIMEOUT)
            .build()
            .unwrap_or_else(|e| {
                warn!("Impossible de configurer le client HTTP Gemini ({}), client par défaut utilisé", e);
                reqwest::Client::new()
            })
    })
}

/// Catégorie d'échec d'un appel Gemini, pour afficher un message clair à
/// l'utilisateur au lieu d'un simple échec silencieux.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeminiErrorKind {
    /// Clé absente, invalide ou sans droit sur l'API.
    InvalidKey,
    /// Quota ou limite de débit dépassé (HTTP 429).
    Quota,
    /// Service Google indisponible (HTTP 5xx).
    Unavailable,
    /// Pas de réponse dans le délai imparti.
    Timeout,
    /// Connexion impossible (réseau coupé, DNS…).
    Network,
    /// Réponse bloquée par les filtres de sécurité de Gemini.
    Blocked,
    /// Modèle inconnu ou retiré par Google (HTTP 404).
    ModelNotFound,
    /// Toute autre erreur (requête refusée, réponse illisible…).
    Other,
}

impl GeminiErrorKind {
    /// Code transmis à l'interface (traduit côté frontend).
    pub fn code(self) -> &'static str {
        match self {
            GeminiErrorKind::InvalidKey => "gemini_invalid_key",
            GeminiErrorKind::Quota => "gemini_quota",
            GeminiErrorKind::Unavailable => "gemini_unavailable",
            GeminiErrorKind::Timeout => "timeout",
            GeminiErrorKind::Network => "network",
            GeminiErrorKind::Blocked => "gemini_blocked",
            GeminiErrorKind::ModelNotFound => "gemini_model_unavailable",
            GeminiErrorKind::Other => "transcription_failed",
        }
    }
}

#[derive(Debug)]
pub struct GeminiError {
    pub kind: GeminiErrorKind,
    pub detail: String,
}

impl GeminiError {
    fn new(kind: GeminiErrorKind, detail: impl Into<String>) -> Self {
        Self {
            kind,
            detail: detail.into(),
        }
    }

    /// Erreur « clé API absente », signalée comme une clé invalide.
    pub fn missing_key() -> Self {
        Self::new(GeminiErrorKind::InvalidKey, "clé API Gemini non configurée")
    }
}

impl fmt::Display for GeminiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Gemini ({:?}) : {}", self.kind, self.detail)
    }
}

impl std::error::Error for GeminiError {}

/// Un nom de modèle est inséré tel quel dans l'URL : on n'accepte que des
/// caractères sûrs (ex. « gemini-2.5-flash »).
fn validate_model_name(model: &str) -> std::result::Result<(), GeminiError> {
    let valid = !model.is_empty()
        && model
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
    if valid {
        Ok(())
    } else {
        Err(GeminiError::new(
            GeminiErrorKind::Other,
            format!("nom de modèle invalide : {:?}", model),
        ))
    }
}

/// Consigne de transcription. Nommer les commandes dictées aide Gemini à les
/// reconnaître : sans cette phrase, « tiret » sortait en « tirer » (même son)
/// et les tirets d'une liste disparaissaient ; avec elle, les dictées sans
/// commande ne changent pas (vérifié sur de vraies dictées le 2026-10-17).
/// `audio_toolkit::text` applique ensuite les commandes.
const TRANSCRIPTION_PROMPT: &str =
    "Transcribe this audio. Return only the transcript text, nothing else. \
     The speaker may dictate French punctuation and formatting commands such as \
     « tiret », « à la ligne », « nouveau paragraphe » or « point d'exclamation »: \
     write these commands as the spoken words, do not apply them.";

/// Réglages de transcription selon le modèle :
/// - seuls les modèles « Flash » 2.5 acceptent de désactiver la réflexion
///   (thinkingBudget = 0) ; sur les autres (2.5 Pro…), ce réglage provoque une
///   erreur 400. Pour une simple transcription, la réflexion n'apporte rien et
///   ajoute de la latence ;
/// - température 0 sur les modèles 2.x : à la valeur par défaut (1.0), une même
///   dictée sortait différemment à chaque essai (« Tirez », « tiré »,
///   « Tire »…), ce qui rend les commandes dictées aléatoires. Google
///   déconseille au contraire de la baisser sur les Gemini 3 (boucles,
///   dégradation) : on la leur laisse.
fn generation_config_for(model: &str) -> Option<GenerationConfig> {
    if !model.starts_with("gemini-2.") {
        return None;
    }
    Some(GenerationConfig {
        thinking_config: model
            .starts_with("gemini-2.5-flash")
            .then_some(ThinkingConfig { thinking_budget: 0 }),
        temperature: Some(0.0),
    })
}

#[derive(Serialize)]
struct InlineData {
    #[serde(rename = "mimeType")]
    mime_type: String,
    data: String,
}

#[derive(Serialize)]
struct Part {
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    inline_data: Option<InlineData>,
}

#[derive(Serialize)]
struct Content {
    parts: Vec<Part>,
}

#[derive(Serialize)]
struct SystemInstruction {
    parts: Vec<Part>,
}

#[derive(Serialize)]
struct ThinkingConfig {
    #[serde(rename = "thinkingBudget")]
    thinking_budget: u32,
}

#[derive(Serialize)]
struct GenerationConfig {
    #[serde(rename = "thinkingConfig")]
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking_config: Option<ThinkingConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Serialize)]
struct GenerateContentRequest {
    contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "systemInstruction")]
    system_instruction: Option<SystemInstruction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "generationConfig")]
    generation_config: Option<GenerationConfig>,
}

#[derive(Deserialize)]
struct GenerateContentResponse {
    candidates: Option<Vec<Candidate>>,
    #[serde(rename = "promptFeedback")]
    prompt_feedback: Option<PromptFeedback>,
}

#[derive(Deserialize)]
struct PromptFeedback {
    #[serde(rename = "blockReason")]
    block_reason: Option<String>,
}

#[derive(Deserialize)]
struct Candidate {
    content: Option<CandidateContent>,
    #[serde(rename = "finishReason")]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct CandidateContent {
    parts: Option<Vec<ResponsePart>>,
}

#[derive(Deserialize)]
struct ResponsePart {
    text: Option<String>,
    /// Les modèles dédiés à la transcription (gemini-3.5-transcribe) renvoient
    /// le texte ici et non dans `text`.
    #[serde(rename = "audioTranscription")]
    audio_transcription: Option<AudioTranscription>,
    /// Les parties de « réflexion » éventuelles ne font pas partie du résultat.
    #[serde(default)]
    thought: bool,
}

#[derive(Deserialize)]
struct AudioTranscription {
    text: Option<String>,
}

impl ResponsePart {
    /// Texte utile de la partie (hors réflexion), quel que soit le champ
    /// employé par le modèle.
    fn into_text(self) -> Option<String> {
        if self.thought {
            return None;
        }
        self.text
            .or_else(|| self.audio_transcription.and_then(|t| t.text))
    }
}

fn encode_samples_to_wav(samples: &[f32]) -> Result<Vec<u8>> {
    let spec = WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut buffer = Vec::new();
    {
        let cursor = Cursor::new(&mut buffer);
        let mut writer = WavWriter::new(cursor, spec)?;
        for sample in samples {
            let sample_i16 = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            writer.write_sample(sample_i16)?;
        }
        writer.finalize()?;
    }

    Ok(buffer)
}

fn api_key_header(api_key: &str) -> std::result::Result<HeaderMap, GeminiError> {
    if api_key.trim().is_empty() {
        return Err(GeminiError::new(
            GeminiErrorKind::InvalidKey,
            "clé API Gemini non configurée",
        ));
    }
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(
        "x-goog-api-key",
        HeaderValue::from_str(api_key.trim()).map_err(|_| {
            GeminiError::new(
                GeminiErrorKind::InvalidKey,
                "la clé API contient des caractères invalides",
            )
        })?,
    );
    Ok(headers)
}

/// Traduit une erreur réseau de reqwest en catégorie compréhensible.
fn classify_transport_error(e: &reqwest::Error) -> GeminiError {
    let kind = if e.is_timeout() {
        GeminiErrorKind::Timeout
    } else if e.is_connect() || e.is_request() {
        GeminiErrorKind::Network
    } else {
        GeminiErrorKind::Other
    };
    GeminiError::new(kind, format!("requête impossible : {}", e))
}

/// Traduit un statut HTTP d'erreur en catégorie compréhensible. `body` est le
/// corps de la réponse (il ne contient jamais la clé).
fn classify_http_error(status: reqwest::StatusCode, body: &str) -> GeminiError {
    let kind = match status.as_u16() {
        // Google répond 400 « API_KEY_INVALID » pour une clé erronée.
        400 if body.contains("API_KEY_INVALID") || body.contains("API key not valid") => {
            GeminiErrorKind::InvalidKey
        }
        401 | 403 => GeminiErrorKind::InvalidKey,
        404 => GeminiErrorKind::ModelNotFound,
        429 => GeminiErrorKind::Quota,
        500..=599 => GeminiErrorKind::Unavailable,
        _ => GeminiErrorKind::Other,
    };
    // Le corps d'erreur de Google peut être long : on n'en garde que le début.
    let short: String = body.chars().take(300).collect();
    GeminiError::new(kind, format!("HTTP {} : {}", status, short))
}

/// Envoie une requête generateContent et renvoie le texte de la réponse
/// (toutes les parties concaténées, hors réflexion).
async fn generate_content(
    api_key: &str,
    model: &str,
    request: &GenerateContentRequest,
) -> std::result::Result<String, GeminiError> {
    validate_model_name(model)?;
    let headers = api_key_header(api_key)?;
    let url = format!("{}/{}:generateContent", GEMINI_API_BASE, model);

    let response = http_client()
        .post(&url)
        .headers(headers)
        .json(request)
        .send()
        .await
        .map_err(|e| classify_transport_error(&e))?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(classify_http_error(status, &body));
    }

    let resp: GenerateContentResponse = response.json().await.map_err(|e| {
        if e.is_timeout() {
            GeminiError::new(GeminiErrorKind::Timeout, "réponse trop lente")
        } else {
            GeminiError::new(GeminiErrorKind::Other, format!("réponse illisible : {}", e))
        }
    })?;

    if let Some(reason) = resp.prompt_feedback.and_then(|f| f.block_reason) {
        return Err(GeminiError::new(
            GeminiErrorKind::Blocked,
            format!("requête bloquée ({})", reason),
        ));
    }

    let candidate = resp.candidates.and_then(|c| c.into_iter().next());
    let Some(candidate) = candidate else {
        return Ok(String::new());
    };

    let text: String = candidate
        .content
        .and_then(|c| c.parts)
        .unwrap_or_default()
        .into_iter()
        .filter_map(ResponsePart::into_text)
        .collect();

    let finish_reason = candidate.finish_reason.unwrap_or_default();
    match finish_reason.as_str() {
        "SAFETY" | "RECITATION" | "PROHIBITED_CONTENT" | "BLOCKLIST" | "SPII"
            if text.trim().is_empty() =>
        {
            return Err(GeminiError::new(
                GeminiErrorKind::Blocked,
                format!("réponse bloquée ({})", finish_reason),
            ));
        }
        "MAX_TOKENS" => {
            warn!("Réponse Gemini tronquée (MAX_TOKENS)");
        }
        _ => {}
    }

    Ok(text.trim().to_string())
}

pub async fn transcribe_audio(api_key: &str, model: &str, audio_samples: &[f32]) -> Result<String> {
    let wav_bytes = encode_samples_to_wav(audio_samples)?;
    let audio_base64 = base64::engine::general_purpose::STANDARD.encode(&wav_bytes);

    debug!(
        "Gemini transcribe: {} samples, {} bytes WAV, {} bytes base64",
        audio_samples.len(),
        wav_bytes.len(),
        audio_base64.len()
    );

    let mut request = GenerateContentRequest {
        contents: vec![Content {
            parts: vec![
                Part {
                    text: None,
                    inline_data: Some(InlineData {
                        mime_type: "audio/wav".to_string(),
                        data: audio_base64,
                    }),
                },
                Part {
                    text: Some(TRANSCRIPTION_PROMPT.to_string()),
                    inline_data: None,
                },
            ],
        }],
        system_instruction: None,
        generation_config: generation_config_for(model),
    };

    let text = match generate_content(api_key, model, &request).await {
        // Filet de sécurité : si Google refuse un jour ces réglages (« sans
        // réflexion », température) pour ce modèle (HTTP 400), on renvoie la
        // même requête sans eux plutôt que de faire échouer toutes les dictées.
        Err(e)
            if e.kind == GeminiErrorKind::Other
                && e.detail.starts_with("HTTP 400")
                && request.generation_config.is_some() =>
        {
            warn!("Gemini a refusé thinkingConfig, nouvel essai sans : {}", e);
            request.generation_config = None;
            generate_content(api_key, model, &request).await?
        }
        result => result?,
    };
    // Le texte dicté n'est jamais écrit dans le journal (donnée personnelle).
    debug!("Gemini transcription result length: {} chars", text.len());
    Ok(text)
}

pub async fn generate_text(
    api_key: &str,
    model: &str,
    system_prompt: &str,
    user_text: &str,
) -> Result<String> {
    debug!(
        "Gemini generate_text: model={}, prompt_len={}, text_len={}",
        model,
        system_prompt.len(),
        user_text.len()
    );

    let request = GenerateContentRequest {
        contents: vec![Content {
            parts: vec![Part {
                text: Some(user_text.to_string()),
                inline_data: None,
            }],
        }],
        system_instruction: Some(SystemInstruction {
            parts: vec![Part {
                text: Some(system_prompt.to_string()),
                inline_data: None,
            }],
        }),
        generation_config: None,
    };

    let text = generate_content(api_key, model, &request).await?;
    debug!("Gemini text generation result length: {}", text.len());
    Ok(text)
}

/// Vérifie qu'une clé API est acceptée par Google, via la liste des modèles
/// (requête gratuite, sans audio).
pub async fn check_api_key(api_key: &str) -> std::result::Result<(), GeminiError> {
    let headers = api_key_header(api_key)?;
    let response = http_client()
        .get(GEMINI_API_BASE)
        .headers(headers)
        .send()
        .await
        .map_err(|e| classify_transport_error(&e))?;

    let status = response.status();
    if status.is_success() {
        Ok(())
    } else {
        let body = response.text().await.unwrap_or_default();
        Err(classify_http_error(status, &body))
    }
}

/// Un modèle de la liste Google, réduit à ce qui nous sert.
struct ListedModel {
    /// Identifiant sans le préfixe « models/ » (ex. « gemini-2.5-flash »).
    id: String,
    /// Méthodes acceptées (« generateContent »…).
    methods: Vec<String>,
}

/// Récupère la liste des modèles accessibles avec cette clé. `pageSize` au
/// maximum : avec la valeur par défaut (50), une partie des modèles manquait.
async fn fetch_model_list(api_key: &str) -> std::result::Result<Vec<ListedModel>, GeminiError> {
    let headers = api_key_header(api_key)?;
    let response = http_client()
        .get(GEMINI_API_BASE)
        .query(&[("pageSize", "1000")])
        .headers(headers)
        .send()
        .await
        .map_err(|e| classify_transport_error(&e))?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(classify_http_error(status, &body));
    }

    let parsed: serde_json::Value = response.json().await.map_err(|e| {
        GeminiError::new(GeminiErrorKind::Other, format!("réponse illisible : {}", e))
    })?;

    Ok(parsed
        .get("models")
        .and_then(|d| d.as_array())
        .map(|entries| {
            entries
                .iter()
                .filter_map(|entry| {
                    let name = entry.get("name")?.as_str()?;
                    // Gemini renvoie « models/gemini-2.5-flash » : on retire le préfixe
                    let id = name.strip_prefix("models/").unwrap_or(name).to_string();
                    let methods = entry
                        .get("supportedGenerationMethods")
                        .and_then(|m| m.as_array())
                        .map(|m| {
                            m.iter()
                                .filter_map(|v| v.as_str().map(str::to_string))
                                .collect()
                        })
                        .unwrap_or_default();
                    Some(ListedModel { id, methods })
                })
                .collect()
        })
        .unwrap_or_default())
}

/// Liste les modèles Gemini disponibles pour cette clé (sans le préfixe
/// « models/ »).
pub async fn list_models(api_key: &str) -> std::result::Result<Vec<String>, GeminiError> {
    Ok(fetch_model_list(api_key)
        .await?
        .into_iter()
        .map(|m| m.id)
        .filter(|id| id.contains("gemini"))
        .collect())
}

/// Ce modèle peut-il transcrire une dictée ? (modèle Gemini qui accepte
/// generateContent, hors variantes spécialisées : synthèse vocale, images,
/// temps réel, robotique…)
fn is_transcription_candidate(id: &str, methods: &[String]) -> bool {
    const EXCLUDED: [&str; 10] = [
        "tts",
        "image",
        "embedding",
        "live",
        "robotics",
        "computer-use",
        "customtools",
        "banana",
        "native-audio",
        "aqa",
    ];
    id.starts_with("gemini")
        && methods.iter().any(|m| m == "generateContent")
        && !EXCLUDED.iter().any(|word| id.contains(word))
}

/// Modèles utilisables pour la transcription, triés : modèles dédiés à la
/// transcription d'abord, puis du plus récent au plus ancien.
pub async fn list_transcription_models(
    api_key: &str,
) -> std::result::Result<Vec<String>, GeminiError> {
    let mut models: Vec<String> = fetch_model_list(api_key)
        .await?
        .into_iter()
        .filter(|m| is_transcription_candidate(&m.id, &m.methods))
        .map(|m| m.id)
        .collect();
    models.sort_by(|a, b| {
        let a_transcribe = a.contains("transcribe");
        let b_transcribe = b.contains("transcribe");
        b_transcribe.cmp(&a_transcribe).then_with(|| b.cmp(a))
    });
    models.dedup();
    Ok(models)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_name_validation() {
        assert!(validate_model_name("gemini-2.5-flash").is_ok());
        assert!(validate_model_name("gemini-2.5-flash-lite").is_ok());
        assert!(validate_model_name("").is_err());
        assert!(validate_model_name("../x").is_err());
        assert!(validate_model_name("a?key=1").is_err());
    }

    #[test]
    fn transcription_prompt_is_the_tested_one() {
        assert_eq!(
            TRANSCRIPTION_PROMPT,
            "Transcribe this audio. Return only the transcript text, nothing else. The speaker may dictate French punctuation and formatting commands such as « tiret », « à la ligne », « nouveau paragraphe » or « point d'exclamation »: write these commands as the spoken words, do not apply them."
        );
    }

    #[test]
    fn thinking_disabled_only_for_flash() {
        let json = |m| serde_json::to_string(&generation_config_for(m)).unwrap();
        assert_eq!(
            json("gemini-2.5-flash"),
            r#"{"thinkingConfig":{"thinkingBudget":0},"temperature":0.0}"#
        );
        assert_eq!(
            json("gemini-2.5-flash-lite"),
            r#"{"thinkingConfig":{"thinkingBudget":0},"temperature":0.0}"#
        );
        // 2.5 Pro : réflexion obligatoire, mais température 0 quand même
        assert_eq!(json("gemini-2.5-pro"), r#"{"temperature":0.0}"#);
        // Gemini 3 : réglages par défaut, comme le recommande Google
        assert_eq!(json("gemini-3.5-transcribe"), "null");
        assert_eq!(json("gemini-3.8-flash"), "null");
    }

    #[test]
    fn http_errors_are_classified() {
        let s = |c| reqwest::StatusCode::from_u16(c).unwrap();
        assert_eq!(
            classify_http_error(s(400), r#"{"reason":"API_KEY_INVALID"}"#).kind,
            GeminiErrorKind::InvalidKey
        );
        assert_eq!(
            classify_http_error(s(400), "bad request").kind,
            GeminiErrorKind::Other
        );
        assert_eq!(
            classify_http_error(s(403), "").kind,
            GeminiErrorKind::InvalidKey
        );
        assert_eq!(classify_http_error(s(429), "").kind, GeminiErrorKind::Quota);
        assert_eq!(
            classify_http_error(s(404), "").kind,
            GeminiErrorKind::ModelNotFound
        );
        assert_eq!(
            classify_http_error(s(503), "").kind,
            GeminiErrorKind::Unavailable
        );
    }

    /// Réponses réelles (raccourcies) de l'API, relevées le 2026-10-16.
    #[test]
    fn response_text_is_read_for_flash_and_transcribe_models() {
        let texts = |json: &str| -> String {
            let resp: GenerateContentResponse = serde_json::from_str(json).unwrap();
            resp.candidates
                .unwrap()
                .into_iter()
                .next()
                .unwrap()
                .content
                .unwrap()
                .parts
                .unwrap()
                .into_iter()
                .filter_map(ResponsePart::into_text)
                .collect()
        };
        let flash = r#"{"candidates":[{"content":{"parts":[{"text":"Salut. Ça va bien?"}],"role":"model"},"finishReason":"STOP"}]}"#;
        let transcribe = r#"{"candidates":[{"content":{"parts":[{"audioTranscription":{"text":"Salut, ça va bien ?"}}],"role":"model"},"finishReason":"STOP"}]}"#;
        let thinking = r#"{"candidates":[{"content":{"parts":[{"text":"je réfléchis","thought":true},{"text":"Bonjour"}]}}]}"#;
        assert_eq!(texts(flash), "Salut. Ça va bien?");
        assert_eq!(texts(transcribe), "Salut, ça va bien ?");
        assert_eq!(texts(thinking), "Bonjour");
    }

    #[test]
    fn only_transcription_capable_models_are_offered() {
        let gen = vec!["generateContent".to_string()];
        assert!(is_transcription_candidate("gemini-3.5-transcribe", &gen));
        assert!(is_transcription_candidate("gemini-2.5-flash", &gen));
        assert!(!is_transcription_candidate(
            "gemini-2.5-flash-preview-tts",
            &gen
        ));
        assert!(!is_transcription_candidate("gemini-nano-banana-2.1", &gen));
        assert!(!is_transcription_candidate(
            "gemini-robotics-er-2-preview",
            &gen
        ));
        assert!(!is_transcription_candidate("gemma-3-27b-it", &gen));
        assert!(!is_transcription_candidate("gemini-2.5-flash", &[]));
    }

    #[test]
    fn empty_key_is_rejected_before_any_request() {
        assert_eq!(
            api_key_header("  ").unwrap_err().kind,
            GeminiErrorKind::InvalidKey
        );
    }

    #[test]
    fn wav_encoding_clamps_samples() {
        let wav = encode_samples_to_wav(&[0.0, 2.0, -2.0]).unwrap();
        // 44 octets d'en-tête + 3 échantillons de 2 octets
        assert_eq!(wav.len(), 44 + 6);
    }
}
