#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::apple_intelligence;
use crate::audio_feedback::{play_feedback_sound, play_feedback_sound_blocking, SoundType};
use crate::managers::audio::{AudioRecordingManager, StartRecordingError};
use crate::managers::history::HistoryManager;
use crate::managers::transcription::TranscriptionManager;
use crate::settings::{get_settings, AppSettings, APPLE_INTELLIGENCE_PROVIDER_ID};
use crate::shortcut;
use crate::tray::{change_tray_icon, TrayIconState};
use crate::utils::{
    self, show_processing_overlay, show_recording_overlay, show_transcribing_overlay,
};
use crate::TranscriptionCoordinator;
use ferrous_opencc::{config::BuiltinConfig, OpenCC};
use log::{debug, error, info, warn};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::AppHandle;
use tauri::Manager;

pub struct ActiveActionState(pub Mutex<Option<u8>>);

/// Drop guard that notifies the [`TranscriptionCoordinator`] when the
/// transcription pipeline finishes — whether it completes normally or panics.
struct FinishGuard(AppHandle);
impl Drop for FinishGuard {
    fn drop(&mut self) {
        if let Some(c) = self.0.try_state::<TranscriptionCoordinator>() {
            c.notify_processing_finished();
        }
    }
}

/// Moment où l'enregistrement en cours a démarré : permet de distinguer un
/// appui accidentel d'une vraie dictée restée sans résultat.
static RECORDING_STARTED_AT: Lazy<Mutex<Option<Instant>>> = Lazy::new(|| Mutex::new(None));

/// Durée d'appui au-delà de laquelle une dictée sans texte est signalée, et
/// au-delà de laquelle l'audio d'une transcription en échec est conservé dans
/// l'historique (en dessous, c'est un appui accidentel).
const MIN_MEANINGFUL_RECORDING: Duration = Duration::from_millis(1500);

/// Identifiant du « modèle » Gemini (transcription en ligne)
const GEMINI_MODEL_ID: &str = "gemini-api";

// Shortcut Action Trait
pub trait ShortcutAction: Send + Sync {
    fn start(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str);
    fn stop(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str);
}

// Transcribe Action
struct TranscribeAction {
    post_process: bool,
}

/// Field name for structured output JSON schema
const TRANSCRIPTION_FIELD: &str = "transcription";

/// Résultat d'un post-traitement par IA.
enum ProcessOutcome {
    /// Texte retouché.
    Done(String),
    /// Rien à faire (fonction non configurée) : le texte brut est utilisé.
    Skipped,
    /// L'appel a échoué : le texte brut est utilisé et l'utilisateur est prévenu.
    Failed,
}

/// Strip invisible Unicode characters that some LLMs may insert
fn strip_invisible_chars(s: &str) -> String {
    s.replace(['\u{200B}', '\u{200C}', '\u{200D}', '\u{FEFF}'], "")
}

/// Build a system prompt from the user's prompt template.
/// Removes `${output}` placeholder since the transcription is sent as the user message.
fn build_system_prompt(prompt_template: &str) -> String {
    prompt_template.replace("${output}", "").trim().to_string()
}

/// La dictée qui se termine est-elle toujours la plus récente ? `ticket` est
/// la génération de l'overlay relevée lors de son dernier affichage. Si elle a
/// changé, une nouvelle dictée a commencé entre-temps : l'ancienne ne doit plus
/// toucher ni à l'overlay ni à l'icône (sinon elle masquerait la nouvelle).
fn is_current_dictation(ticket: u64) -> bool {
    utils::current_overlay_generation() == ticket
}

/// Termine une dictée en erreur : message traduit dans l'overlay, icône au repos.
fn finish_with_error(app: &AppHandle, code: &str, ticket: u64) {
    if !is_current_dictation(ticket) {
        warn!(
            "Dictation error '{}' not shown: a newer dictation is in progress",
            code
        );
        return;
    }
    change_tray_icon(app, TrayIconState::Idle);
    utils::show_error_overlay(app, code);
}

/// Termine une dictée sans rien afficher (appui accidentel, annulation…).
fn finish_quietly(app: &AppHandle, ticket: u64) {
    if !is_current_dictation(ticket) {
        debug!("Overlay left as is: a newer dictation is in progress");
        return;
    }
    utils::hide_recording_overlay(app);
    change_tray_icon(app, TrayIconState::Idle);
}

/// Code d'erreur (traduit par l'interface) correspondant à un échec de transcription.
fn transcription_error_code(err: &anyhow::Error) -> &'static str {
    if let Some(gemini) = err.downcast_ref::<crate::gemini_client::GeminiError>() {
        return gemini.kind.code();
    }
    let message = err.to_string();
    if message.contains("not loaded")
        || message.contains("failed to load")
        || message.contains("Model not")
    {
        "model_not_loaded"
    } else {
        "transcription_failed"
    }
}

/// Enregistre une dictée dans l'historique, en tâche de fond.
#[allow(clippy::too_many_arguments)]
fn save_to_history(
    hm: Arc<HistoryManager>,
    samples: Vec<f32>,
    transcription: String,
    post_processed_text: Option<String>,
    post_process_prompt: Option<String>,
    action_key: Option<u8>,
    model_name: Option<String>,
) {
    tauri::async_runtime::spawn(async move {
        if let Err(e) = hm
            .save_transcription(
                samples,
                transcription,
                post_processed_text,
                post_process_prompt,
                action_key,
                model_name,
            )
            .await
        {
            error!("Failed to save transcription to history: {}", e);
        }
    });
}

async fn post_process_transcription(settings: &AppSettings, transcription: &str) -> ProcessOutcome {
    let provider = match settings.active_post_process_provider().cloned() {
        Some(provider) => provider,
        None => {
            debug!("Post-processing enabled but no provider is selected");
            return ProcessOutcome::Skipped;
        }
    };

    let model = settings
        .post_process_models
        .get(&provider.id)
        .cloned()
        .unwrap_or_default();

    if model.trim().is_empty() {
        debug!(
            "Post-processing skipped because provider '{}' has no model configured",
            provider.id
        );
        return ProcessOutcome::Skipped;
    }

    let selected_prompt_id = match &settings.post_process_selected_prompt_id {
        Some(id) => id.clone(),
        None => {
            debug!("Post-processing skipped because no prompt is selected");
            return ProcessOutcome::Skipped;
        }
    };

    let prompt = match settings
        .post_process_prompts
        .iter()
        .find(|prompt| prompt.id == selected_prompt_id)
    {
        Some(prompt) => prompt.prompt.clone(),
        None => {
            debug!(
                "Post-processing skipped because prompt '{}' was not found",
                selected_prompt_id
            );
            return ProcessOutcome::Skipped;
        }
    };

    if prompt.trim().is_empty() {
        debug!("Post-processing skipped because the selected prompt is empty");
        return ProcessOutcome::Skipped;
    }

    debug!(
        "Starting LLM post-processing with provider '{}' (model: {})",
        provider.id, model
    );

    let api_key = settings
        .post_process_api_keys
        .get(&provider.id)
        .cloned()
        .unwrap_or_default();

    if provider.supports_structured_output {
        debug!("Using structured outputs for provider '{}'", provider.id);

        let system_prompt = build_system_prompt(&prompt);
        let user_content = transcription.to_string();

        // Handle Apple Intelligence separately since it uses native Swift APIs
        if provider.id == APPLE_INTELLIGENCE_PROVIDER_ID {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            {
                if !apple_intelligence::check_apple_intelligence_availability() {
                    debug!(
                        "Apple Intelligence selected but not currently available on this device"
                    );
                    return ProcessOutcome::Skipped;
                }

                let token_limit = model.trim().parse::<i32>().unwrap_or(0);
                return match apple_intelligence::process_text_with_system_prompt(
                    &system_prompt,
                    &user_content,
                    token_limit,
                ) {
                    Ok(result) => {
                        if result.trim().is_empty() {
                            debug!("Apple Intelligence returned an empty response");
                            ProcessOutcome::Failed
                        } else {
                            let result = strip_invisible_chars(&result);
                            debug!(
                                "Apple Intelligence post-processing succeeded. Output length: {} chars",
                                result.len()
                            );
                            ProcessOutcome::Done(result)
                        }
                    }
                    Err(err) => {
                        error!("Apple Intelligence post-processing failed: {}", err);
                        ProcessOutcome::Failed
                    }
                };
            }

            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            {
                debug!("Apple Intelligence provider selected on unsupported platform");
                return ProcessOutcome::Skipped;
            }
        }

        // Define JSON schema for transcription output
        let json_schema = serde_json::json!({
            "type": "object",
            "properties": {
                (TRANSCRIPTION_FIELD): {
                    "type": "string",
                    "description": "The cleaned and processed transcription text"
                }
            },
            "required": [TRANSCRIPTION_FIELD],
            "additionalProperties": false
        });

        match crate::llm_client::send_chat_completion_with_schema(
            &provider,
            api_key.clone(),
            &model,
            user_content,
            Some(system_prompt),
            Some(json_schema),
        )
        .await
        {
            Ok(Some(content)) => {
                // Parse the JSON response to extract the transcription field
                return match serde_json::from_str::<serde_json::Value>(&content) {
                    Ok(json) => {
                        if let Some(transcription_value) =
                            json.get(TRANSCRIPTION_FIELD).and_then(|t| t.as_str())
                        {
                            let result = strip_invisible_chars(transcription_value);
                            debug!(
                                "Structured output post-processing succeeded for provider '{}'. Output length: {} chars",
                                provider.id,
                                result.len()
                            );
                            ProcessOutcome::Done(result)
                        } else {
                            error!("Structured output response missing 'transcription' field");
                            ProcessOutcome::Done(strip_invisible_chars(&content))
                        }
                    }
                    Err(e) => {
                        error!(
                            "Failed to parse structured output JSON: {}. Returning raw content.",
                            e
                        );
                        ProcessOutcome::Done(strip_invisible_chars(&content))
                    }
                };
            }
            Ok(None) => {
                error!("LLM API response has no content");
                return ProcessOutcome::Failed;
            }
            Err(e) => {
                warn!(
                    "Structured output failed for provider '{}': {}. Falling back to legacy mode.",
                    provider.id, e
                );
                // Fall through to legacy mode below
            }
        }
    }

    // Legacy mode: Replace ${output} variable in the prompt with the actual text
    let processed_prompt = prompt.replace("${output}", transcription);
    debug!("Processed prompt length: {} chars", processed_prompt.len());

    match crate::llm_client::send_chat_completion(&provider, api_key, &model, processed_prompt)
        .await
    {
        Ok(Some(content)) => {
            let content = strip_invisible_chars(&content);
            debug!(
                "LLM post-processing succeeded for provider '{}'. Output length: {} chars",
                provider.id,
                content.len()
            );
            ProcessOutcome::Done(content)
        }
        Ok(None) => {
            error!("LLM API response has no content");
            ProcessOutcome::Failed
        }
        Err(e) => {
            error!(
                "LLM post-processing failed for provider '{}': {}. Falling back to original transcription.",
                provider.id,
                e
            );
            ProcessOutcome::Failed
        }
    }
}

async fn process_action(
    settings: &AppSettings,
    transcription: &str,
    prompt: &str,
    action_model: Option<&str>,
    action_provider_id: Option<&str>,
) -> ProcessOutcome {
    let provider = match action_provider_id
        .filter(|p| !p.is_empty())
        .and_then(|pid| {
            let found = settings.post_process_provider(pid).cloned();
            if found.is_none() {
                debug!(
                    "Action provider '{}' not found, falling back to active provider",
                    pid
                );
            }
            found
        })
        .or_else(|| settings.active_post_process_provider().cloned())
    {
        Some(p) => p,
        None => {
            debug!("Action processing skipped: no provider configured");
            return ProcessOutcome::Skipped;
        }
    };

    let model = action_model
        .filter(|m| !m.trim().is_empty())
        .map(|m| m.to_string())
        .or_else(|| settings.post_process_models.get(&provider.id).cloned())
        .unwrap_or_default();

    let full_prompt = if prompt.contains("${output}") {
        prompt.replace("${output}", transcription)
    } else {
        format!("{}\n\n{}", prompt, transcription)
    };

    debug!(
        "Starting action processing with provider '{}', model '{}', prompt length: {}",
        provider.id,
        model,
        full_prompt.len()
    );

    // Handle Apple Intelligence via native Swift APIs
    if provider.id == APPLE_INTELLIGENCE_PROVIDER_ID {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            if !apple_intelligence::check_apple_intelligence_availability() {
                debug!("Apple Intelligence selected but not available for action processing");
                return ProcessOutcome::Skipped;
            }
            let token_limit = model.trim().parse::<i32>().unwrap_or(0);
            return match apple_intelligence::process_text_with_system_prompt(
                &full_prompt,
                transcription,
                token_limit,
            ) {
                Ok(result) if !result.trim().is_empty() => {
                    let result = strip_invisible_chars(&result);
                    debug!(
                        "Apple Intelligence action processing succeeded. Output length: {} chars",
                        result.len()
                    );
                    ProcessOutcome::Done(result)
                }
                Ok(_) => {
                    debug!("Apple Intelligence action returned empty result");
                    ProcessOutcome::Failed
                }
                Err(err) => {
                    error!("Apple Intelligence action processing failed: {}", err);
                    ProcessOutcome::Failed
                }
            };
        }

        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            debug!("Apple Intelligence provider selected on unsupported platform");
            return ProcessOutcome::Skipped;
        }
    }

    if model.trim().is_empty() {
        debug!(
            "Action processing skipped: no model configured for provider '{}'",
            provider.id
        );
        return ProcessOutcome::Skipped;
    }

    let api_key = settings
        .post_process_api_keys
        .get(&provider.id)
        .cloned()
        .unwrap_or_default();

    let system_prompt = "You are a text processing assistant. Output ONLY the final processed text. Do not add any explanation, commentary, preamble, or formatting such as markdown code blocks. Just output the raw result text, nothing else.".to_string();

    match crate::llm_client::send_chat_completion_with_schema(
        &provider,
        api_key,
        &model,
        full_prompt,
        Some(system_prompt),
        None,
    )
    .await
    {
        Ok(Some(content)) if !content.is_empty() => {
            let result = strip_invisible_chars(&content);
            debug!(
                "Action processing succeeded for provider '{}'. Output length: {} chars",
                provider.id,
                result.len()
            );
            ProcessOutcome::Done(result)
        }
        Ok(_) => {
            debug!("Action processing returned empty result");
            ProcessOutcome::Failed
        }
        Err(e) => {
            error!(
                "Action processing failed for provider '{}': {}",
                provider.id, e
            );
            ProcessOutcome::Failed
        }
    }
}

async fn maybe_convert_chinese_variant(
    settings: &AppSettings,
    transcription: &str,
) -> Option<String> {
    // Check if language is set to Simplified or Traditional Chinese
    let is_simplified = settings.selected_language == "zh-Hans";
    let is_traditional = settings.selected_language == "zh-Hant";

    if !is_simplified && !is_traditional {
        debug!("selected_language is not Simplified or Traditional Chinese; skipping translation");
        return None;
    }

    debug!(
        "Starting Chinese translation using OpenCC for language: {}",
        settings.selected_language
    );

    // Use OpenCC to convert based on selected language
    let config = if is_simplified {
        // Convert Traditional Chinese to Simplified Chinese
        BuiltinConfig::Tw2sp
    } else {
        // Convert Simplified Chinese to Traditional Chinese
        BuiltinConfig::S2twp
    };

    match OpenCC::from_config(config) {
        Ok(converter) => {
            let converted = converter.convert(transcription);
            debug!(
                "OpenCC translation completed. Input length: {}, Output length: {}",
                transcription.len(),
                converted.len()
            );
            Some(converted)
        }
        Err(e) => {
            error!("Failed to initialize OpenCC converter: {}. Falling back to original transcription.", e);
            None
        }
    }
}

/// Retranscrit en local quand Gemini a échoué ou n'a rien rendu : le modèle
/// principal s'il est local, sinon le meilleur modèle local téléchargé pour la
/// langue dictée. None si aucun modèle local n'a pu servir.
fn retry_locally(
    tm: &TranscriptionManager,
    main: &str,
    samples: Vec<f32>,
) -> Option<anyhow::Result<String>> {
    let fallback = if main == GEMINI_MODEL_ID {
        tm.local_fallback_model()?
    } else {
        main.to_string()
    };
    if let Err(e) = tm.load_model(&fallback) {
        warn!(
            "Failed to load local model '{}' for fallback: {}",
            fallback, e
        );
        return None;
    }
    info!("Retranscription with local model: {}", fallback);
    Some(tm.transcribe(samples))
}

/// Transcrit l'audio, en basculant au besoin sur le modèle « audio long » (et en
/// le réessayant si le modèle courant ne renvoie rien sur un audio conséquent).
/// Renvoie le résultat et indique si le modèle a été changé (à restaurer).
fn transcribe_with_fallback(
    tm: &TranscriptionManager,
    samples: Vec<f32>,
    settings: &AppSettings,
    original_model: Option<&str>,
) -> (anyhow::Result<String>, bool) {
    let duration_seconds = samples.len() as f32 / 16000.0;
    let mut switched_model = false;

    if let Some(ref long_model_id) = settings.long_audio_model {
        if duration_seconds > settings.long_audio_threshold_seconds
            && original_model != Some(long_model_id.as_str())
        {
            debug!(
                "Audio duration {:.1}s exceeds threshold {:.1}s, switching to long audio model: {}",
                duration_seconds, settings.long_audio_threshold_seconds, long_model_id
            );
            if let Err(e) = tm.load_model(long_model_id) {
                warn!(
                    "Failed to load long audio model '{}': {}, using current model",
                    long_model_id, e
                );
            } else {
                switched_model = true;
            }
        }
    }

    let transcription_time = Instant::now();
    // Copie gardée pour un nouvel essai (modèle local, modèle « audio long »)
    let retry_samples = samples.clone();
    let result = tm.transcribe(samples);

    // Modèle principal encore en chargement (None) : celui des réglages
    let main = original_model.unwrap_or(settings.selected_model.as_str());
    // Gemini vient-il de transcrire ? Sans clé, il ne se charge pas : c'est
    // alors le modèle principal des réglages qui le dit.
    let used_gemini = match tm.get_current_model() {
        Some(id) => id == GEMINI_MODEL_ID,
        None => main == GEMINI_MODEL_ID,
    };

    let transcription = match result {
        Ok(text) => text,
        // Gemini a échoué — crédits épuisés, quota, modèle retiré, réseau
        // coupé : on retente en local plutôt que de perdre la dictée
        // (constaté le 2026-10-27 : HTTP 402, puis 404). Si le local échoue
        // aussi, c'est l'erreur de Gemini qui est signalée (elle dit quoi faire).
        Err(e) if used_gemini => {
            warn!("Gemini failed ({}), retrying with a local model", e);
            let local = retry_locally(tm, main, retry_samples);
            let restore = tm.get_current_model().as_deref() != Some(main);
            return match local {
                Some(Ok(text)) => (Ok(text), restore),
                Some(Err(local_err)) => {
                    warn!("Local model fallback failed too: {}", local_err);
                    (Err(e), restore)
                }
                None => (Err(e), restore),
            };
        }
        Err(e) => return (Err(e), switched_model),
    };
    // Le texte dicté n'est jamais écrit dans le journal (donnée personnelle).
    debug!(
        "Transcription completed in {:?} ({} chars)",
        transcription_time.elapsed(),
        transcription.len()
    );

    // Gemini n'a rien rendu sur un vrai enregistrement (réponse sans texte) :
    // le local réessaie, sinon la dictée serait perdue (rien n'est gardé dans
    // l'historique quand le texte est vide).
    if transcription.is_empty() && duration_seconds > 1.0 && used_gemini {
        info!(
            "Gemini returned nothing for {:.1}s audio, retrying with a local model",
            duration_seconds
        );
        let local = retry_locally(tm, main, retry_samples);
        let restore = tm.get_current_model().as_deref() != Some(main);
        return match local {
            Some(Ok(text)) => (Ok(text), restore),
            Some(Err(local_err)) => {
                warn!("Local model fallback error: {}", local_err);
                (Ok(transcription), restore)
            }
            None => (Ok(transcription), restore),
        };
    }

    // Fallback: if cheap model returned nothing on meaningful audio, retry with accurate model
    if transcription.is_empty() && duration_seconds > 1.0 && !switched_model {
        if let Some(long_model_id) = settings.long_audio_model.as_ref() {
            if original_model != Some(long_model_id.as_str()) {
                info!(
                    "Transcription empty for {:.1}s audio, retrying with long audio model: {}",
                    duration_seconds, long_model_id
                );
                match tm.load_model(long_model_id) {
                    Ok(()) => {
                        switched_model = true;
                        match tm.transcribe(retry_samples) {
                            Ok(retry_result) => {
                                debug!(
                                    "Fallback transcription returned {} chars",
                                    retry_result.len()
                                );
                                return (Ok(retry_result), switched_model);
                            }
                            Err(e) => warn!("Fallback transcription error: {}", e),
                        }
                    }
                    Err(e) => warn!("Failed to load long audio model for fallback: {}", e),
                }
            }
        }
    }

    (Ok(transcription), switched_model)
}

impl ShortcutAction for TranscribeAction {
    fn start(&self, app: &AppHandle, binding_id: &str, _shortcut_str: &str) {
        let start_time = Instant::now();
        debug!("TranscribeAction::start called for binding: {}", binding_id);

        // Load model in the background
        let tm = app.state::<Arc<TranscriptionManager>>();
        tm.initiate_model_load();

        let binding_id = binding_id.to_string();
        change_tray_icon(app, TrayIconState::Recording);
        show_recording_overlay(app);

        let rm = app.state::<Arc<AudioRecordingManager>>();

        // Get the microphone mode to determine audio feedback timing
        let settings = get_settings(app);
        let is_always_on = settings.always_on_microphone;
        debug!("Microphone mode - always_on: {}", is_always_on);

        let recording_start_time = Instant::now();
        let start_result = rm.try_start_recording(&binding_id);

        match &start_result {
            Ok(()) => {
                debug!("Recording started in {:?}", recording_start_time.elapsed());
                if let Ok(mut started_at) = RECORDING_STARTED_AT.lock() {
                    *started_at = Some(Instant::now());
                }
                let app_clone = app.clone();
                let rm_clone = Arc::clone(&rm);
                // The blocking helper exits immediately if audio feedback is disabled,
                // so we can always reuse this thread to ensure mute happens right after playback.
                std::thread::spawn(move || {
                    if !is_always_on {
                        // Small delay to ensure microphone stream is active
                        std::thread::sleep(Duration::from_millis(100));
                    }
                    play_feedback_sound_blocking(&app_clone, SoundType::Start);
                    rm_clone.apply_mute();
                });

                shortcut::register_cancel_shortcut(app);
                shortcut::register_pause_shortcut(app);
                shortcut::register_action_shortcuts(app);
            }
            Err(StartRecordingError::Busy) => {
                debug!("Recording not started: a recording is already in progress");
            }
            Err(StartRecordingError::Microphone(e)) => {
                error!("Failed to start recording: {}", e);
                // L'overlay et l'icône « enregistrement » ont été affichés avant
                // de savoir si le micro démarrerait : on les remplace par l'erreur.
                finish_with_error(app, "mic_unavailable", utils::current_overlay_generation());
            }
        }

        debug!(
            "TranscribeAction::start completed in {:?}",
            start_time.elapsed()
        );
    }

    fn stop(&self, app: &AppHandle, binding_id: &str, _shortcut_str: &str) {
        crate::shortcut::handler::reset_cancel_confirmation();
        shortcut::unregister_cancel_shortcut(app);
        shortcut::unregister_pause_shortcut(app);
        shortcut::unregister_action_shortcuts(app);

        let stop_time = Instant::now();
        debug!("TranscribeAction::stop called for binding: {}", binding_id);

        let recording_duration = RECORDING_STARTED_AT
            .lock()
            .ok()
            .and_then(|mut started_at| started_at.take())
            .map(|t| t.elapsed())
            .unwrap_or_default();

        let ah = app.clone();
        let rm = Arc::clone(&app.state::<Arc<AudioRecordingManager>>());
        let tm = Arc::clone(&app.state::<Arc<TranscriptionManager>>());
        let hm = Arc::clone(&app.state::<Arc<HistoryManager>>());

        change_tray_icon(app, TrayIconState::Transcribing);
        show_transcribing_overlay(app);
        let mut overlay_ticket = utils::current_overlay_generation();

        // Unmute before playing audio feedback so the stop sound is audible
        rm.remove_mute();

        // Play audio feedback for recording stop
        play_feedback_sound(app, SoundType::Stop);

        let binding_id = binding_id.to_string(); // Clone binding_id for the async task
        let post_process = self.post_process;

        // Read and clear the selected action before spawning the async task
        let selected_action_key =
            app.try_state::<ActiveActionState>()
                .and_then(|s| match s.0.lock() {
                    Ok(mut guard) => guard.take(),
                    Err(poisoned) => {
                        error!("ActiveActionState mutex poisoned, recovering");
                        poisoned.into_inner().take()
                    }
                });

        tauri::async_runtime::spawn(async move {
            let _guard = FinishGuard(ah.clone());
            debug!(
                "Starting async transcription task for binding: {}, action: {:?}",
                binding_id, selected_action_key
            );

            let stop_recording_time = Instant::now();
            let Some(samples) = rm.stop_recording(&binding_id) else {
                debug!("No samples retrieved from recording stop");
                finish_quietly(&ah, overlay_ticket);
                return;
            };
            debug!(
                "Recording stopped and samples retrieved in {:?}, sample count: {}",
                stop_recording_time.elapsed(),
                samples.len()
            );

            let settings = get_settings(&ah);
            let original_model = tm.get_current_model();
            let history_samples = samples.clone();

            let (result, switched_model) =
                transcribe_with_fallback(&tm, samples, &settings, original_model.as_deref());

            match result {
                Ok(transcription) if transcription.is_empty() => {
                    // Rien d'exploitable : appui accidentel, ou vraie dictée sans
                    // parole détectée (micro coupé, mauvais micro…) à signaler.
                    // Rien n'est ajouté à l'historique : des appuis accidentels
                    // y remplaceraient les vraies dictées (limite de 5 entrées).
                    if recording_duration >= MIN_MEANINGFUL_RECORDING {
                        finish_with_error(&ah, "no_speech", overlay_ticket);
                    } else {
                        finish_quietly(&ah, overlay_ticket);
                    }
                }
                Ok(transcription) => {
                    let mut final_text = transcription.clone();
                    let mut post_processed_text: Option<String> = None;
                    let mut post_process_prompt: Option<String> = None;

                    // First, check if Chinese variant conversion is needed
                    if let Some(converted_text) =
                        maybe_convert_chinese_variant(&settings, &transcription).await
                    {
                        final_text = converted_text;
                    }

                    let selected_action = selected_action_key.and_then(|key| {
                        settings
                            .post_process_actions
                            .iter()
                            .find(|a| a.key == key)
                            .cloned()
                    });

                    let post_process_requested = selected_action.is_some() || post_process;
                    if post_process_requested {
                        show_processing_overlay(&ah);
                        overlay_ticket = utils::current_overlay_generation();
                    }

                    // Action processing takes priority over default post-processing
                    let outcome = if let Some(ref action) = selected_action {
                        process_action(
                            &settings,
                            &final_text,
                            &action.prompt,
                            action.model.as_deref(),
                            action.provider_id.as_deref(),
                        )
                        .await
                    } else if post_process {
                        post_process_transcription(&settings, &final_text).await
                    } else {
                        ProcessOutcome::Skipped
                    };

                    let post_process_failed = matches!(outcome, ProcessOutcome::Failed);
                    // Retouche demandée (raccourci dédié ou action) mais rien de
                    // configuré : texte brut collé, on le signale.
                    let post_process_skipped =
                        post_process_requested && matches!(outcome, ProcessOutcome::Skipped);
                    if let ProcessOutcome::Done(processed_text) = outcome {
                        post_processed_text = Some(processed_text.clone());
                        final_text = processed_text;

                        if let Some(ref action) = selected_action {
                            post_process_prompt = Some(action.prompt.clone());
                        } else if let Some(prompt_id) = &settings.post_process_selected_prompt_id {
                            post_process_prompt = settings
                                .post_process_prompts
                                .iter()
                                .find(|p| &p.id == prompt_id)
                                .map(|p| p.prompt.clone());
                        }
                    } else if final_text != transcription {
                        // Chinese conversion was applied but no LLM post-processing
                        post_processed_text = Some(final_text.clone());
                    }

                    // Paste the final text (either processed or original)
                    let ah_clone = ah.clone();
                    let paste_time = Instant::now();
                    ah.run_on_main_thread(move || {
                        let pasted = match utils::paste(final_text, ah_clone.clone()) {
                            Ok(()) => {
                                debug!("Text pasted successfully in {:?}", paste_time.elapsed());
                                true
                            }
                            Err(e) => {
                                error!("Failed to paste transcription: {}", e);
                                false
                            }
                        };
                        if !pasted {
                            finish_with_error(&ah_clone, "paste_failed", overlay_ticket);
                        } else if post_process_failed {
                            // Texte brut collé : on prévient que la retouche IA a échoué.
                            finish_with_error(&ah_clone, "post_process_failed", overlay_ticket);
                        } else if post_process_skipped {
                            finish_with_error(
                                &ah_clone,
                                "post_process_not_configured",
                                overlay_ticket,
                            );
                        } else {
                            finish_quietly(&ah_clone, overlay_ticket);
                        }
                    })
                    .unwrap_or_else(|e| {
                        error!("Failed to run paste on main thread: {:?}", e);
                        finish_with_error(&ah, "paste_failed", overlay_ticket);
                    });

                    let action_key_for_history = if post_processed_text.is_some() {
                        selected_action_key
                    } else {
                        None
                    };
                    save_to_history(
                        Arc::clone(&hm),
                        history_samples,
                        transcription,
                        post_processed_text,
                        post_process_prompt,
                        action_key_for_history,
                        tm.get_current_model_name(),
                    );
                }
                Err(err) => {
                    error!("Transcription failed: {:#}", err);
                    finish_with_error(&ah, transcription_error_code(&err), overlay_ticket);
                    // L'audio d'une vraie dictée est gardé dans l'historique pour
                    // pouvoir relancer la transcription une fois le problème réglé.
                    if recording_duration >= MIN_MEANINGFUL_RECORDING {
                        save_to_history(
                            Arc::clone(&hm),
                            history_samples,
                            String::new(),
                            None,
                            None,
                            None,
                            tm.get_current_model_name(),
                        );
                    }
                }
            }

            // Restore original model if we switched for long audio
            if switched_model {
                // Sans modèle chargé au départ, c'est celui des réglages
                let restore_id = original_model
                    .clone()
                    .or(Some(settings.selected_model.clone()));
                if let Some(ref orig_id) = restore_id {
                    debug!("Restoring original model: {}", orig_id);
                    if let Err(e) = tm.load_model(orig_id) {
                        warn!("Failed to restore original model '{}': {}", orig_id, e);
                    }
                }
            }
        });

        debug!(
            "TranscribeAction::stop completed in {:?}",
            stop_time.elapsed()
        );
    }
}

// Cancel Action
struct CancelAction;

impl ShortcutAction for CancelAction {
    fn start(&self, app: &AppHandle, _binding_id: &str, _shortcut_str: &str) {
        utils::cancel_current_operation(app);
    }

    fn stop(&self, _app: &AppHandle, _binding_id: &str, _shortcut_str: &str) {
        // Nothing to do on stop for cancel
    }
}

// Test Action
struct TestAction;

impl ShortcutAction for TestAction {
    fn start(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str) {
        log::info!(
            "Shortcut ID '{}': Started - {} (App: {})", // Changed "Pressed" to "Started" for consistency
            binding_id,
            shortcut_str,
            app.package_info().name
        );
    }

    fn stop(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str) {
        log::info!(
            "Shortcut ID '{}': Stopped - {} (App: {})", // Changed "Released" to "Stopped" for consistency
            binding_id,
            shortcut_str,
            app.package_info().name
        );
    }
}

// Static Action Map
pub static ACTION_MAP: Lazy<HashMap<String, Arc<dyn ShortcutAction>>> = Lazy::new(|| {
    let mut map = HashMap::new();
    map.insert(
        "transcribe".to_string(),
        Arc::new(TranscribeAction {
            post_process: false,
        }) as Arc<dyn ShortcutAction>,
    );
    map.insert(
        "transcribe_with_post_process".to_string(),
        Arc::new(TranscribeAction { post_process: true }) as Arc<dyn ShortcutAction>,
    );
    map.insert(
        "cancel".to_string(),
        Arc::new(CancelAction) as Arc<dyn ShortcutAction>,
    );
    map.insert(
        "test".to_string(),
        Arc::new(TestAction) as Arc<dyn ShortcutAction>,
    );
    map
});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemini_client::GeminiError;

    #[test]
    fn error_codes_for_transcription_failures() {
        let missing_key = anyhow::Error::new(GeminiError::missing_key());
        assert_eq!(transcription_error_code(&missing_key), "gemini_invalid_key");

        let not_loaded = anyhow::anyhow!("Model is not loaded for transcription.");
        assert_eq!(transcription_error_code(&not_loaded), "model_not_loaded");

        let other = anyhow::anyhow!("boom");
        assert_eq!(transcription_error_code(&other), "transcription_failed");
    }
}
