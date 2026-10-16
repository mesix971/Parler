use log::{debug, error, warn};
use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use specta::Type;
use std::collections::HashMap;
use tauri::{AppHandle, Manager};
use tauri_plugin_store::StoreExt;

pub const APPLE_INTELLIGENCE_PROVIDER_ID: &str = "apple_intelligence";
pub const APPLE_INTELLIGENCE_DEFAULT_MODEL_ID: &str = "Apple Intelligence";

#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq, Type)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

// Custom deserializer to handle both old numeric format (1-5) and new string format ("trace", "debug", etc.)
impl<'de> Deserialize<'de> for LogLevel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct LogLevelVisitor;

        impl<'de> Visitor<'de> for LogLevelVisitor {
            type Value = LogLevel;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a string or integer representing log level")
            }

            fn visit_str<E: de::Error>(self, value: &str) -> Result<LogLevel, E> {
                match value.to_lowercase().as_str() {
                    "trace" => Ok(LogLevel::Trace),
                    "debug" => Ok(LogLevel::Debug),
                    "info" => Ok(LogLevel::Info),
                    "warn" => Ok(LogLevel::Warn),
                    "error" => Ok(LogLevel::Error),
                    _ => Err(E::unknown_variant(
                        value,
                        &["trace", "debug", "info", "warn", "error"],
                    )),
                }
            }

            fn visit_u64<E: de::Error>(self, value: u64) -> Result<LogLevel, E> {
                match value {
                    1 => Ok(LogLevel::Trace),
                    2 => Ok(LogLevel::Debug),
                    3 => Ok(LogLevel::Info),
                    4 => Ok(LogLevel::Warn),
                    5 => Ok(LogLevel::Error),
                    _ => Err(E::invalid_value(de::Unexpected::Unsigned(value), &"1-5")),
                }
            }
        }

        deserializer.deserialize_any(LogLevelVisitor)
    }
}

impl From<LogLevel> for tauri_plugin_log::LogLevel {
    fn from(level: LogLevel) -> Self {
        match level {
            LogLevel::Trace => tauri_plugin_log::LogLevel::Trace,
            LogLevel::Debug => tauri_plugin_log::LogLevel::Debug,
            LogLevel::Info => tauri_plugin_log::LogLevel::Info,
            LogLevel::Warn => tauri_plugin_log::LogLevel::Warn,
            LogLevel::Error => tauri_plugin_log::LogLevel::Error,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Type)]
pub struct ShortcutBinding {
    pub id: String,
    pub name: String,
    pub description: String,
    pub default_binding: String,
    pub current_binding: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, Type)]
pub struct LLMPrompt {
    pub id: String,
    pub name: String,
    pub prompt: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, Type)]
pub struct PostProcessAction {
    pub key: u8,
    pub name: String,
    pub prompt: String,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub provider_id: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Type)]
pub struct SavedProcessingModel {
    pub id: String,
    pub provider_id: String,
    pub model_id: String,
    pub label: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, Type)]
pub struct PostProcessProvider {
    pub id: String,
    pub label: String,
    pub base_url: String,
    #[serde(default)]
    pub allow_base_url_edit: bool,
    #[serde(default)]
    pub models_endpoint: Option<String>,
    #[serde(default)]
    pub supports_structured_output: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Type)]
#[serde(rename_all = "lowercase")]
pub enum OverlayPosition {
    None,
    Top,
    Bottom,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Type, Default)]
#[serde(rename_all = "snake_case")]
pub enum ModelUnloadTimeout {
    #[default]
    Never,
    Immediately,
    Min2,
    Min5,
    Min10,
    Min15,
    Hour1,
    Sec5, // Debug mode only
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Type)]
#[serde(rename_all = "snake_case")]
pub enum PasteMethod {
    CtrlV,
    Direct,
    None,
    ShiftInsert,
    CtrlShiftV,
    ExternalScript,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Type, Default)]
#[serde(rename_all = "snake_case")]
pub enum ClipboardHandling {
    #[default]
    DontModify,
    CopyToClipboard,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Type, Default)]
#[serde(rename_all = "snake_case")]
pub enum AutoSubmitKey {
    #[default]
    Enter,
    CtrlEnter,
    CmdEnter,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Type)]
#[serde(rename_all = "snake_case")]
pub enum RecordingRetentionPeriod {
    Never,
    PreserveLimit,
    Days3,
    Weeks2,
    Months3,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Type)]
#[serde(rename_all = "snake_case")]
pub enum KeyboardImplementation {
    Tauri,
    HandyKeys,
}

impl Default for KeyboardImplementation {
    fn default() -> Self {
        // Default to HandyKeys only on macOS where it's well-tested.
        // Windows and Linux use Tauri by default (handy-keys not sufficiently tested yet).
        #[cfg(target_os = "macos")]
        return KeyboardImplementation::HandyKeys;
        #[cfg(not(target_os = "macos"))]
        return KeyboardImplementation::Tauri;
    }
}

impl Default for PasteMethod {
    fn default() -> Self {
        // Default to CtrlV for macOS and Windows, Direct for Linux
        #[cfg(target_os = "linux")]
        return PasteMethod::Direct;
        #[cfg(not(target_os = "linux"))]
        return PasteMethod::CtrlV;
    }
}

impl ModelUnloadTimeout {
    pub fn to_minutes(self) -> Option<u64> {
        match self {
            ModelUnloadTimeout::Never => None,
            ModelUnloadTimeout::Immediately => Some(0), // Special case for immediate unloading
            ModelUnloadTimeout::Min2 => Some(2),
            ModelUnloadTimeout::Min5 => Some(5),
            ModelUnloadTimeout::Min10 => Some(10),
            ModelUnloadTimeout::Min15 => Some(15),
            ModelUnloadTimeout::Hour1 => Some(60),
            ModelUnloadTimeout::Sec5 => Some(0), // Special case for debug - handled separately
        }
    }

    pub fn to_seconds(self) -> Option<u64> {
        match self {
            ModelUnloadTimeout::Never => None,
            ModelUnloadTimeout::Immediately => Some(0), // Special case for immediate unloading
            ModelUnloadTimeout::Sec5 => Some(5),
            _ => self.to_minutes().map(|m| m * 60),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Type)]
#[serde(rename_all = "snake_case")]
pub enum SoundTheme {
    Marimba,
    Pop,
    Custom,
}

impl SoundTheme {
    fn as_str(&self) -> &'static str {
        match self {
            SoundTheme::Marimba => "marimba",
            SoundTheme::Pop => "pop",
            SoundTheme::Custom => "custom",
        }
    }

    pub fn to_start_path(self) -> String {
        format!("resources/{}_start.wav", self.as_str())
    }

    pub fn to_stop_path(self) -> String {
        format!("resources/{}_stop.wav", self.as_str())
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Type, Default)]
#[serde(rename_all = "snake_case")]
pub enum TypingTool {
    #[default]
    Auto,
    Wtype,
    Kwtype,
    Dotool,
    Ydotool,
    Xdotool,
}

/* still handy for composing the initial JSON in the store ------------- */
// `Debug` est implémenté à la main plus bas pour masquer les clés API : un
// `{:?}` sur les réglages ne doit jamais écrire un secret dans le journal.
#[derive(Serialize, Deserialize, Clone, Type)]
pub struct AppSettings {
    pub bindings: HashMap<String, ShortcutBinding>,
    pub push_to_talk: bool,
    pub audio_feedback: bool,
    #[serde(default = "default_audio_feedback_volume")]
    pub audio_feedback_volume: f32,
    #[serde(default = "default_sound_theme")]
    pub sound_theme: SoundTheme,
    #[serde(default = "default_start_hidden")]
    pub start_hidden: bool,
    #[serde(default = "default_autostart_enabled")]
    pub autostart_enabled: bool,
    #[serde(default = "default_update_checks_enabled")]
    pub update_checks_enabled: bool,
    #[serde(default = "default_model")]
    pub selected_model: String,
    #[serde(default = "default_always_on_microphone")]
    pub always_on_microphone: bool,
    #[serde(default)]
    pub selected_microphone: Option<String>,
    #[serde(default)]
    pub clamshell_microphone: Option<String>,
    #[serde(default)]
    pub selected_output_device: Option<String>,
    #[serde(default = "default_translate_to_english")]
    pub translate_to_english: bool,
    #[serde(default = "default_selected_language")]
    pub selected_language: String,
    #[serde(default = "default_overlay_position")]
    pub overlay_position: OverlayPosition,
    #[serde(default = "default_debug_mode")]
    pub debug_mode: bool,
    #[serde(default = "default_log_level")]
    pub log_level: LogLevel,
    #[serde(default)]
    pub custom_words: Vec<String>,
    #[serde(default)]
    pub model_unload_timeout: ModelUnloadTimeout,
    #[serde(default = "default_word_correction_threshold")]
    pub word_correction_threshold: f64,
    #[serde(default = "default_history_limit")]
    pub history_limit: usize,
    #[serde(default = "default_recording_retention_period")]
    pub recording_retention_period: RecordingRetentionPeriod,
    #[serde(default)]
    pub paste_method: PasteMethod,
    #[serde(default)]
    pub clipboard_handling: ClipboardHandling,
    #[serde(default = "default_auto_submit")]
    pub auto_submit: bool,
    #[serde(default)]
    pub auto_submit_key: AutoSubmitKey,
    #[serde(default = "default_post_process_enabled")]
    pub post_process_enabled: bool,
    #[serde(default = "default_post_process_provider_id")]
    pub post_process_provider_id: String,
    #[serde(default = "default_post_process_providers")]
    pub post_process_providers: Vec<PostProcessProvider>,
    #[serde(default = "default_post_process_api_keys")]
    pub post_process_api_keys: HashMap<String, String>,
    #[serde(default = "default_post_process_models")]
    pub post_process_models: HashMap<String, String>,
    #[serde(default = "default_post_process_prompts")]
    pub post_process_prompts: Vec<LLMPrompt>,
    #[serde(default)]
    pub post_process_selected_prompt_id: Option<String>,
    #[serde(default)]
    pub mute_while_recording: bool,
    #[serde(default)]
    pub append_trailing_space: bool,
    #[serde(default = "default_app_language")]
    pub app_language: String,
    #[serde(default)]
    pub experimental_enabled: bool,
    #[serde(default)]
    pub keyboard_implementation: KeyboardImplementation,
    #[serde(default = "default_show_tray_icon")]
    pub show_tray_icon: bool,
    #[serde(default = "default_paste_delay_ms")]
    pub paste_delay_ms: u64,
    #[serde(default = "default_typing_tool")]
    pub typing_tool: TypingTool,
    pub external_script_path: Option<String>,
    #[serde(default)]
    pub long_audio_model: Option<String>,
    #[serde(default = "default_long_audio_threshold_seconds")]
    pub long_audio_threshold_seconds: f32,
    #[serde(default)]
    pub gemini_api_key: Option<String>,
    #[serde(default = "default_gemini_model")]
    pub gemini_model: String,
    #[serde(default)]
    pub post_process_actions: Vec<PostProcessAction>,
    #[serde(default)]
    pub saved_processing_models: Vec<SavedProcessingModel>,
}

/// Valeur affichée à la place d'un secret dans le journal.
const SECRET_MASK: &str = "[MASQUÉ]";

/// Remplace les clés API non vides par un masque dans une représentation JSON
/// des réglages.
fn mask_secrets(value: &mut serde_json::Value) {
    if let Some(key) = value.get_mut("gemini_api_key") {
        if key.as_str().is_some_and(|k| !k.is_empty()) {
            *key = serde_json::Value::String(SECRET_MASK.to_string());
        }
    }
    if let Some(keys) = value
        .get_mut("post_process_api_keys")
        .and_then(|k| k.as_object_mut())
    {
        for key in keys.values_mut() {
            if key.as_str().is_some_and(|k| !k.is_empty()) {
                *key = serde_json::Value::String(SECRET_MASK.to_string());
            }
        }
    }
}

impl std::fmt::Debug for AppSettings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut value = serde_json::to_value(self).map_err(|_| std::fmt::Error)?;
        mask_secrets(&mut value);
        write!(f, "AppSettings {}", value)
    }
}

impl AppSettings {
    /// Copie destinée à l'export : sans aucune clé API, pour qu'un fichier de
    /// réglages partagé ou synchronisé ne divulgue pas de secret.
    pub fn without_secrets(&self) -> AppSettings {
        let mut copy = self.clone();
        copy.gemini_api_key = None;
        for key in copy.post_process_api_keys.values_mut() {
            key.clear();
        }
        copy
    }

    /// Prépare des réglages importés depuis un fichier : les champs sensibles
    /// (clés API, script externe exécuté à chaque dictée, adresses des
    /// fournisseurs qui reçoivent la clé) sont repris des réglages actuels et
    /// jamais du fichier, qui peut venir d'un tiers.
    pub fn sanitize_import(mut imported: AppSettings, current: &AppSettings) -> AppSettings {
        imported.gemini_api_key = current.gemini_api_key.clone();
        imported.post_process_api_keys = current.post_process_api_keys.clone();
        imported.external_script_path = current.external_script_path.clone();
        if imported.paste_method == PasteMethod::ExternalScript
            && current.paste_method != PasteMethod::ExternalScript
        {
            imported.paste_method = current.paste_method;
        }
        for provider in imported.post_process_providers.iter_mut() {
            match current
                .post_process_providers
                .iter()
                .find(|p| p.id == provider.id)
            {
                Some(existing) => provider.base_url = existing.base_url.clone(),
                None => {
                    if let Some(default) = default_post_process_providers()
                        .into_iter()
                        .find(|p| p.id == provider.id)
                    {
                        provider.base_url = default.base_url;
                    }
                }
            }
        }
        // Un fournisseur inconnu (absent des réglages actuels et des défauts)
        // n'est pas importé : on ne sait pas où il enverrait les données.
        imported.post_process_providers.retain(|p| {
            current.post_process_providers.iter().any(|c| c.id == p.id)
                || default_post_process_providers()
                    .iter()
                    .any(|d| d.id == p.id)
        });
        imported
    }
}

fn default_model() -> String {
    "".to_string()
}

fn default_always_on_microphone() -> bool {
    false
}

fn default_translate_to_english() -> bool {
    false
}

fn default_start_hidden() -> bool {
    false
}

fn default_autostart_enabled() -> bool {
    false
}

fn default_update_checks_enabled() -> bool {
    true
}

fn default_selected_language() -> String {
    "auto".to_string()
}

fn default_overlay_position() -> OverlayPosition {
    #[cfg(target_os = "linux")]
    return OverlayPosition::None;
    #[cfg(not(target_os = "linux"))]
    return OverlayPosition::Bottom;
}

fn default_debug_mode() -> bool {
    false
}

fn default_log_level() -> LogLevel {
    // « Info » par défaut : le niveau « debug » reste disponible dans les
    // réglages avancés pour diagnostiquer un problème.
    LogLevel::Info
}

fn default_word_correction_threshold() -> f64 {
    0.18
}

fn default_paste_delay_ms() -> u64 {
    60
}

fn default_auto_submit() -> bool {
    false
}

fn default_history_limit() -> usize {
    5
}

fn default_recording_retention_period() -> RecordingRetentionPeriod {
    RecordingRetentionPeriod::PreserveLimit
}

fn default_audio_feedback_volume() -> f32 {
    1.0
}

fn default_sound_theme() -> SoundTheme {
    SoundTheme::Marimba
}

fn default_post_process_enabled() -> bool {
    false
}

fn default_app_language() -> String {
    tauri_plugin_os::locale()
        .map(|l| l.replace('_', "-"))
        .unwrap_or_else(|| "en".to_string())
}

fn default_show_tray_icon() -> bool {
    true
}

fn default_post_process_provider_id() -> String {
    "openai".to_string()
}

fn default_post_process_providers() -> Vec<PostProcessProvider> {
    let mut providers = vec![
        PostProcessProvider {
            id: "openai".to_string(),
            label: "OpenAI".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            allow_base_url_edit: false,
            models_endpoint: Some("/models".to_string()),
            supports_structured_output: true,
        },
        PostProcessProvider {
            id: "zai".to_string(),
            label: "Z.AI".to_string(),
            base_url: "https://api.z.ai/api/paas/v4".to_string(),
            allow_base_url_edit: false,
            models_endpoint: Some("/models".to_string()),
            supports_structured_output: true,
        },
        PostProcessProvider {
            id: "openrouter".to_string(),
            label: "OpenRouter".to_string(),
            base_url: "https://openrouter.ai/api/v1".to_string(),
            allow_base_url_edit: false,
            models_endpoint: Some("/models".to_string()),
            supports_structured_output: true,
        },
        PostProcessProvider {
            id: "anthropic".to_string(),
            label: "Anthropic".to_string(),
            base_url: "https://api.anthropic.com/v1".to_string(),
            allow_base_url_edit: false,
            models_endpoint: Some("/models".to_string()),
            supports_structured_output: false,
        },
        PostProcessProvider {
            id: "groq".to_string(),
            label: "Groq".to_string(),
            base_url: "https://api.groq.com/openai/v1".to_string(),
            allow_base_url_edit: false,
            models_endpoint: Some("/models".to_string()),
            supports_structured_output: false,
        },
        PostProcessProvider {
            id: "cerebras".to_string(),
            label: "Cerebras".to_string(),
            base_url: "https://api.cerebras.ai/v1".to_string(),
            allow_base_url_edit: false,
            models_endpoint: Some("/models".to_string()),
            supports_structured_output: true,
        },
    ];

    // Note: We always include Apple Intelligence on macOS ARM64 without checking availability
    // at startup. The availability check is deferred to when the user actually tries to use it
    // (in actions.rs). This prevents crashes on macOS 26.x beta where accessing
    // SystemLanguageModel.default during early app initialization causes SIGABRT.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        providers.push(PostProcessProvider {
            id: APPLE_INTELLIGENCE_PROVIDER_ID.to_string(),
            label: "Apple Intelligence".to_string(),
            base_url: "apple-intelligence://local".to_string(),
            allow_base_url_edit: false,
            models_endpoint: None,
            supports_structured_output: true,
        });
    }

    providers.push(PostProcessProvider {
        id: "gemini".to_string(),
        label: "Gemini".to_string(),
        base_url: "https://generativelanguage.googleapis.com/v1beta".to_string(),
        allow_base_url_edit: false,
        models_endpoint: None,
        supports_structured_output: false,
    });

    // Custom provider always comes last
    providers.push(PostProcessProvider {
        id: "custom".to_string(),
        label: "Custom".to_string(),
        base_url: "http://localhost:11434/v1".to_string(),
        allow_base_url_edit: true,
        models_endpoint: Some("/models".to_string()),
        supports_structured_output: false,
    });

    providers
}

fn default_post_process_api_keys() -> HashMap<String, String> {
    let mut map = HashMap::new();
    for provider in default_post_process_providers() {
        map.insert(provider.id, String::new());
    }
    map
}

fn default_model_for_provider(provider_id: &str) -> String {
    if provider_id == APPLE_INTELLIGENCE_PROVIDER_ID {
        return APPLE_INTELLIGENCE_DEFAULT_MODEL_ID.to_string();
    }
    String::new()
}

fn default_post_process_models() -> HashMap<String, String> {
    let mut map = HashMap::new();
    for provider in default_post_process_providers() {
        map.insert(
            provider.id.clone(),
            default_model_for_provider(&provider.id),
        );
    }
    map
}

fn default_post_process_prompts() -> Vec<LLMPrompt> {
    vec![LLMPrompt {
        id: "default_improve_transcriptions".to_string(),
        name: "Improve Transcriptions".to_string(),
        prompt: "Clean this transcript:\n1. Fix spelling, capitalization, and punctuation errors\n2. Convert number words to digits (twenty-five → 25, ten percent → 10%, five dollars → $5)\n3. Replace spoken punctuation with symbols (period → ., comma → ,, question mark → ?)\n4. Remove filler words (um, uh, like as filler)\n5. Keep the language in the original version (if it was french, keep it in french for example)\n\nPreserve exact meaning and word order. Do not paraphrase or reorder content.\n\nReturn only the cleaned transcript.\n\nTranscript:\n${output}".to_string(),
    }]
}

fn default_typing_tool() -> TypingTool {
    TypingTool::Auto
}

fn default_long_audio_threshold_seconds() -> f32 {
    10.0
}

fn default_gemini_model() -> String {
    "gemini-2.5-flash".to_string()
}

fn ensure_post_process_defaults(settings: &mut AppSettings) -> bool {
    let mut changed = false;
    for provider in default_post_process_providers() {
        // Use match to do a single lookup - either sync existing or add new
        match settings
            .post_process_providers
            .iter_mut()
            .find(|p| p.id == provider.id)
        {
            Some(existing) => {
                // Sync supports_structured_output field for existing providers (migration)
                if existing.supports_structured_output != provider.supports_structured_output {
                    debug!(
                        "Updating supports_structured_output for provider '{}' from {} to {}",
                        provider.id,
                        existing.supports_structured_output,
                        provider.supports_structured_output
                    );
                    existing.supports_structured_output = provider.supports_structured_output;
                    changed = true;
                }
            }
            None => {
                // Provider doesn't exist, add it
                settings.post_process_providers.push(provider.clone());
                changed = true;
            }
        }

        if !settings.post_process_api_keys.contains_key(&provider.id) {
            settings
                .post_process_api_keys
                .insert(provider.id.clone(), String::new());
            changed = true;
        }

        let default_model = default_model_for_provider(&provider.id);
        match settings.post_process_models.get_mut(&provider.id) {
            Some(existing) => {
                if existing.is_empty() && !default_model.is_empty() {
                    *existing = default_model.clone();
                    changed = true;
                }
            }
            None => {
                settings
                    .post_process_models
                    .insert(provider.id.clone(), default_model);
                changed = true;
            }
        }
    }

    changed
}

pub const SETTINGS_STORE_PATH: &str = "settings_store.json";

pub fn get_default_settings() -> AppSettings {
    #[cfg(target_os = "windows")]
    let default_shortcut = "ctrl+space";
    #[cfg(target_os = "macos")]
    let default_shortcut = "option+space";
    #[cfg(target_os = "linux")]
    let default_shortcut = "ctrl+space";
    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    let default_shortcut = "alt+space";

    let mut bindings = HashMap::new();
    bindings.insert(
        "transcribe".to_string(),
        ShortcutBinding {
            id: "transcribe".to_string(),
            name: "Transcribe".to_string(),
            description: "Converts your speech into text.".to_string(),
            default_binding: default_shortcut.to_string(),
            current_binding: default_shortcut.to_string(),
        },
    );
    #[cfg(target_os = "windows")]
    let default_post_process_shortcut = "ctrl+shift+space";
    #[cfg(target_os = "macos")]
    let default_post_process_shortcut = "option+shift+space";
    #[cfg(target_os = "linux")]
    let default_post_process_shortcut = "ctrl+shift+space";
    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    let default_post_process_shortcut = "alt+shift+space";

    bindings.insert(
        "transcribe_with_post_process".to_string(),
        ShortcutBinding {
            id: "transcribe_with_post_process".to_string(),
            name: "Transcribe with Post-Processing".to_string(),
            description: "Converts your speech into text and applies AI post-processing."
                .to_string(),
            default_binding: default_post_process_shortcut.to_string(),
            current_binding: default_post_process_shortcut.to_string(),
        },
    );
    bindings.insert(
        "cancel".to_string(),
        ShortcutBinding {
            id: "cancel".to_string(),
            name: "Cancel".to_string(),
            description: "Cancels the current recording.".to_string(),
            default_binding: "escape".to_string(),
            current_binding: "escape".to_string(),
        },
    );
    bindings.insert(
        "pause".to_string(),
        ShortcutBinding {
            id: "pause".to_string(),
            name: "Pause / Resume".to_string(),
            description: "Pauses or resumes the current recording.".to_string(),
            default_binding: "f6".to_string(),
            current_binding: "f6".to_string(),
        },
    );
    bindings.insert(
        "show_history".to_string(),
        ShortcutBinding {
            id: "show_history".to_string(),
            name: "Show History".to_string(),
            description: "Opens the app window and navigates to the History tab.".to_string(),
            default_binding: "".to_string(),
            current_binding: "".to_string(),
        },
    );
    bindings.insert(
        "copy_latest_history".to_string(),
        ShortcutBinding {
            id: "copy_latest_history".to_string(),
            name: "Copy Latest History".to_string(),
            description: "Copies the latest transcription entry to your clipboard.".to_string(),
            default_binding: "".to_string(),
            current_binding: "".to_string(),
        },
    );

    AppSettings {
        bindings,
        push_to_talk: true,
        audio_feedback: false,
        audio_feedback_volume: default_audio_feedback_volume(),
        sound_theme: default_sound_theme(),
        start_hidden: default_start_hidden(),
        autostart_enabled: default_autostart_enabled(),
        update_checks_enabled: default_update_checks_enabled(),
        selected_model: "".to_string(),
        always_on_microphone: false,
        selected_microphone: None,
        clamshell_microphone: None,
        selected_output_device: None,
        translate_to_english: false,
        selected_language: "auto".to_string(),
        overlay_position: default_overlay_position(),
        debug_mode: false,
        log_level: default_log_level(),
        custom_words: Vec::new(),
        model_unload_timeout: ModelUnloadTimeout::Never,
        word_correction_threshold: default_word_correction_threshold(),
        history_limit: default_history_limit(),
        recording_retention_period: default_recording_retention_period(),
        paste_method: PasteMethod::default(),
        clipboard_handling: ClipboardHandling::default(),
        auto_submit: default_auto_submit(),
        auto_submit_key: AutoSubmitKey::default(),
        post_process_enabled: default_post_process_enabled(),
        post_process_provider_id: default_post_process_provider_id(),
        post_process_providers: default_post_process_providers(),
        post_process_api_keys: default_post_process_api_keys(),
        post_process_models: default_post_process_models(),
        post_process_prompts: default_post_process_prompts(),
        post_process_selected_prompt_id: None,
        mute_while_recording: false,
        append_trailing_space: false,
        app_language: default_app_language(),
        experimental_enabled: false,
        keyboard_implementation: KeyboardImplementation::default(),
        show_tray_icon: default_show_tray_icon(),
        paste_delay_ms: default_paste_delay_ms(),
        typing_tool: default_typing_tool(),
        external_script_path: None,
        long_audio_model: None,
        long_audio_threshold_seconds: default_long_audio_threshold_seconds(),
        gemini_api_key: None,
        gemini_model: default_gemini_model(),
        post_process_actions: Vec::new(),
        saved_processing_models: Vec::new(),
    }
}

impl AppSettings {
    pub fn active_post_process_provider(&self) -> Option<&PostProcessProvider> {
        self.post_process_providers
            .iter()
            .find(|provider| provider.id == self.post_process_provider_id)
    }

    pub fn post_process_provider(&self, provider_id: &str) -> Option<&PostProcessProvider> {
        self.post_process_providers
            .iter()
            .find(|provider| provider.id == provider_id)
    }

    pub fn post_process_provider_mut(
        &mut self,
        provider_id: &str,
    ) -> Option<&mut PostProcessProvider> {
        self.post_process_providers
            .iter_mut()
            .find(|provider| provider.id == provider_id)
    }
}

/// Lit les réglages stockés en tolérant un fichier partiellement invalide :
/// d'abord tel quel, sinon champ par champ par-dessus les valeurs par défaut
/// (un champ illisible est remplacé par sa valeur par défaut au lieu de faire
/// perdre TOUS les réglages, clés API comprises). Le booléen indique si une
/// réparation a été nécessaire.
fn parse_settings_lenient(value: &serde_json::Value) -> (AppSettings, bool) {
    if let Ok(settings) = serde_json::from_value::<AppSettings>(value.clone()) {
        return (settings, false);
    }

    let defaults = get_default_settings();
    let Ok(mut merged) = serde_json::to_value(&defaults) else {
        return (defaults, true);
    };
    if let (Some(stored), Some(target)) = (value.as_object(), merged.as_object_mut()) {
        for (key, field) in stored {
            if !target.contains_key(key) {
                continue;
            }
            let previous = target.insert(key.clone(), field.clone());
            // On ne garde le champ stocké que s'il reste lisible.
            if serde_json::from_value::<AppSettings>(serde_json::Value::Object(target.clone()))
                .is_err()
            {
                warn!(
                    "Réglage illisible ignoré, valeur par défaut utilisée : {}",
                    key
                );
                if let Some(previous) = previous {
                    target.insert(key.clone(), previous);
                }
            }
        }
    }
    let settings = serde_json::from_value::<AppSettings>(merged).unwrap_or(defaults);
    (settings, true)
}

/// Garde une copie du contenu brut des réglages avant de le réparer, pour ne
/// rien perdre définitivement si la réparation se trompe.
fn backup_unreadable_settings(app: &AppHandle, value: &serde_json::Value) {
    let Ok(dir) = app.path().app_data_dir() else {
        return;
    };
    let file = dir.join(format!(
        "settings_store.illisible-{}.json",
        chrono::Local::now().format("%Y%m%d-%H%M%S")
    ));
    match serde_json::to_string_pretty(value) {
        Ok(json) => match std::fs::write(&file, json) {
            Ok(()) => warn!("Copie des réglages illisibles conservée dans {:?}", file),
            Err(e) => warn!("Impossible de sauvegarder les réglages illisibles : {}", e),
        },
        Err(e) => warn!("Impossible de sérialiser les réglages illisibles : {}", e),
    }
}

/// Écrit les réglages dans le store (sans jamais paniquer).
fn store_settings<R: tauri::Runtime>(store: &tauri_plugin_store::Store<R>, settings: &AppSettings) {
    match serde_json::to_value(settings) {
        Ok(value) => store.set("settings", value),
        Err(e) => error!("Impossible d'enregistrer les réglages : {}", e),
    }
}

/// Charge les réglages stockés (réparés si besoin) ou les valeurs par défaut.
fn read_settings(app: &AppHandle) -> AppSettings {
    let store = match app.store(SETTINGS_STORE_PATH) {
        Ok(store) => store,
        Err(e) => {
            error!("Impossible d'ouvrir le store des réglages : {}", e);
            return get_default_settings();
        }
    };

    let mut settings = if let Some(settings_value) = store.get("settings") {
        let (settings, repaired) = parse_settings_lenient(&settings_value);
        if repaired {
            warn!("Réglages partiellement illisibles : réparés avec les valeurs par défaut");
            backup_unreadable_settings(app, &settings_value);
            store_settings(&store, &settings);
        }
        settings
    } else {
        let default_settings = get_default_settings();
        store_settings(&store, &default_settings);
        default_settings
    };

    if ensure_post_process_defaults(&mut settings) {
        store_settings(&store, &settings);
    }

    settings
}

pub fn load_or_create_app_settings(app: &AppHandle) -> AppSettings {
    let mut settings = read_settings(app);
    // Les réglages complets ne sont jamais écrits dans le journal (clés API).
    debug!("Settings loaded");

    // Merge default bindings into existing settings
    let mut updated = false;
    for (key, value) in get_default_settings().bindings {
        if let std::collections::hash_map::Entry::Vacant(entry) = settings.bindings.entry(key) {
            debug!("Adding missing binding: {}", entry.key());
            entry.insert(value);
            updated = true;
        }
    }

    if updated {
        debug!("Settings updated with new bindings");
        write_settings(app, settings.clone());
    }

    settings
}

pub fn get_settings(app: &AppHandle) -> AppSettings {
    read_settings(app)
}

pub fn write_settings(app: &AppHandle, settings: AppSettings) {
    match app.store(SETTINGS_STORE_PATH) {
        Ok(store) => store_settings(&store, &settings),
        Err(e) => error!("Impossible d'ouvrir le store des réglages : {}", e),
    }
}

pub fn get_bindings(app: &AppHandle) -> HashMap<String, ShortcutBinding> {
    let settings = get_settings(app);

    settings.bindings
}

pub fn get_stored_binding(app: &AppHandle, id: &str) -> ShortcutBinding {
    let bindings = get_bindings(app);

    // Raccourci absent des réglages : on retombe sur sa valeur par défaut
    // plutôt que de faire paniquer l'application.
    bindings
        .get(id)
        .cloned()
        .or_else(|| get_default_settings().bindings.get(id).cloned())
        .unwrap_or_else(|| {
            warn!("Unknown binding requested: {}", id);
            ShortcutBinding {
                id: id.to_string(),
                name: id.to_string(),
                description: String::new(),
                default_binding: String::new(),
                current_binding: String::new(),
            }
        })
}

pub fn get_history_limit(app: &AppHandle) -> usize {
    let settings = get_settings(app);
    settings.history_limit
}

pub fn get_recording_retention_period(app: &AppHandle) -> RecordingRetentionPeriod {
    let settings = get_settings(app);
    settings.recording_retention_period
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_settings_disable_auto_submit() {
        let settings = get_default_settings();
        assert!(!settings.auto_submit);
        assert_eq!(settings.auto_submit_key, AutoSubmitKey::Enter);
    }

    fn settings_with_secrets() -> AppSettings {
        let mut settings = get_default_settings();
        settings.gemini_api_key = Some("cle-test-gemini-123".to_string());
        settings
            .post_process_api_keys
            .insert("openai".to_string(), "cle-test-openai-456".to_string());
        settings
    }

    #[test]
    fn debug_output_masks_api_keys() {
        let output = format!("{:?}", settings_with_secrets());
        assert!(!output.contains("cle-test-gemini-123"));
        assert!(!output.contains("cle-test-openai-456"));
        assert!(output.contains(SECRET_MASK));
    }

    #[test]
    fn export_copy_has_no_secret() {
        let exported = settings_with_secrets().without_secrets();
        assert!(exported.gemini_api_key.is_none());
        assert!(exported
            .post_process_api_keys
            .values()
            .all(|k| k.is_empty()));
    }

    #[test]
    fn import_keeps_sensitive_fields_from_current_settings() {
        let current = settings_with_secrets();
        let mut imported = get_default_settings();
        imported.gemini_api_key = Some("cle-du-fichier".to_string());
        imported.external_script_path = Some("C:/malveillant.exe".to_string());
        imported.paste_method = PasteMethod::ExternalScript;
        if let Some(p) = imported
            .post_process_providers
            .iter_mut()
            .find(|p| p.id == "openai")
        {
            p.base_url = "https://serveur-pirate.example".to_string();
        }
        imported.post_process_providers.push(PostProcessProvider {
            id: "inconnu".to_string(),
            label: "Inconnu".to_string(),
            base_url: "https://ailleurs.example".to_string(),
            allow_base_url_edit: true,
            models_endpoint: None,
            supports_structured_output: false,
        });

        let result = AppSettings::sanitize_import(imported, &current);

        assert_eq!(result.gemini_api_key, current.gemini_api_key);
        assert_eq!(result.external_script_path, None);
        assert_ne!(result.paste_method, PasteMethod::ExternalScript);
        let openai = result.post_process_provider("openai").unwrap();
        assert_eq!(openai.base_url, "https://api.openai.com/v1");
        assert!(result.post_process_provider("inconnu").is_none());
    }

    #[test]
    fn lenient_parse_keeps_valid_fields_and_repairs_invalid_ones() {
        let mut value = serde_json::to_value(settings_with_secrets()).unwrap();
        let obj = value.as_object_mut().unwrap();
        obj.insert(
            "history_limit".to_string(),
            serde_json::json!("pas un nombre"),
        );
        obj.remove("push_to_talk");

        let (settings, repaired) = parse_settings_lenient(&value);

        assert!(repaired);
        assert_eq!(
            settings.gemini_api_key.as_deref(),
            Some("cle-test-gemini-123")
        );
        assert_eq!(settings.history_limit, default_history_limit());
        assert!(settings.push_to_talk);
    }

    #[test]
    fn lenient_parse_does_not_flag_valid_settings() {
        let value = serde_json::to_value(get_default_settings()).unwrap();
        let (_, repaired) = parse_settings_lenient(&value);
        assert!(!repaired);
    }
}
