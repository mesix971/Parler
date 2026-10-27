use crate::audio_toolkit::{apply_custom_words, filter_transcription_output};
use crate::managers::model::{EngineType, ModelManager};
use crate::settings::{get_settings, ModelUnloadTimeout};
use anyhow::Result;
use log::{debug, error, info, warn};
use serde::Serialize;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, SystemTime};
use tauri::{AppHandle, Emitter, Manager};
use transcribe_rs::{
    onnx::{
        moonshine::{
            MoonshineModel, MoonshineParams, MoonshineStreamingParams, MoonshineVariant,
            StreamingModel,
        },
        parakeet::{ParakeetModel, ParakeetParams, TimestampGranularity},
        sense_voice::{SenseVoiceModel, SenseVoiceParams},
        Quantization,
    },
    whisper_cpp::{WhisperEngine, WhisperInferenceParams},
    TranscriptionResult,
};

#[derive(Clone, Debug, Serialize)]
pub struct ModelStateEvent {
    pub event_type: String,
    pub model_id: Option<String>,
    pub model_name: Option<String>,
    pub error: Option<String>,
}

enum LoadedEngine {
    Whisper(WhisperEngine),
    Parakeet(ParakeetModel),
    Moonshine(MoonshineModel),
    MoonshineStreaming(StreamingModel),
    SenseVoice(SenseVoiceModel),
    GeminiApi,
}

/// Horodatage actuel en millisecondes (0 si l'horloge système est incohérente).
fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// Pas d'implémentation de `Drop` ici : ce type est cloné (fil de chargement,
// surveillance d'inactivité), et un `Drop` s'exécutait à la destruction de
// CHAQUE clone — il arrêtait la surveillance d'inactivité dès le premier
// chargement de modèle, et le déchargement automatique ne marchait plus.
// La surveillance vit aussi longtemps que l'application.
#[derive(Clone)]
pub struct TranscriptionManager {
    engine: Arc<Mutex<Option<LoadedEngine>>>,
    model_manager: Arc<ModelManager>,
    app_handle: AppHandle,
    current_model_id: Arc<Mutex<Option<String>>>,
    last_activity: Arc<AtomicU64>,
    is_loading: Arc<Mutex<bool>>,
    loading_condvar: Arc<Condvar>,
}

impl TranscriptionManager {
    pub fn new(app_handle: &AppHandle, model_manager: Arc<ModelManager>) -> Result<Self> {
        let manager = Self {
            engine: Arc::new(Mutex::new(None)),
            model_manager,
            app_handle: app_handle.clone(),
            current_model_id: Arc::new(Mutex::new(None)),
            last_activity: Arc::new(AtomicU64::new(now_millis())),
            is_loading: Arc::new(Mutex::new(false)),
            loading_condvar: Arc::new(Condvar::new()),
        };

        // Start the idle watcher
        {
            let app_handle_cloned = app_handle.clone();
            let manager_cloned = manager.clone();
            thread::spawn(move || {
                loop {
                    thread::sleep(Duration::from_secs(10)); // Check every 10 seconds

                    let settings = get_settings(&app_handle_cloned);
                    let timeout_seconds = settings.model_unload_timeout.to_seconds();

                    if let Some(limit_seconds) = timeout_seconds {
                        // Skip polling-based unloading for immediate timeout since it's handled directly in transcribe()
                        if settings.model_unload_timeout == ModelUnloadTimeout::Immediately {
                            continue;
                        }

                        let last = manager_cloned.last_activity.load(Ordering::Relaxed);
                        let now_ms = now_millis();

                        // Jamais de déchargement pendant une dictée en cours :
                        // elle échouerait à la transcription (« aucun modèle »).
                        let recording = app_handle_cloned
                            .try_state::<Arc<crate::managers::audio::AudioRecordingManager>>()
                            .is_some_and(|audio| audio.is_recording());

                        if !recording && now_ms.saturating_sub(last) > limit_seconds * 1000 {
                            // idle -> unload
                            if manager_cloned.is_model_loaded() {
                                let unload_start = std::time::Instant::now();
                                debug!("Starting to unload model due to inactivity");

                                if let Ok(()) = manager_cloned.unload_model() {
                                    let _ = app_handle_cloned.emit(
                                        "model-state-changed",
                                        ModelStateEvent {
                                            event_type: "unloaded".to_string(),
                                            model_id: None,
                                            model_name: None,
                                            error: None,
                                        },
                                    );
                                    let unload_duration = unload_start.elapsed();
                                    debug!(
                                        "Model unloaded due to inactivity (took {}ms)",
                                        unload_duration.as_millis()
                                    );
                                }
                            }
                        }
                    }
                }
            });
        }

        Ok(manager)
    }

    /// Lock the engine mutex, recovering from poison if a previous transcription panicked.
    fn lock_engine(&self) -> MutexGuard<'_, Option<LoadedEngine>> {
        self.engine.lock().unwrap_or_else(|poisoned| {
            warn!("Engine mutex was poisoned by a previous panic, recovering");
            poisoned.into_inner()
        })
    }

    pub fn is_model_loaded(&self) -> bool {
        let engine = self.lock_engine();
        engine.is_some()
    }

    pub fn unload_model(&self) -> Result<()> {
        let unload_start = std::time::Instant::now();
        debug!("Starting to unload model");

        {
            let mut engine = self.lock_engine();
            // Drop the engine to free memory (v0.3 engines are cleaned up on drop)
            *engine = None;
        }
        {
            let mut current_model = self
                .current_model_id
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            *current_model = None;
        }

        // Emit unloaded event
        let _ = self.app_handle.emit(
            "model-state-changed",
            ModelStateEvent {
                event_type: "unloaded".to_string(),
                model_id: None,
                model_name: None,
                error: None,
            },
        );

        let unload_duration = unload_start.elapsed();
        debug!(
            "Model unloaded manually (took {}ms)",
            unload_duration.as_millis()
        );
        Ok(())
    }

    /// Unloads the model immediately if the setting is enabled and the model is loaded
    pub fn maybe_unload_immediately(&self, context: &str) {
        let settings = get_settings(&self.app_handle);
        if settings.model_unload_timeout == ModelUnloadTimeout::Immediately
            && self.is_model_loaded()
        {
            info!("Immediately unloading model after {}", context);
            if let Err(e) = self.unload_model() {
                warn!("Failed to immediately unload model: {}", e);
            }
        }
    }

    pub fn load_model(&self, model_id: &str) -> Result<()> {
        let load_start = std::time::Instant::now();
        debug!("Starting to load model: {}", model_id);

        // Emit loading started event
        let _ = self.app_handle.emit(
            "model-state-changed",
            ModelStateEvent {
                event_type: "loading_started".to_string(),
                model_id: Some(model_id.to_string()),
                model_name: None,
                error: None,
            },
        );

        let model_info = self
            .model_manager
            .get_model_info(model_id)
            .ok_or_else(|| anyhow::anyhow!("Model not found: {}", model_id))?;

        if !model_info.is_downloaded {
            let error_msg = "Model not downloaded";
            let _ = self.app_handle.emit(
                "model-state-changed",
                ModelStateEvent {
                    event_type: "loading_failed".to_string(),
                    model_id: Some(model_id.to_string()),
                    model_name: Some(model_info.name.clone()),
                    error: Some(error_msg.to_string()),
                },
            );
            return Err(anyhow::anyhow!(error_msg));
        }

        let model_path = if matches!(model_info.engine_type, EngineType::GeminiApi) {
            std::path::PathBuf::new()
        } else {
            self.model_manager.get_model_path(model_id)?
        };

        // Create appropriate engine based on model type
        let loaded_engine = match model_info.engine_type {
            EngineType::Whisper => {
                let engine = WhisperEngine::load(&model_path).map_err(|e| {
                    let error_msg = format!("Failed to load whisper model {}: {}", model_id, e);
                    let _ = self.app_handle.emit(
                        "model-state-changed",
                        ModelStateEvent {
                            event_type: "loading_failed".to_string(),
                            model_id: Some(model_id.to_string()),
                            model_name: Some(model_info.name.clone()),
                            error: Some(error_msg.clone()),
                        },
                    );
                    anyhow::anyhow!(error_msg)
                })?;
                LoadedEngine::Whisper(engine)
            }
            EngineType::Parakeet => {
                let engine =
                    ParakeetModel::load(&model_path, &Quantization::Int8).map_err(|e| {
                        let error_msg =
                            format!("Failed to load parakeet model {}: {}", model_id, e);
                        let _ = self.app_handle.emit(
                            "model-state-changed",
                            ModelStateEvent {
                                event_type: "loading_failed".to_string(),
                                model_id: Some(model_id.to_string()),
                                model_name: Some(model_info.name.clone()),
                                error: Some(error_msg.clone()),
                            },
                        );
                        anyhow::anyhow!(error_msg)
                    })?;
                LoadedEngine::Parakeet(engine)
            }
            EngineType::Moonshine => {
                let engine = MoonshineModel::load(
                    &model_path,
                    MoonshineVariant::Base,
                    &Quantization::default(),
                )
                .map_err(|e| {
                    let error_msg = format!("Failed to load moonshine model {}: {}", model_id, e);
                    let _ = self.app_handle.emit(
                        "model-state-changed",
                        ModelStateEvent {
                            event_type: "loading_failed".to_string(),
                            model_id: Some(model_id.to_string()),
                            model_name: Some(model_info.name.clone()),
                            error: Some(error_msg.clone()),
                        },
                    );
                    anyhow::anyhow!(error_msg)
                })?;
                LoadedEngine::Moonshine(engine)
            }
            EngineType::MoonshineStreaming => {
                let engine = StreamingModel::load(&model_path, 4, &Quantization::default())
                    .map_err(|e| {
                        let error_msg = format!(
                            "Failed to load moonshine streaming model {}: {}",
                            model_id, e
                        );
                        let _ = self.app_handle.emit(
                            "model-state-changed",
                            ModelStateEvent {
                                event_type: "loading_failed".to_string(),
                                model_id: Some(model_id.to_string()),
                                model_name: Some(model_info.name.clone()),
                                error: Some(error_msg.clone()),
                            },
                        );
                        anyhow::anyhow!(error_msg)
                    })?;
                LoadedEngine::MoonshineStreaming(engine)
            }
            EngineType::SenseVoice => {
                let engine =
                    SenseVoiceModel::load(&model_path, &Quantization::Int8).map_err(|e| {
                        let error_msg =
                            format!("Failed to load SenseVoice model {}: {}", model_id, e);
                        let _ = self.app_handle.emit(
                            "model-state-changed",
                            ModelStateEvent {
                                event_type: "loading_failed".to_string(),
                                model_id: Some(model_id.to_string()),
                                model_name: Some(model_info.name.clone()),
                                error: Some(error_msg.clone()),
                            },
                        );
                        anyhow::anyhow!(error_msg)
                    })?;
                LoadedEngine::SenseVoice(engine)
            }
            EngineType::GeminiApi => {
                let settings = get_settings(&self.app_handle);
                if settings.gemini_api_key.is_none()
                    || settings
                        .gemini_api_key
                        .as_ref()
                        .is_none_or(|k| k.is_empty())
                {
                    let error_msg = "Gemini API key not configured";
                    let _ = self.app_handle.emit(
                        "model-state-changed",
                        ModelStateEvent {
                            event_type: "loading_failed".to_string(),
                            model_id: Some(model_id.to_string()),
                            model_name: Some(model_info.name.clone()),
                            error: Some(error_msg.to_string()),
                        },
                    );
                    return Err(anyhow::anyhow!(error_msg));
                }
                LoadedEngine::GeminiApi
            }
        };

        // Update the current engine and model ID
        {
            let mut engine = self.lock_engine();
            *engine = Some(loaded_engine);
        }
        {
            let mut current_model = self
                .current_model_id
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            *current_model = Some(model_id.to_string());
        }

        // Emit loading completed event
        let _ = self.app_handle.emit(
            "model-state-changed",
            ModelStateEvent {
                event_type: "loading_completed".to_string(),
                model_id: Some(model_id.to_string()),
                model_name: Some(model_info.name.clone()),
                error: None,
            },
        );

        let load_duration = load_start.elapsed();
        debug!(
            "Successfully loaded transcription model: {} (took {}ms)",
            model_id,
            load_duration.as_millis()
        );
        Ok(())
    }

    /// Kicks off the model loading in a background thread if it's not already loaded
    pub fn initiate_model_load(&self) {
        // Appelé au début de chaque dictée : compte comme une activité, pour
        // que le déchargement automatique ne frappe pas pendant l'enregistrement.
        self.last_activity.store(now_millis(), Ordering::Relaxed);
        let mut is_loading = self
            .is_loading
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // Un modèle chargé qui n'est pas celui des réglages (modèle local resté
        // après un repli dont la restauration a échoué) est remplacé.
        let selected = get_settings(&self.app_handle).selected_model;
        if *is_loading
            || (self.is_model_loaded() && self.get_current_model().as_deref() == Some(&selected))
        {
            return;
        }

        *is_loading = true;
        let self_clone = self.clone();
        thread::spawn(move || {
            let settings = get_settings(&self_clone.app_handle);
            // Une panique pendant le chargement ne doit pas laisser `is_loading`
            // à vrai : la transcription attendrait alors indéfiniment.
            match catch_unwind(AssertUnwindSafe(|| {
                self_clone.load_model(&settings.selected_model)
            })) {
                Ok(Ok(())) => {}
                Ok(Err(e)) => error!("Failed to load model: {}", e),
                Err(_) => error!("Model loading panicked"),
            }
            let mut is_loading = self_clone
                .is_loading
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            *is_loading = false;
            self_clone.loading_condvar.notify_all();
        });
    }

    pub fn get_current_model(&self) -> Option<String> {
        let current_model = self
            .current_model_id
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        current_model.clone()
    }

    /// Le modèle local téléchargé le plus précis qui connaît la langue dictée
    /// (celle des réglages, ou celle de l'interface en détection automatique) :
    /// il retranscrit une dictée quand Gemini échoue et que le modèle principal
    /// est Gemini. Sans cela, un modèle anglais seul (Parakeet V2) passerait
    /// devant le multilingue et rendrait du charabia en français.
    pub fn local_fallback_model(&self) -> Option<String> {
        let settings = get_settings(&self.app_handle);
        let language = if settings.selected_language == "auto" {
            settings
                .app_language
                .split(['-', '_'])
                .next()
                .unwrap_or_default()
                .to_string()
        } else {
            settings.selected_language
        };
        self.model_manager
            .get_available_models()
            .into_iter()
            .filter(|m| {
                m.is_downloaded
                    && !matches!(m.engine_type, EngineType::GeminiApi)
                    && m.supported_languages.contains(&language)
            })
            // À précision égale, le plus rapide, puis l'ordre alphabétique (stable)
            .max_by(|a, b| {
                a.accuracy_score
                    .total_cmp(&b.accuracy_score)
                    .then(a.speed_score.total_cmp(&b.speed_score))
                    .then(b.id.cmp(&a.id))
            })
            .map(|m| m.id)
    }

    pub fn get_current_model_name(&self) -> Option<String> {
        let model_id = self.get_current_model()?;
        self.model_manager
            .get_model_info(&model_id)
            .map(|info| info.name)
    }

    pub fn transcribe(&self, audio: Vec<f32>) -> Result<String> {
        // Update last activity timestamp
        self.last_activity.store(now_millis(), Ordering::Relaxed);

        let st = std::time::Instant::now();

        debug!("Audio vector length: {}", audio.len());

        if audio.is_empty() {
            debug!("Empty audio vector");
            self.maybe_unload_immediately("empty audio");
            return Ok(String::new());
        }

        // Check if model is loaded, if not try to load it
        {
            // If the model is loading, wait for it to complete.
            let mut is_loading = self
                .is_loading
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            while *is_loading {
                is_loading = self
                    .loading_condvar
                    .wait(is_loading)
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
            }

            let engine_guard = self.lock_engine();
            if engine_guard.is_none() {
                // Cas fréquent : Gemini choisi mais clé absente, le « chargement »
                // a échoué. On le dit clairement au lieu de « aucun modèle ».
                let settings = get_settings(&self.app_handle);
                let gemini_without_key = settings.selected_model == "gemini-api"
                    && settings
                        .gemini_api_key
                        .as_deref()
                        .is_none_or(|key| key.trim().is_empty());
                if gemini_without_key {
                    return Err(anyhow::Error::new(
                        crate::gemini_client::GeminiError::missing_key(),
                    ));
                }
                return Err(anyhow::anyhow!("Model is not loaded for transcription."));
            }
        }

        // Get current settings for configuration
        let settings = get_settings(&self.app_handle);

        // Handle Gemini API separately (requires async HTTP call)
        {
            let engine_guard = self.lock_engine();
            if let Some(LoadedEngine::GeminiApi) = engine_guard.as_ref() {
                drop(engine_guard);
                let api_key = settings
                    .gemini_api_key
                    .clone()
                    .filter(|k| !k.trim().is_empty())
                    .ok_or_else(|| {
                        anyhow::Error::new(crate::gemini_client::GeminiError::missing_key())
                    })?;
                let gemini_model = settings.gemini_model.clone();

                let request = crate::gemini_client::transcribe_audio_with_backup(
                    &api_key,
                    settings.gemini_api_key_backup.as_deref(),
                    &gemini_model,
                    &audio,
                );
                let result = match tokio::runtime::Handle::try_current() {
                    // Use block_in_place to safely run async code from a tokio worker thread.
                    // Handle::block_on() panics if called directly from an async context,
                    // so block_in_place tells tokio to move its work off this thread first.
                    Ok(handle) => tokio::task::block_in_place(|| handle.block_on(request))?,
                    // Appel depuis un fil hors runtime : on passe par celui de Tauri.
                    Err(_) => tauri::async_runtime::block_on(request)?,
                };

                let corrected = if !settings.custom_words.is_empty() {
                    apply_custom_words(
                        &result,
                        &settings.custom_words,
                        settings.word_correction_threshold,
                    )
                } else {
                    result
                };
                let final_result = filter_transcription_output(&corrected);

                let et = std::time::Instant::now();
                info!(
                    "Gemini transcription completed in {}ms",
                    (et - st).as_millis()
                );

                self.maybe_unload_immediately("gemini transcription");
                return Ok(final_result);
            }
        }

        // Perform transcription with the appropriate engine.
        // We use catch_unwind to prevent engine panics from poisoning the mutex,
        // which would make the app hang indefinitely on subsequent operations.
        let result = {
            let mut engine_guard = self.lock_engine();

            // Take the engine out so we own it during transcription.
            // If the engine panics, we simply don't put it back (effectively unloading it)
            // instead of poisoning the mutex.
            let mut engine = match engine_guard.take() {
                Some(e) => e,
                None => {
                    return Err(anyhow::anyhow!(
                        "Model failed to load after auto-load attempt. Please check your model settings."
                    ));
                }
            };

            // Release the lock before transcribing — no mutex held during the engine call
            drop(engine_guard);

            let transcribe_result =
                catch_unwind(AssertUnwindSafe(|| -> Result<TranscriptionResult> {
                    match &mut engine {
                        LoadedEngine::Whisper(whisper_engine) => {
                            let whisper_language = if settings.selected_language == "auto" {
                                None
                            } else {
                                let normalized = if settings.selected_language == "zh-Hans"
                                    || settings.selected_language == "zh-Hant"
                                {
                                    "zh".to_string()
                                } else {
                                    settings.selected_language.clone()
                                };
                                Some(normalized)
                            };

                            let params = WhisperInferenceParams {
                                language: whisper_language,
                                translate: settings.translate_to_english,
                                ..Default::default()
                            };

                            whisper_engine
                                .transcribe_with(&audio, &params)
                                .map_err(|e| anyhow::anyhow!("Whisper transcription failed: {}", e))
                        }
                        LoadedEngine::Parakeet(parakeet_engine) => {
                            let params = ParakeetParams {
                                timestamp_granularity: Some(TimestampGranularity::Segment),
                                ..Default::default()
                            };
                            parakeet_engine
                                .transcribe_with(&audio, &params)
                                .map_err(|e| {
                                    anyhow::anyhow!("Parakeet transcription failed: {}", e)
                                })
                        }
                        LoadedEngine::Moonshine(moonshine_engine) => moonshine_engine
                            .transcribe_with(&audio, &MoonshineParams::default())
                            .map_err(|e| anyhow::anyhow!("Moonshine transcription failed: {}", e)),
                        LoadedEngine::MoonshineStreaming(streaming_engine) => streaming_engine
                            .transcribe_with(&audio, &MoonshineStreamingParams::default())
                            .map_err(|e| {
                                anyhow::anyhow!("Moonshine streaming transcription failed: {}", e)
                            }),
                        LoadedEngine::SenseVoice(sense_voice_engine) => {
                            let language = match settings.selected_language.as_str() {
                                "zh" | "zh-Hans" | "zh-Hant" => Some("zh".to_string()),
                                "en" => Some("en".to_string()),
                                "ja" => Some("ja".to_string()),
                                "ko" => Some("ko".to_string()),
                                "yue" => Some("yue".to_string()),
                                _ => None,
                            };
                            let params = SenseVoiceParams {
                                language,
                                use_itn: Some(true),
                            };
                            sense_voice_engine
                                .transcribe_with(&audio, &params)
                                .map_err(|e| {
                                    anyhow::anyhow!("SenseVoice transcription failed: {}", e)
                                })
                        }
                        LoadedEngine::GeminiApi => {
                            unreachable!("GeminiApi handled before catch_unwind")
                        }
                    }
                }));

            match transcribe_result {
                Ok(inner_result) => {
                    // Success or normal error — put the engine back
                    let mut engine_guard = self.lock_engine();
                    *engine_guard = Some(engine);
                    inner_result?
                }
                Err(panic_payload) => {
                    // Engine panicked — do NOT put it back (it's in an unknown state).
                    // The engine is dropped here, effectively unloading it.
                    let panic_msg = if let Some(s) = panic_payload.downcast_ref::<&str>() {
                        s.to_string()
                    } else if let Some(s) = panic_payload.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "unknown panic".to_string()
                    };
                    error!(
                        "Transcription engine panicked: {}. Model has been unloaded.",
                        panic_msg
                    );

                    // Clear the model ID so it will be reloaded on next attempt
                    {
                        let mut current_model = self
                            .current_model_id
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                        *current_model = None;
                    }

                    let _ = self.app_handle.emit(
                        "model-state-changed",
                        ModelStateEvent {
                            event_type: "unloaded".to_string(),
                            model_id: None,
                            model_name: None,
                            error: Some(format!("Engine panicked: {}", panic_msg)),
                        },
                    );

                    return Err(anyhow::anyhow!(
                        "Transcription engine panicked: {}. The model has been unloaded and will reload on next attempt.",
                        panic_msg
                    ));
                }
            }
        };

        // Apply word correction if custom words are configured
        let corrected_result = if !settings.custom_words.is_empty() {
            apply_custom_words(
                &result.text,
                &settings.custom_words,
                settings.word_correction_threshold,
            )
        } else {
            result.text
        };

        // Filter out filler words and hallucinations
        let filtered_result = filter_transcription_output(&corrected_result);

        let et = std::time::Instant::now();
        let translation_note = if settings.translate_to_english {
            " (translated)"
        } else {
            ""
        };
        info!(
            "Transcription completed in {}ms{}",
            (et - st).as_millis(),
            translation_note
        );

        let final_result = filtered_result;

        if final_result.is_empty() {
            info!("Transcription result is empty");
        } else {
            // Le texte dicté n'est jamais écrit dans le journal (donnée personnelle).
            info!(
                "Transcription result: {} chars",
                final_result.chars().count()
            );
        }

        self.maybe_unload_immediately("transcription");

        Ok(final_result)
    }
}
