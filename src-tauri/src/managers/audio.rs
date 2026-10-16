use crate::audio_toolkit::{list_input_devices, vad::SmoothedVad, AudioRecorder, SileroVad};
use crate::helpers::clamshell;
use crate::settings::{get_settings, AppSettings};
use crate::utils;
use log::{debug, error, info, warn};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tauri::Manager;

fn set_mute(mute: bool) {
    // Expected behavior:
    // - Windows: works on most systems using standard audio drivers.
    // - Linux: works on many systems (PipeWire, PulseAudio, ALSA),
    //   but some distros may lack the tools used.
    // - macOS: works on most standard setups via AppleScript.
    // If unsupported, fails silently.

    #[cfg(target_os = "windows")]
    {
        unsafe {
            use windows::Win32::{
                Media::Audio::{
                    eMultimedia, eRender, Endpoints::IAudioEndpointVolume, IMMDeviceEnumerator,
                    MMDeviceEnumerator,
                },
                System::Com::{CoCreateInstance, CoInitializeEx, CLSCTX_ALL, COINIT_MULTITHREADED},
            };

            macro_rules! unwrap_or_return {
                ($expr:expr) => {
                    match $expr {
                        Ok(val) => val,
                        Err(_) => return,
                    }
                };
            }

            // Initialize the COM library for this thread.
            // If already initialized (e.g., by another library like Tauri), this does nothing.
            let _ = CoInitializeEx(None, COINIT_MULTITHREADED);

            let all_devices: IMMDeviceEnumerator =
                unwrap_or_return!(CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL));
            let default_device =
                unwrap_or_return!(all_devices.GetDefaultAudioEndpoint(eRender, eMultimedia));
            let volume_interface = unwrap_or_return!(
                default_device.Activate::<IAudioEndpointVolume>(CLSCTX_ALL, None)
            );

            let _ = volume_interface.SetMute(mute, std::ptr::null());
        }
    }

    #[cfg(target_os = "linux")]
    {
        use std::process::Command;

        let mute_val = if mute { "1" } else { "0" };
        let amixer_state = if mute { "mute" } else { "unmute" };

        // Try multiple backends to increase compatibility
        // 1. PipeWire (wpctl)
        if Command::new("wpctl")
            .args(["set-mute", "@DEFAULT_AUDIO_SINK@", mute_val])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
        {
            return;
        }

        // 2. PulseAudio (pactl)
        if Command::new("pactl")
            .args(["set-sink-mute", "@DEFAULT_SINK@", mute_val])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
        {
            return;
        }

        // 3. ALSA (amixer)
        let _ = Command::new("amixer")
            .args(["set", "Master", amixer_state])
            .output();
    }

    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        let script = format!(
            "set volume output muted {}",
            if mute { "true" } else { "false" }
        );
        let _ = Command::new("osascript").args(["-e", &script]).output();
    }
}

/// Checks if the system is already muted by the user before we apply our own mute.
/// Returns true if the system output is muted or volume is 0.
#[cfg(target_os = "macos")]
fn is_system_already_muted() -> bool {
    use std::process::Command;

    let result = Command::new("osascript")
        .arg("-e")
        .arg("set v to (get volume settings)\nreturn (output muted of v) as text & \",\" & (output volume of v) as text")
        .output();

    match result {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let parts: Vec<&str> = stdout.trim().split(',').collect();
            if parts.len() == 2 {
                let muted = parts[0].trim() == "true";
                let volume_zero = parts[1].trim().parse::<i32>().unwrap_or(100) == 0;
                return muted || volume_zero;
            }
            false
        }
        Err(_) => false,
    }
}

#[cfg(target_os = "windows")]
fn is_system_already_muted() -> bool {
    unsafe {
        use windows::Win32::{
            Media::Audio::{
                eMultimedia, eRender, Endpoints::IAudioEndpointVolume, IMMDeviceEnumerator,
                MMDeviceEnumerator,
            },
            System::Com::{CoCreateInstance, CoInitializeEx, CLSCTX_ALL, COINIT_MULTITHREADED},
        };

        let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
        let enumerator: IMMDeviceEnumerator =
            match CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL) {
                Ok(e) => e,
                Err(_) => return false,
            };
        let device = match enumerator.GetDefaultAudioEndpoint(eRender, eMultimedia) {
            Ok(d) => d,
            Err(_) => return false,
        };
        let volume: IAudioEndpointVolume =
            match device.Activate::<IAudioEndpointVolume>(CLSCTX_ALL, None) {
                Ok(v) => v,
                Err(_) => return false,
            };

        volume.GetMute().unwrap_or(false.into()).as_bool()
    }
}

#[cfg(target_os = "linux")]
fn is_system_already_muted() -> bool {
    false
}

const WHISPER_SAMPLE_RATE: usize = 16000;

/* ──────────────────────────────────────────────────────────────── */

#[derive(Clone, Debug)]
pub enum RecordingState {
    Idle,
    Recording { binding_id: String },
}

#[derive(Clone, Debug)]
pub enum MicrophoneMode {
    AlwaysOn,
    OnDemand,
}

/// Raison pour laquelle un enregistrement n'a pas pu démarrer.
#[derive(Debug)]
pub enum StartRecordingError {
    /// Un enregistrement est déjà en cours.
    Busy,
    /// Le micro n'a pas pu être ouvert ou démarré (absent, occupé, débranché…).
    Microphone(String),
}

/* ──────────────────────────────────────────────────────────────── */

fn create_audio_recorder(
    vad_path: &str,
    app_handle: &tauri::AppHandle,
    is_paused: Arc<AtomicBool>,
) -> Result<AudioRecorder, anyhow::Error> {
    let silero = SileroVad::new(vad_path, 0.3)
        .map_err(|e| anyhow::anyhow!("Failed to create SileroVad: {}", e))?;
    let smoothed_vad = SmoothedVad::new(Box::new(silero), 15, 15, 2);

    // Recorder with VAD plus a spectrum-level callback that forwards updates to
    // the frontend.
    let recorder = AudioRecorder::new()
        .map_err(|e| anyhow::anyhow!("Failed to create AudioRecorder: {}", e))?
        .with_vad(Box::new(smoothed_vad))
        .with_pause_flag(is_paused.clone())
        .with_level_callback({
            let app_handle = app_handle.clone();
            let is_paused = is_paused.clone();
            move |levels| {
                if is_paused.load(Ordering::Relaxed) {
                    let zero_levels = vec![0.0f32; levels.len()];
                    utils::emit_levels(&app_handle, &zero_levels);
                } else {
                    utils::emit_levels(&app_handle, &levels);
                }
            }
        });

    Ok(recorder)
}

/* ──────────────────────────────────────────────────────────────── */

// Ordre de prise des verrous, à respecter partout pour éviter tout
// interblocage : state → mode → is_open → did_mute → recorder → is_recording.
#[derive(Clone)]
pub struct AudioRecordingManager {
    state: Arc<Mutex<RecordingState>>,
    mode: Arc<Mutex<MicrophoneMode>>,
    app_handle: tauri::AppHandle,

    recorder: Arc<Mutex<Option<AudioRecorder>>>,
    is_open: Arc<Mutex<bool>>,
    is_recording: Arc<Mutex<bool>>,
    is_paused: Arc<AtomicBool>,
    did_mute: Arc<Mutex<bool>>,
    /// Vrai entre le démarrage d'un enregistrement et `remove_mute` : empêche
    /// une coupure du son différée (après le son de départ) d'arriver après
    /// la fin d'un appui très court, ce qui laisserait le son coupé.
    mute_allowed: Arc<AtomicBool>,
}

impl AudioRecordingManager {
    /* ---------- construction ------------------------------------------------ */

    pub fn new(app: &tauri::AppHandle) -> Result<Self, anyhow::Error> {
        let settings = get_settings(app);
        let mode = if settings.always_on_microphone {
            MicrophoneMode::AlwaysOn
        } else {
            MicrophoneMode::OnDemand
        };

        let manager = Self {
            state: Arc::new(Mutex::new(RecordingState::Idle)),
            mode: Arc::new(Mutex::new(mode.clone())),
            app_handle: app.clone(),

            recorder: Arc::new(Mutex::new(None)),
            is_open: Arc::new(Mutex::new(false)),
            is_recording: Arc::new(Mutex::new(false)),
            is_paused: Arc::new(AtomicBool::new(false)),
            did_mute: Arc::new(Mutex::new(false)),
            mute_allowed: Arc::new(AtomicBool::new(false)),
        };

        // Always-on?  Open immediately. Un micro absent au démarrage ne doit pas
        // empêcher l'application de se lancer : il sera rouvert à la première dictée.
        if matches!(mode, MicrophoneMode::AlwaysOn) {
            if let Err(e) = manager.start_microphone_stream() {
                error!("Failed to open always-on microphone at startup: {e}");
            }
        }

        Ok(manager)
    }

    /* ---------- helper methods --------------------------------------------- */

    fn get_effective_microphone_device(&self, settings: &AppSettings) -> Option<cpal::Device> {
        // Check if we're in clamshell mode and have a clamshell microphone configured
        let clamshell_mic = settings
            .clamshell_microphone
            .as_ref()
            .filter(|_| clamshell::is_clamshell().unwrap_or(false));

        let device_name = match clamshell_mic {
            Some(name) => name,
            None => settings.selected_microphone.as_ref()?,
        };

        // Find the device by name
        match list_input_devices() {
            Ok(devices) => devices
                .into_iter()
                .find(|d| d.name == *device_name)
                .map(|d| d.device),
            Err(e) => {
                debug!("Failed to list devices, using default: {}", e);
                None
            }
        }
    }

    /* ---------- microphone life-cycle -------------------------------------- */

    /// Applies mute if mute_while_recording is enabled and stream is open.
    /// Skips muting (and later unmuting) if the system was already muted by the user.
    pub fn apply_mute(&self) {
        let settings = get_settings(&self.app_handle);
        if !settings.mute_while_recording {
            return;
        }
        // is_open avant did_mute (ordre des verrous), sans les imbriquer.
        let is_open = *self.is_open.lock().unwrap();
        if !is_open {
            return;
        }

        let mut did_mute_guard = self.did_mute.lock().unwrap();
        // Enregistrement déjà terminé (appui très court) : on ne coupe plus le son.
        if !self.mute_allowed.load(Ordering::SeqCst) {
            debug!("Recording already stopped, skipping mute");
            return;
        }
        if is_system_already_muted() {
            debug!("System already muted by user, skipping app mute");
            return;
        }
        set_mute(true);
        *did_mute_guard = true;
        debug!("Mute applied");
    }

    /// Removes mute if it was applied
    pub fn remove_mute(&self) {
        let mut did_mute_guard = self.did_mute.lock().unwrap();
        self.mute_allowed.store(false, Ordering::SeqCst);
        if *did_mute_guard {
            set_mute(false);
            *did_mute_guard = false;
            debug!("Mute removed");
        }
    }

    /// Le flux micro ouvert fonctionne-t-il encore ? (faux si le micro a été
    /// débranché ou si le pilote a signalé une erreur)
    fn is_stream_healthy(&self) -> bool {
        self.recorder
            .lock()
            .unwrap()
            .as_ref()
            .is_none_or(|rec| rec.is_healthy())
    }

    pub fn start_microphone_stream(&self) -> Result<(), anyhow::Error> {
        let mut open_flag = self.is_open.lock().unwrap();
        if *open_flag {
            debug!("Microphone stream already active");
            return Ok(());
        }

        let start_time = Instant::now();

        // Don't mute immediately - caller will handle muting after audio feedback
        let mut did_mute_guard = self.did_mute.lock().unwrap();
        *did_mute_guard = false;

        let vad_path = self
            .app_handle
            .path()
            .resolve(
                "resources/models/silero_vad_v4.onnx",
                tauri::path::BaseDirectory::Resource,
            )
            .map_err(|e| anyhow::anyhow!("Failed to resolve VAD path: {}", e))?;
        let vad_path = vad_path
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("VAD model path is not valid UTF-8: {:?}", vad_path))?
            .to_string();
        let mut recorder_opt = self.recorder.lock().unwrap();

        if recorder_opt.is_none() {
            *recorder_opt = Some(create_audio_recorder(
                &vad_path,
                &self.app_handle,
                Arc::clone(&self.is_paused),
            )?);
        }

        // Get the selected device from settings, considering clamshell mode
        let settings = get_settings(&self.app_handle);
        let selected_device = self.get_effective_microphone_device(&settings);

        if let Some(rec) = recorder_opt.as_mut() {
            rec.open(selected_device)
                .map_err(|e| anyhow::anyhow!("Failed to open recorder: {}", e))?;
        }

        *open_flag = true;
        info!(
            "Microphone stream initialized in {:?}",
            start_time.elapsed()
        );
        Ok(())
    }

    pub fn stop_microphone_stream(&self) {
        let mut open_flag = self.is_open.lock().unwrap();
        if !*open_flag {
            return;
        }

        let mut did_mute_guard = self.did_mute.lock().unwrap();
        if *did_mute_guard {
            set_mute(false);
        }
        *did_mute_guard = false;

        if let Some(rec) = self.recorder.lock().unwrap().as_mut() {
            // If still recording, stop first.
            if *self.is_recording.lock().unwrap() {
                let _ = rec.stop();
                *self.is_recording.lock().unwrap() = false;
            }
            let _ = rec.close();
        }

        *open_flag = false;
        debug!("Microphone stream stopped");
    }

    /* ---------- mode switching --------------------------------------------- */

    pub fn update_mode(&self, new_mode: MicrophoneMode) -> Result<(), anyhow::Error> {
        // Lu puis relâché aussitôt : prendre `state` en tenant `mode` inverserait
        // l'ordre des verrous de try_start_recording (risque d'interblocage).
        let cur_mode = self.mode.lock().unwrap().clone();

        match (cur_mode, &new_mode) {
            (MicrophoneMode::AlwaysOn, MicrophoneMode::OnDemand) => {
                let state = self.state.lock().unwrap();
                if matches!(*state, RecordingState::Idle) {
                    self.stop_microphone_stream();
                }
            }
            (MicrophoneMode::OnDemand, MicrophoneMode::AlwaysOn) => {
                self.start_microphone_stream()?;
            }
            _ => {}
        }

        *self.mode.lock().unwrap() = new_mode;
        Ok(())
    }

    /* ---------- recording --------------------------------------------------- */

    pub fn toggle_pause(&self) -> bool {
        let prev = self.is_paused.fetch_xor(true, Ordering::Relaxed);
        let new_state = !prev;
        if new_state {
            debug!("Recording paused");
        } else {
            debug!("Recording resumed");
        }
        new_state
    }

    pub fn try_start_recording(&self, binding_id: &str) -> Result<(), StartRecordingError> {
        self.is_paused.store(false, Ordering::Relaxed);
        let mut state = self.state.lock().unwrap();

        if !matches!(*state, RecordingState::Idle) {
            return Err(StartRecordingError::Busy);
        }

        let always_on = matches!(*self.mode.lock().unwrap(), MicrophoneMode::AlwaysOn);
        // Micro toujours ouvert mais tombé (débranché, erreur pilote) : on le
        // referme pour le rouvrir proprement juste après.
        if always_on && !self.is_stream_healthy() {
            warn!("Always-on microphone stream is broken, reopening it");
            self.stop_microphone_stream();
        }

        // Ouvre le micro (mode à la demande, ou mode permanent qui n'a pas pu
        // s'ouvrir au démarrage) ; ne fait rien s'il est déjà ouvert.
        if let Err(e) = self.start_microphone_stream() {
            return Err(StartRecordingError::Microphone(format!(
                "failed to open microphone stream: {e}"
            )));
        }

        let started = match self.recorder.lock().unwrap().as_ref() {
            Some(rec) => rec.start().map_err(|e| e.to_string()),
            None => Err("recorder not available".to_string()),
        };

        match started {
            Ok(()) => {
                *self.is_recording.lock().unwrap() = true;
                self.mute_allowed.store(true, Ordering::SeqCst);
                *state = RecordingState::Recording {
                    binding_id: binding_id.to_string(),
                };
                debug!("Recording started for binding {binding_id}");
                Ok(())
            }
            Err(e) => {
                // Flux inutilisable : on le ferme pour repartir de zéro au prochain essai.
                self.stop_microphone_stream();
                Err(StartRecordingError::Microphone(format!(
                    "failed to start recording: {e}"
                )))
            }
        }
    }

    /// Ferme le micro en arrière-plan après une dictée (mode à la demande).
    /// La fermeture du flux Windows prend plusieurs centaines de ms : la faire
    /// en tâche de fond évite de retarder la transcription d'autant. Si une
    /// nouvelle dictée a démarré entre-temps, le micro reste ouvert.
    fn close_stream_in_background(&self) {
        let manager = self.clone();
        std::thread::spawn(move || {
            let state = manager.state.lock().unwrap();
            if !matches!(*state, RecordingState::Idle) {
                return;
            }
            if !matches!(*manager.mode.lock().unwrap(), MicrophoneMode::OnDemand) {
                return;
            }
            manager.stop_microphone_stream();
            drop(state);
        });
    }

    pub fn update_selected_device(&self) -> Result<(), anyhow::Error> {
        // If currently open, restart the microphone stream to use the new device
        if *self.is_open.lock().unwrap() {
            self.stop_microphone_stream();
            self.start_microphone_stream()?;
        }
        Ok(())
    }

    pub fn stop_recording(&self, binding_id: &str) -> Option<Vec<f32>> {
        self.is_paused.store(false, Ordering::Relaxed);
        let mut state = self.state.lock().unwrap();

        match *state {
            RecordingState::Recording {
                binding_id: ref active,
            } if active == binding_id => {
                *state = RecordingState::Idle;
                drop(state);

                let samples = if let Some(rec) = self.recorder.lock().unwrap().as_ref() {
                    match rec.stop() {
                        Ok(buf) => buf,
                        Err(e) => {
                            error!("stop() failed: {e}");
                            Vec::new()
                        }
                    }
                } else {
                    error!("Recorder not available");
                    Vec::new()
                };

                *self.is_recording.lock().unwrap() = false;

                // In on-demand mode turn the mic off again (in the background:
                // the samples are already here, the transcription can start now)
                self.close_stream_in_background();

                // Pad if very short
                let s_len = samples.len();
                if s_len < WHISPER_SAMPLE_RATE && s_len > 0 {
                    let mut padded = samples;
                    padded.resize(WHISPER_SAMPLE_RATE * 5 / 4, 0.0);
                    Some(padded)
                } else {
                    Some(samples)
                }
            }
            _ => None,
        }
    }
    pub fn is_recording(&self) -> bool {
        matches!(
            *self.state.lock().unwrap(),
            RecordingState::Recording { .. }
        )
    }

    /// Cancel any ongoing recording without returning audio samples
    pub fn cancel_recording(&self) {
        // Le son coupé pendant l'enregistrement est rétabli aussi en cas
        // d'annulation (avant, il restait coupé en mode micro permanent).
        self.remove_mute();
        self.is_paused.store(false, Ordering::Relaxed);
        let mut state = self.state.lock().unwrap();

        if let RecordingState::Recording { .. } = *state {
            *state = RecordingState::Idle;
            drop(state);

            if let Some(rec) = self.recorder.lock().unwrap().as_ref() {
                let _ = rec.stop(); // Discard the result
            }

            *self.is_recording.lock().unwrap() = false;

            // In on-demand mode turn the mic off again
            self.close_stream_in_background();
        }
    }
}
