use std::{
    io::Error,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Mutex,
    },
    time::Duration,
};

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    Device, Sample, SizedSample,
};

use crate::audio_toolkit::{
    audio::{AudioVisualiser, FrameResampler},
    constants,
    vad::{self, VadFrame},
    VoiceActivityDetector,
};

enum Cmd {
    Start,
    Stop(mpsc::Sender<Vec<f32>>),
    Shutdown,
}

/// Délai maximal d'ouverture du flux micro avant de déclarer l'échec.
const OPEN_TIMEOUT: Duration = Duration::from_secs(5);
/// Délai maximal pour récupérer l'audio à l'arrêt : sans lui, un micro qui
/// n'envoie plus rien (casque débranché) bloquait l'application pour toujours.
const STOP_TIMEOUT: Duration = Duration::from_secs(3);
/// Le fil de traitement vérifie les commandes au moins à ce rythme, même
/// quand le micro ne produit plus d'audio.
const COMMAND_POLL_INTERVAL: Duration = Duration::from_millis(100);

pub struct AudioRecorder {
    device: Option<Device>,
    cmd_tx: Option<mpsc::Sender<Cmd>>,
    worker_handle: Option<std::thread::JoinHandle<()>>,
    vad: Option<Arc<Mutex<Box<dyn vad::VoiceActivityDetector>>>>,
    level_cb: Option<Arc<dyn Fn(Vec<f32>) + Send + Sync + 'static>>,
    pause_flag: Option<Arc<AtomicBool>>,
    /// Passe à faux si le pilote signale une erreur de flux (micro débranché…).
    healthy: Arc<AtomicBool>,
}

impl AudioRecorder {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(AudioRecorder {
            device: None,
            cmd_tx: None,
            worker_handle: None,
            vad: None,
            level_cb: None,
            pause_flag: None,
            healthy: Arc::new(AtomicBool::new(true)),
        })
    }

    /// Le flux ouvert fonctionne-t-il encore ?
    pub fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Relaxed)
    }

    pub fn with_vad(mut self, vad: Box<dyn VoiceActivityDetector>) -> Self {
        self.vad = Some(Arc::new(Mutex::new(vad)));
        self
    }

    pub fn with_level_callback<F>(mut self, cb: F) -> Self
    where
        F: Fn(Vec<f32>) + Send + Sync + 'static,
    {
        self.level_cb = Some(Arc::new(cb));
        self
    }

    pub fn with_pause_flag(mut self, flag: Arc<AtomicBool>) -> Self {
        self.pause_flag = Some(flag);
        self
    }

    pub fn open(&mut self, device: Option<Device>) -> Result<(), Box<dyn std::error::Error>> {
        if self.worker_handle.is_some() {
            return Ok(()); // already open
        }

        let (sample_tx, sample_rx) = mpsc::channel::<Vec<f32>>();
        let (cmd_tx, cmd_rx) = mpsc::channel::<Cmd>();

        let host = crate::audio_toolkit::get_cpal_host();
        let device = match device {
            Some(dev) => dev,
            None => host
                .default_input_device()
                .ok_or_else(|| Error::new(std::io::ErrorKind::NotFound, "No input device found"))?,
        };

        let thread_device = device.clone();
        let vad = self.vad.clone();
        // Move the optional level callback into the worker thread
        let level_cb = self.level_cb.clone();
        let pause_flag = self.pause_flag.clone();
        let healthy = Arc::clone(&self.healthy);
        healthy.store(true, Ordering::Relaxed);

        // Le fil signale le résultat de l'ouverture : une erreur (micro absent,
        // format non géré…) est renvoyée à l'appelant au lieu de faire paniquer
        // le fil en silence et de laisser croire que le micro est ouvert.
        let (init_tx, init_rx) = mpsc::channel::<Result<(), String>>();

        let worker = std::thread::spawn(move || {
            let stream =
                match AudioRecorder::open_stream(&thread_device, sample_tx, Arc::clone(&healthy)) {
                    Ok(opened) => opened,
                    Err(e) => {
                        let _ = init_tx.send(Err(e));
                        return;
                    }
                };
            let (stream, sample_rate) = stream;
            let _ = init_tx.send(Ok(()));

            // keep the stream alive while we process samples
            run_consumer(sample_rate, vad, sample_rx, cmd_rx, level_cb, pause_flag);
            // stream is dropped here, after run_consumer returns
            drop(stream);
        });

        match init_rx.recv_timeout(OPEN_TIMEOUT) {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                let _ = worker.join();
                return Err(Box::new(Error::other(e)));
            }
            Err(_) => {
                // Le fil est bloqué ou mort : on ne garde rien, il se terminera
                // de lui-même quand ses canaux seront fermés.
                drop(cmd_tx);
                return Err(Box::new(Error::new(
                    std::io::ErrorKind::TimedOut,
                    "microphone did not open in time",
                )));
            }
        }

        self.device = Some(device);
        self.cmd_tx = Some(cmd_tx);
        self.worker_handle = Some(worker);

        Ok(())
    }

    /// Ouvre et démarre le flux d'entrée ; renvoie le flux et sa fréquence.
    fn open_stream(
        device: &Device,
        sample_tx: mpsc::Sender<Vec<f32>>,
        healthy: Arc<AtomicBool>,
    ) -> Result<(cpal::Stream, u32), String> {
        let config = AudioRecorder::get_preferred_config(device)
            .map_err(|e| format!("failed to fetch preferred config: {e}"))?;

        let sample_rate = config.sample_rate().0;
        let channels = config.channels() as usize;

        log::info!(
            "Using device: {:?}\nSample rate: {}\nChannels: {}\nFormat: {:?}",
            device.name(),
            sample_rate,
            channels,
            config.sample_format()
        );

        let stream = match config.sample_format() {
            cpal::SampleFormat::U8 => {
                AudioRecorder::build_stream::<u8>(device, &config, sample_tx, channels, healthy)
            }
            cpal::SampleFormat::I8 => {
                AudioRecorder::build_stream::<i8>(device, &config, sample_tx, channels, healthy)
            }
            cpal::SampleFormat::I16 => {
                AudioRecorder::build_stream::<i16>(device, &config, sample_tx, channels, healthy)
            }
            cpal::SampleFormat::I32 => {
                AudioRecorder::build_stream::<i32>(device, &config, sample_tx, channels, healthy)
            }
            cpal::SampleFormat::F32 => {
                AudioRecorder::build_stream::<f32>(device, &config, sample_tx, channels, healthy)
            }
            other => return Err(format!("unsupported sample format: {other:?}")),
        }
        .map_err(|e| format!("failed to build input stream: {e}"))?;

        stream
            .play()
            .map_err(|e| format!("failed to start stream: {e}"))?;

        Ok((stream, sample_rate))
    }

    pub fn start(&self) -> Result<(), Box<dyn std::error::Error>> {
        match &self.cmd_tx {
            Some(tx) => tx.send(Cmd::Start)?,
            None => return Err("recorder is not open".into()),
        }
        Ok(())
    }

    pub fn stop(&self) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let (resp_tx, resp_rx) = mpsc::channel();
        match &self.cmd_tx {
            Some(tx) => tx.send(Cmd::Stop(resp_tx))?,
            None => return Err("recorder is not open".into()),
        }
        // wait for the samples, but never forever
        Ok(resp_rx.recv_timeout(STOP_TIMEOUT)?)
    }

    pub fn close(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(tx) = self.cmd_tx.take() {
            let _ = tx.send(Cmd::Shutdown);
        }
        if let Some(h) = self.worker_handle.take() {
            let _ = h.join();
        }
        self.device = None;
        Ok(())
    }

    fn build_stream<T>(
        device: &cpal::Device,
        config: &cpal::SupportedStreamConfig,
        sample_tx: mpsc::Sender<Vec<f32>>,
        channels: usize,
        healthy: Arc<AtomicBool>,
    ) -> Result<cpal::Stream, cpal::BuildStreamError>
    where
        T: Sample + SizedSample + Send + 'static,
        f32: cpal::FromSample<T>,
    {
        let mut output_buffer: Vec<f32> = Vec::new();

        let stream_cb = move |data: &[T], _: &cpal::InputCallbackInfo| {
            // Nouveau tampon à chaque appel : il est envoyé tel quel (sans copie).
            output_buffer.reserve(data.len() / channels.max(1));

            if channels == 1 {
                // Direct conversion without intermediate Vec
                output_buffer.extend(data.iter().map(|&sample| sample.to_sample::<f32>()));
            } else {
                // Convert to mono directly
                let frame_count = data.len() / channels;
                output_buffer.reserve(frame_count);

                for frame in data.chunks_exact(channels) {
                    let mono_sample = frame
                        .iter()
                        .map(|&sample| sample.to_sample::<f32>())
                        .sum::<f32>()
                        / channels as f32;
                    output_buffer.push(mono_sample);
                }
            }

            if sample_tx.send(std::mem::take(&mut output_buffer)).is_err() {
                log::error!("Failed to send samples");
            }
        };

        device.build_input_stream(
            &config.clone().into(),
            stream_cb,
            move |err| {
                log::error!("Stream error: {}", err);
                healthy.store(false, Ordering::Relaxed);
            },
            None,
        )
    }

    fn get_preferred_config(
        device: &cpal::Device,
    ) -> Result<cpal::SupportedStreamConfig, Box<dyn std::error::Error>> {
        let supported_configs = device.supported_input_configs()?;
        let mut best_config: Option<cpal::SupportedStreamConfigRange> = None;

        // Try to find a config that supports 16kHz, prioritizing better formats
        for config_range in supported_configs {
            if config_range.min_sample_rate().0 <= constants::WHISPER_SAMPLE_RATE
                && config_range.max_sample_rate().0 >= constants::WHISPER_SAMPLE_RATE
            {
                match best_config {
                    None => best_config = Some(config_range),
                    Some(ref current) => {
                        // Prioritize F32 > I16 > I32 > others
                        let score = |fmt: cpal::SampleFormat| match fmt {
                            cpal::SampleFormat::F32 => 4,
                            cpal::SampleFormat::I16 => 3,
                            cpal::SampleFormat::I32 => 2,
                            _ => 1,
                        };

                        if score(config_range.sample_format()) > score(current.sample_format()) {
                            best_config = Some(config_range);
                        }
                    }
                }
            }
        }

        if let Some(config) = best_config {
            return Ok(config.with_sample_rate(cpal::SampleRate(constants::WHISPER_SAMPLE_RATE)));
        }

        // If no config supports 16kHz, fall back to default
        Ok(device.default_input_config()?)
    }
}

fn run_consumer(
    in_sample_rate: u32,
    vad: Option<Arc<Mutex<Box<dyn vad::VoiceActivityDetector>>>>,
    sample_rx: mpsc::Receiver<Vec<f32>>,
    cmd_rx: mpsc::Receiver<Cmd>,
    level_cb: Option<Arc<dyn Fn(Vec<f32>) + Send + Sync + 'static>>,
    pause_flag: Option<Arc<AtomicBool>>,
) {
    let mut frame_resampler = FrameResampler::new(
        in_sample_rate as usize,
        constants::WHISPER_SAMPLE_RATE as usize,
        Duration::from_millis(30),
    );

    let mut processed_samples = Vec::<f32>::new();
    let mut recording = false;

    // ---------- spectrum visualisation setup ---------------------------- //
    const BUCKETS: usize = 16;
    const WINDOW_SIZE: usize = 512;
    let mut visualizer = AudioVisualiser::new(
        in_sample_rate,
        WINDOW_SIZE,
        BUCKETS,
        400.0,  // vocal_min_hz
        4000.0, // vocal_max_hz
    );

    fn handle_frame(
        samples: &[f32],
        recording: bool,
        vad: &Option<Arc<Mutex<Box<dyn vad::VoiceActivityDetector>>>>,
        out_buf: &mut Vec<f32>,
    ) {
        if !recording {
            return;
        }

        if let Some(vad_arc) = vad {
            let mut det = vad_arc
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            match det.push_frame(samples).unwrap_or(VadFrame::Speech(samples)) {
                VadFrame::Speech(buf) => out_buf.extend_from_slice(buf),
                VadFrame::Noise => {}
            }
        } else {
            out_buf.extend_from_slice(samples);
        }
    }

    loop {
        // Attente bornée : les commandes (dont Stop) sont traitées même si le
        // micro n'envoie plus rien, au lieu de bloquer l'arrêt indéfiniment.
        match sample_rx.recv_timeout(COMMAND_POLL_INTERVAL) {
            Ok(raw) => {
                // ---------- spectrum processing -------------------------- //
                if let Some(buckets) = visualizer.feed(&raw) {
                    if let Some(cb) = &level_cb {
                        cb(buckets);
                    }
                }

                // ---------- existing pipeline ---------------------------- //
                let is_paused = pause_flag
                    .as_ref()
                    .is_some_and(|f| f.load(Ordering::Relaxed));
                if !is_paused {
                    frame_resampler.push(&raw, &mut |frame: &[f32]| {
                        handle_frame(frame, recording, &vad, &mut processed_samples)
                    });
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break, // stream closed
        }

        // non-blocking check for a command
        loop {
            let cmd = match cmd_rx.try_recv() {
                Ok(cmd) => cmd,
                Err(mpsc::TryRecvError::Empty) => break,
                // Plus personne pour piloter l'enregistreur : on libère le micro.
                Err(mpsc::TryRecvError::Disconnected) => return,
            };
            match cmd {
                Cmd::Start => {
                    processed_samples.clear();
                    recording = true;
                    visualizer.reset(); // Reset visualization buffer
                    if let Some(v) = &vad {
                        v.lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .reset();
                    }
                }
                Cmd::Stop(reply_tx) => {
                    recording = false;

                    // Drain any audio chunks that were captured but not yet consumed
                    while let Ok(remaining) = sample_rx.try_recv() {
                        frame_resampler.push(&remaining, &mut |frame: &[f32]| {
                            handle_frame(frame, true, &vad, &mut processed_samples)
                        });
                    }

                    frame_resampler.finish(&mut |frame: &[f32]| {
                        handle_frame(frame, true, &vad, &mut processed_samples)
                    });

                    let _ = reply_tx.send(std::mem::take(&mut processed_samples));
                }
                Cmd::Shutdown => return,
            }
        }
    }
}
