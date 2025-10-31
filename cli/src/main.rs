use anyhow::{anyhow, Context, Result};
use crossterm::{
    event::{DisableMouseCapture, EnableMouseCapture},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use serde_json::Value;
use std::{
    collections::HashMap,
    fs,
    fs::File,
    io::{self, BufReader},
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration as StdDuration, Instant},
};
use tokio::{
    sync::{
        mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender},
        Mutex,
    },
    time::{sleep, Duration as TokioDuration},
};
use tracing::{error, info};

use rodio::{Decoder, OutputStream, Sink, Source};

mod api;
mod app;
mod config;
mod planner;
mod session;
mod session_store;
mod types;
mod ui;

use app::{AppCommand, AppEvent, AppState, LocalArtifact};
use config::AppConfig;
use session::LAYER_ORDER;
use types::{
    ClipLayer, CompositionPlan, GenerationArtifact, GenerationRequest, JobState,
    SessionClipRequest, SessionCreateRequest,
};

const PULSE_MESSAGES: &[&str] = &[
    "Stirring the sound cauldron…",
    "Layering motifs with care…",
    "Polishing transitions, hang tight…",
    "Dialing in warmth and shimmer…",
    "Locking sections to the groove…",
];

struct AudioPlayer {
    _stream: OutputStream,
    handle: rodio::OutputStreamHandle,
    sink: Option<Sink>,
    clip_sinks: HashMap<ClipLayer, Sink>,
}

unsafe impl Send for AudioPlayer {}
unsafe impl Sync for AudioPlayer {}

impl AudioPlayer {
    fn new() -> Result<Self> {
        let (stream, handle) =
            OutputStream::try_default().context("failed to open audio output")?;
        Ok(Self { _stream: stream, handle, sink: None, clip_sinks: HashMap::new() })
    }

    fn play(&mut self, path: &Path) -> Result<()> {
        self.stop();
        let file =
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        let decoder = Decoder::new(BufReader::new(file)).context("failed to decode audio")?;
        let sink = Sink::try_new(&self.handle).context("failed to create audio sink")?;
        sink.append(decoder);
        sink.play();
        self.sink = Some(sink);
        Ok(())
    }

    fn stop(&mut self) {
        if let Some(sink) = self.sink.take() {
            sink.stop();
        }
    }

    fn play_clip(&mut self, layer: ClipLayer, path: &Path) -> Result<()> {
        self.stop_clip(layer);
        let file =
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        let decoder = Decoder::new(BufReader::new(file)).context("failed to decode audio")?;
        let sink = Sink::try_new(&self.handle).context("failed to create audio sink")?;
        let source = decoder.buffered();
        sink.append(source.repeat_infinite());
        sink.play();
        self.clip_sinks.insert(layer, sink);
        Ok(())
    }

    fn stop_clip(&mut self, layer: ClipLayer) {
        if let Some(sink) = self.clip_sinks.remove(&layer) {
            sink.stop();
        }
    }

    fn stop_all_clips(&mut self) {
        for (_, sink) in self.clip_sinks.drain() {
            sink.stop();
        }
    }

    fn is_playing(&self) -> bool {
        self.sink.as_ref().map(|sink| !sink.empty()).unwrap_or(false)
    }

    fn reset(&mut self) {
        self.stop();
        self.stop_all_clips();
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    setup_tracing()?;
    info!("starting timbre CLI");

    let config = AppConfig::load()?;
    let client = api::Client::new(config.worker_url())?;

    let (event_tx, mut event_rx) = unbounded_channel();
    let (command_tx, command_rx) = unbounded_channel();

    let controller = Controller::new(client.clone(), event_tx.clone(), config.clone())?;
    controller.spawn(command_rx);

    let mut app_state = AppState::new(config.clone());
    seed_health_status(&client, &mut app_state).await;

    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    enable_raw_mode()?;

    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;
    terminal.hide_cursor()?;

    let ui_result = ui::run(&mut terminal, &mut app_state, &mut event_rx, command_tx.clone());

    terminal.show_cursor()?;
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;

    if let Err(err) = app_state.save_session_snapshot() {
        error!("failed to save session snapshot: {err}");
    }

    ui_result
}

async fn seed_health_status(client: &api::Client, app: &mut AppState) {
    match client.health().await {
        Ok(body) => {
            app.ingest_health_payload(&body);
            let status = body.get("status").and_then(|v| v.as_str()).unwrap_or("unknown");
            let default_model = body
                .get("default_model_id")
                .and_then(|v| v.as_str())
                .unwrap_or("musicgen-stereo-medium");
            let musicgen_default = body
                .get("musicgen_default_model_id")
                .and_then(|v| v.as_str())
                .unwrap_or(default_model);
            let artifact_root =
                body.get("artifact_root").and_then(|v| v.as_str()).unwrap_or("~/Music/Timbre");
            let planner_version =
                body.get("planner_version").and_then(|v| v.as_str()).unwrap_or("-");
            let backends = body
                .get("available_backends")
                .and_then(|v| v.as_array())
                .map(|items| items.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>().join(", "))
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "unknown".to_string());
            let base_url = client.base_url().to_string();
            app.handle_event(AppEvent::Info(format!(
                "Worker health: {status} (default model {musicgen_default}; planner {planner_version}; backends: {backends}; artifacts: {artifact_root}) @ {base_url}"
            )));
        }
        Err(err) => {
            app.handle_event(AppEvent::Error(format!("Worker health check failed: {err}")));
        }
    }
}

fn setup_tracing() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .compact()
        .try_init()
        .map_err(|err: Box<dyn std::error::Error + Send + Sync>| {
            anyhow!("failed to initialise tracing: {err}")
        })?;
    Ok(())
}

struct Controller {
    inner: Arc<ControllerInner>,
}

struct ClipClock {
    tempo_bpm: u16,
    beats_per_bar: f32,
    last_launch: Instant,
}

struct ControllerInner {
    client: api::Client,
    event_tx: UnboundedSender<AppEvent>,
    config: AppConfig,
    artifact_paths: Mutex<HashMap<String, PathBuf>>,
    session_jobs: Mutex<HashMap<String, String>>,
    player: Mutex<AudioPlayer>,
    clip_clock: Mutex<Option<ClipClock>>,
}

impl Controller {
    fn new(
        client: api::Client,
        event_tx: UnboundedSender<AppEvent>,
        config: AppConfig,
    ) -> Result<Self> {
        let player = AudioPlayer::new()?;
        let inner = ControllerInner {
            client,
            event_tx,
            config,
            artifact_paths: Mutex::new(HashMap::new()),
            session_jobs: Mutex::new(HashMap::new()),
            player: Mutex::new(player),
            clip_clock: Mutex::new(None),
        };
        Ok(Self { inner: Arc::new(inner) })
    }

    fn spawn(self, mut command_rx: UnboundedReceiver<AppCommand>) {
        let inner = self.inner.clone();
        tokio::spawn(async move {
            while let Some(command) = command_rx.recv().await {
                if let Err(err) = Controller::handle_command(inner.clone(), command).await {
                    error!("command error: {err}");
                    let _ = inner.event_tx.send(AppEvent::Error(format!("{err}")));
                }
            }
        });
    }

    async fn handle_command(inner: Arc<ControllerInner>, command: AppCommand) -> Result<()> {
        match command {
            AppCommand::SubmitPrompt { prompt, request, plan } => {
                Controller::submit_prompt(inner, prompt, request, plan).await?;
            }
            AppCommand::SubmitClip { prompt, layer, request, plan } => {
                Controller::submit_clip(inner, prompt, layer, request, plan).await?;
            }
            AppCommand::PlayJob { job_id } => {
                Controller::play_job(inner, job_id).await?;
            }
            AppCommand::StopPlayback => {
                Controller::stop_playback(inner).await?;
            }
            AppCommand::CreateSession { payload } => {
                Controller::create_session(inner, payload).await?;
            }
            AppCommand::FetchSession { session_id } => {
                Controller::refresh_session(inner, session_id).await?;
            }
            AppCommand::LaunchClip { layer, scene_index, path, tempo_bpm, beats_per_bar } => {
                Controller::launch_clip(inner, layer, scene_index, path, tempo_bpm, beats_per_bar)
                    .await?;
            }
            AppCommand::StopClip { layer } => {
                Controller::stop_clip(inner, layer).await?;
            }
            AppCommand::StopAllClips => {
                Controller::stop_all_clips(inner).await?;
            }
        }
        Ok(())
    }

    async fn submit_prompt(
        inner: Arc<ControllerInner>,
        prompt: String,
        request: GenerationRequest,
        plan: CompositionPlan,
    ) -> Result<()> {
        let status = inner
            .client
            .submit_generation(&request)
            .await
            .context("failed to submit generation request")?;
        {
            let mut player = inner.player.lock().await;
            player.reset();
        }

        if let Some(session_id) = request.session_id.clone() {
            let mut jobs = inner.session_jobs.lock().await;
            jobs.insert(status.job_id.clone(), session_id);
        }

        let _ = inner.event_tx.send(AppEvent::JobQueued {
            status: status.clone(),
            prompt,
            request,
            plan,
        });

        Controller::spawn_poll_task(inner, status.job_id.clone());
        Ok(())
    }

    async fn submit_clip(
        inner: Arc<ControllerInner>,
        prompt: String,
        layer: ClipLayer,
        request: GenerationRequest,
        plan: CompositionPlan,
    ) -> Result<()> {
        let session_id = request
            .session_id
            .clone()
            .ok_or_else(|| anyhow!("clip requests require an active session"))?;

        let payload = SessionClipRequest {
            layer,
            prompt: Some(prompt.clone()),
            bars: request.clip_bars,
            scene_index: request.clip_scene_index,
        };

        let status = inner
            .client
            .submit_session_clip(&session_id, &payload)
            .await
            .context("failed to submit session clip")?;

        {
            let mut jobs = inner.session_jobs.lock().await;
            jobs.insert(status.job_id.clone(), session_id.clone());
        }

        let _ = inner.event_tx.send(AppEvent::JobQueued {
            status: status.clone(),
            prompt,
            request,
            plan,
        });

        Controller::spawn_poll_task(inner, status.job_id.clone());
        Ok(())
    }

    fn spawn_poll_task(inner: Arc<ControllerInner>, job_id: String) {
        tokio::spawn(async move {
            if let Err(err) = Controller::poll_job(inner.clone(), job_id.clone()).await {
                error!("job {job_id} polling error: {err}");
                let _ = inner
                    .event_tx
                    .send(AppEvent::Error(format!("Failed to poll job {job_id}: {err}")));
            }
        });
    }

    async fn create_session(
        inner: Arc<ControllerInner>,
        payload: SessionCreateRequest,
    ) -> Result<()> {
        let summary =
            inner.client.create_session(&payload).await.context("failed to create session")?;
        let _ = inner.event_tx.send(AppEvent::SessionStarted { summary });
        Ok(())
    }

    async fn refresh_session(inner: Arc<ControllerInner>, session_id: String) -> Result<()> {
        let summary =
            inner.client.session_summary(&session_id).await.context("failed to refresh session")?;
        let _ = inner.event_tx.send(AppEvent::SessionUpdated { summary });
        Ok(())
    }

    async fn launch_clip(
        inner: Arc<ControllerInner>,
        layer: ClipLayer,
        scene_index: usize,
        path: PathBuf,
        tempo_bpm: u16,
        beats_per_bar: f32,
    ) -> Result<()> {
        let quant_seconds = (60.0_f32 / tempo_bpm.max(1) as f32) * beats_per_bar.max(1.0);
        let mut wait_duration = StdDuration::default();

        {
            let mut clock = inner.clip_clock.lock().await;
            let now = Instant::now();
            if quant_seconds <= f32::EPSILON {
                *clock = Some(ClipClock { tempo_bpm, beats_per_bar, last_launch: now });
            } else if let Some(clock_state) = clock.as_mut() {
                if clock_state.tempo_bpm != tempo_bpm
                    || (clock_state.beats_per_bar - beats_per_bar).abs() > f32::EPSILON
                {
                    *clock_state = ClipClock { tempo_bpm, beats_per_bar, last_launch: now };
                } else {
                    let elapsed = now.saturating_duration_since(clock_state.last_launch);
                    let elapsed_secs = elapsed.as_secs_f32();
                    let remainder = elapsed_secs.rem_euclid(quant_seconds);
                    if remainder > 0.01 {
                        let wait_secs = quant_seconds - remainder;
                        wait_duration = StdDuration::from_secs_f32(wait_secs);
                        clock_state.last_launch = now + wait_duration;
                    } else {
                        clock_state.last_launch = now;
                    }
                }
            } else {
                *clock = Some(ClipClock { tempo_bpm, beats_per_bar, last_launch: now });
            }
        }

        let inner_clone = inner.clone();
        tokio::spawn(async move {
            if !wait_duration.is_zero() {
                sleep(TokioDuration::from_secs_f64(wait_duration.as_secs_f64())).await;
            }
            let mut player = inner_clone.player.lock().await;
            if let Err(err) = player.play_clip(layer, &path) {
                error!("failed to launch clip: {err}");
                let _ = inner_clone
                    .event_tx
                    .send(AppEvent::Error(format!("Failed to play clip: {err}")));
            } else {
                let _ = inner_clone.event_tx.send(AppEvent::ClipPlaying { layer, scene_index });
            }
        });

        Ok(())
    }

    async fn stop_clip(inner: Arc<ControllerInner>, layer: ClipLayer) -> Result<()> {
        let mut player = inner.player.lock().await;
        player.stop_clip(layer);
        let _ = inner.event_tx.send(AppEvent::ClipStopped { layer });
        Ok(())
    }

    async fn stop_all_clips(inner: Arc<ControllerInner>) -> Result<()> {
        let mut player = inner.player.lock().await;
        player.stop_all_clips();
        drop(player);
        for layer in LAYER_ORDER {
            let _ = inner.event_tx.send(AppEvent::ClipStopped { layer });
        }
        let mut clock = inner.clip_clock.lock().await;
        *clock = None;
        Ok(())
    }

    async fn take_session_job(inner: Arc<ControllerInner>, job_id: &str) -> Option<String> {
        let mut guard = inner.session_jobs.lock().await;
        guard.remove(job_id)
    }

    async fn poll_job(inner: Arc<ControllerInner>, job_id: String) -> Result<()> {
        let mut attempt = 0u32;
        loop {
            let delay = TokioDuration::from_millis(400).saturating_mul((attempt + 1).min(5));
            sleep(delay).await;
            attempt = attempt.saturating_add(1);

            let status = match inner.client.job_status(&job_id).await {
                Ok(status) => status,
                Err(err) => {
                    error!("status poll failed for {job_id}: {err}");
                    let _ = inner
                        .event_tx
                        .send(AppEvent::Error(format!("Status poll failed for {job_id}: {err}")));
                    continue;
                }
            };

            let progress_ratio = status.progress.clamp(0.0, 1.0);
            let pulse_message = pulse_message(attempt);
            let _ = inner.event_tx.send(AppEvent::PollProgress {
                progress: progress_ratio,
                message: pulse_message.to_string(),
            });
            let friendly_update =
                format!("{} ({}%)", pulse_message, (progress_ratio * 100.0).round() as i32);
            let _ = inner.event_tx.send(AppEvent::WorkerNudge { message: friendly_update });

            let _ = inner.event_tx.send(AppEvent::JobUpdated { status: status.clone() });

            match status.state {
                JobState::Succeeded => {
                    let session_binding =
                        Controller::take_session_job(inner.clone(), &job_id).await;
                    let artifact = inner
                        .client
                        .fetch_artifact(&job_id)
                        .await
                        .context("failed to fetch artifact metadata")?;
                    let local = persist_artifact(&inner.config, artifact).await?;
                    {
                        let mut guard = inner.artifact_paths.lock().await;
                        guard.insert(job_id.clone(), local.local_path.clone());
                    }
                    let _ = inner.event_tx.send(AppEvent::JobCompleted { status, artifact: local });
                    if session_binding.is_none() {
                        if let Some(path) = inner.artifact_paths.lock().await.get(&job_id).cloned()
                        {
                            if let Ok(duration) = audio_duration(&path) {
                                let mut player = inner.player.lock().await;
                                if let Err(err) = player.play(&path) {
                                    error!("failed to start playback: {err}");
                                } else {
                                    drop(player);
                                    let _ = inner.event_tx.send(AppEvent::PlaybackStarted {
                                        job_id: job_id.clone(),
                                        path,
                                        duration,
                                    });
                                    Controller::spawn_playback_monitor(inner.clone());
                                }
                            }
                        }
                    }
                    let _ = inner.event_tx.send(AppEvent::PollProgress {
                        progress: 1.0,
                        message: "Render complete — ready for playback".to_string(),
                    });
                    if let Some(session_id) = session_binding {
                        if let Err(err) =
                            Controller::refresh_session(inner.clone(), session_id.clone()).await
                        {
                            error!("failed to refresh session {session_id}: {err}");
                            let _ = inner.event_tx.send(AppEvent::Error(format!(
                                "Failed to refresh session {session_id}: {err}"
                            )));
                        }
                    }
                    break;
                }
                JobState::Failed => {
                    let session_id = Controller::take_session_job(inner.clone(), &job_id).await;
                    if let Some(session_id) = session_id {
                        if let Err(err) =
                            Controller::refresh_session(inner.clone(), session_id.clone()).await
                        {
                            error!("failed to refresh session {session_id}: {err}");
                        }
                    }
                    break;
                }
                _ => {}
            }
        }
        Ok(())
    }

    async fn play_job(inner: Arc<ControllerInner>, job_id: String) -> Result<()> {
        let path = {
            let guard = inner.artifact_paths.lock().await;
            guard.get(&job_id).cloned()
        };

        let Some(path) = path else {
            let _ = inner
                .event_tx
                .send(AppEvent::Error(format!("No artifact available for job {job_id}")));
            return Ok(());
        };

        let _ = inner.event_tx.send(AppEvent::Info(format!(
            "Artifact ready at {} (open manually to preview)",
            path.display()
        )));

        let mut player = inner.player.lock().await;
        if let Err(err) = player.play(&path) {
            let _ = inner
                .event_tx
                .send(AppEvent::Error(format!("Failed to play {}: {err}", path.display())));
        } else if let Ok(duration) = audio_duration(&path) {
            drop(player);
            let _ = inner.event_tx.send(AppEvent::PlaybackStarted { job_id, path, duration });
            Controller::spawn_playback_monitor(inner.clone());
        }

        Ok(())
    }

    async fn stop_playback(inner: Arc<ControllerInner>) -> Result<()> {
        let mut player = inner.player.lock().await;
        player.stop();
        let _ = inner.event_tx.send(AppEvent::PlaybackStopped);
        Ok(())
    }

    fn spawn_playback_monitor(inner: Arc<ControllerInner>) {
        tokio::spawn(async move {
            loop {
                sleep(TokioDuration::from_millis(500)).await;
                let playing = {
                    let player = inner.player.lock().await;
                    player.is_playing()
                };
                let _ = inner.event_tx.send(AppEvent::PlaybackProgress { is_playing: playing });
                if !playing {
                    let _ = inner.event_tx.send(AppEvent::PlaybackStopped);
                    break;
                }
            }
        });
    }
}

fn pulse_message(attempt: u32) -> &'static str {
    if PULSE_MESSAGES.is_empty() {
        return "Working on it…";
    }
    let index = (attempt as usize) % PULSE_MESSAGES.len();
    PULSE_MESSAGES[index]
}

fn audio_duration(path: &Path) -> Result<StdDuration> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let decoder = Decoder::new(BufReader::new(file)).context("failed to decode audio")?;
    decoder
        .total_duration()
        .ok_or_else(|| anyhow!("unable to determine duration for {}", path.display()))
}

async fn persist_artifact(
    config: &AppConfig,
    artifact: GenerationArtifact,
) -> Result<LocalArtifact> {
    let artifact_dir = config.artifact_dir().clone();
    tokio::task::spawn_blocking(move || -> Result<LocalArtifact> {
        let source_path = PathBuf::from(&artifact.artifact_path);
        if !source_path.exists() {
            return Err(anyhow!("artifact path does not exist: {}", source_path.display()));
        }

        fs::create_dir_all(&artifact_dir)
            .with_context(|| format!("failed to create artifact dir {}", artifact_dir.display()))?;
        let job_dir = artifact_dir.join(&artifact.job_id);
        fs::create_dir_all(&job_dir)
            .with_context(|| format!("failed to create job dir {}", job_dir.display()))?;

        let file_name = source_path
            .file_name()
            .map(|name| name.to_owned())
            .unwrap_or_else(|| format!("{}.wav", artifact.job_id).into());
        let target_path = job_dir.join(file_name);

        if source_path != target_path {
            fs::copy(&source_path, &target_path)
                .with_context(|| format!("failed to copy artifact to {}", target_path.display()))?;
        }

        let mut descriptor = artifact;
        descriptor.artifact_path = target_path.to_string_lossy().to_string();

        let mut extras_map = descriptor.metadata.extras.as_object().cloned().unwrap_or_default();
        extras_map
            .insert("local_path".to_string(), Value::String(descriptor.artifact_path.clone()));
        descriptor.metadata.extras = Value::Object(extras_map);

        let metadata_path = job_dir.join("metadata.json");
        let metadata_json =
            serde_json::to_vec_pretty(&descriptor).context("failed to encode artifact metadata")?;
        fs::write(&metadata_path, metadata_json)
            .with_context(|| format!("failed to write metadata at {}", metadata_path.display()))?;

        Ok(LocalArtifact { descriptor, local_path: target_path })
    })
    .await
    .context("artifact persistence task panicked")?
}
