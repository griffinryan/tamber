use anyhow::{anyhow, Context, Result};
use crossterm::{
    event::{DisableMouseCapture, EnableMouseCapture},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use serde_json::Value;
use std::{collections::HashMap, fs, io, path::PathBuf, sync::Arc};
use tokio::{
    sync::{
        mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender},
        Mutex,
    },
    time::{sleep, Duration},
};
use tracing::{error, info};

mod api;
mod app;
mod config;
mod planner;
mod types;
mod ui;

use app::{AppCommand, AppEvent, AppState, LocalArtifact};
use config::AppConfig;
use types::{CompositionPlan, GenerationArtifact, GenerationRequest, JobState};

const PULSE_MESSAGES: &[&str] = &[
    "Stirring the sound cauldron…",
    "Layering motifs with care…",
    "Polishing transitions, hang tight…",
    "Dialing in warmth and shimmer…",
    "Locking sections to the groove…",
];

#[tokio::main]
async fn main() -> Result<()> {
    setup_tracing()?;
    info!("starting timbre CLI");

    let config = AppConfig::load()?;
    let client = api::Client::new(config.worker_url())?;

    let (event_tx, mut event_rx) = unbounded_channel();
    let (command_tx, command_rx) = unbounded_channel();

    Controller::new(client.clone(), event_tx.clone(), config.clone()).spawn(command_rx);

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

    ui_result
}

async fn seed_health_status(client: &api::Client, app: &mut AppState) {
    match client.health().await {
        Ok(body) => {
            let status = body.get("status").and_then(|v| v.as_str()).unwrap_or("unknown");
            let default_model =
                body.get("default_model_id").and_then(|v| v.as_str()).unwrap_or("riffusion-v1");
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
                "Worker health: {status} (model: {default_model}, planner {planner_version}, backends: {backends}, artifacts: {artifact_root}) @ {base_url}"
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

struct ControllerInner {
    client: api::Client,
    event_tx: UnboundedSender<AppEvent>,
    config: AppConfig,
    artifact_paths: Mutex<HashMap<String, PathBuf>>,
}

impl Controller {
    fn new(client: api::Client, event_tx: UnboundedSender<AppEvent>, config: AppConfig) -> Self {
        let inner = ControllerInner {
            client,
            event_tx,
            config,
            artifact_paths: Mutex::new(HashMap::new()),
        };
        Self { inner: Arc::new(inner) }
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
            AppCommand::PlayJob { job_id } => {
                Controller::play_job(inner, job_id).await?;
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

    async fn poll_job(inner: Arc<ControllerInner>, job_id: String) -> Result<()> {
        let mut attempt = 0u32;
        loop {
            let delay = Duration::from_millis(400).saturating_mul((attempt + 1).min(5));
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
                    let _ = inner.event_tx.send(AppEvent::PollProgress {
                        progress: 1.0,
                        message: "Render complete — ready for playback".to_string(),
                    });
                    break;
                }
                JobState::Failed => break,
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

        Ok(())
    }
}

fn pulse_message(attempt: u32) -> &'static str {
    if PULSE_MESSAGES.is_empty() {
        return "Working on it…";
    }
    let index = (attempt as usize) % PULSE_MESSAGES.len();
    PULSE_MESSAGES[index]
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
