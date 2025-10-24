use crate::{
    config::AppConfig,
    planner::CompositionPlanner,
    types::{CompositionPlan, GenerationArtifact, GenerationRequest, GenerationStatus, JobState},
};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use indexmap::{map::Iter, IndexMap};
use serde_json::Value;
use std::path::PathBuf;
use std::time::Duration as StdDuration;

const MAX_CHAT_ENTRIES: usize = 200;
const MAX_STATUS_LINES: usize = 8;

#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub model_id: String,
    pub duration_seconds: u8,
    pub cfg_scale: Option<f32>,
    pub seed: Option<u64>,
}

impl GenerationConfig {
    fn from_app_config(config: &AppConfig) -> Self {
        Self {
            model_id: config.default_model_id().to_string(),
            duration_seconds: config.default_duration_seconds(),
            cfg_scale: None,
            seed: None,
        }
    }

    fn summary(&self) -> String {
        let cfg_text = self
            .cfg_scale
            .map(|value| format!("CFG {:.1}", value))
            .unwrap_or_else(|| "CFG auto".to_string());
        let seed_text = self
            .seed
            .map(|value| format!("Seed {value}"))
            .unwrap_or_else(|| "Seed auto".to_string());
        format!("{} · {}s · {} · {}", self.model_id, self.duration_seconds, cfg_text, seed_text)
    }
}

#[derive(Debug, Clone)]
pub enum ChatRole {
    User,
    Worker,
    System,
}

impl ChatRole {
    pub fn label(&self) -> &'static str {
        match self {
            Self::User => "You",
            Self::Worker => "Worker",
            Self::System => "System",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChatEntry {
    pub role: ChatRole,
    pub content: String,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct LocalArtifact {
    pub descriptor: GenerationArtifact,
    pub local_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct BackendHealthStatus {
    pub name: String,
    pub ready: bool,
    pub device: Option<String>,
    pub dtype: Option<String>,
    pub error: Option<String>,
    pub details: Vec<(String, String)>,
    pub updated_at: Option<String>,
}

impl BackendHealthStatus {
    fn from_json(name: &str, value: &Value) -> Self {
        let ready = value.get("ready").and_then(Value::as_bool).unwrap_or(false);
        let device = value.get("device").and_then(Value::as_str).map(ToString::to_string);
        let dtype = value.get("dtype").and_then(Value::as_str).map(ToString::to_string);
        let error = value.get("error").and_then(Value::as_str).map(ToString::to_string);
        let updated_at = value.get("updated_at").and_then(Value::as_str).map(ToString::to_string);
        let details = value
            .get("details")
            .and_then(Value::as_object)
            .map(|map| {
                map.iter()
                    .map(|(key, val)| (key.clone(), format_detail_value(val)))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        Self { name: name.to_string(), ready, device, dtype, error, details, updated_at }
    }

    pub fn summary(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        if let Some(device) = &self.device {
            parts.push(format!("dev {device}"));
        }
        if let Some(dtype) = &self.dtype {
            parts.push(format!("dtype {dtype}"));
        }
        if parts.is_empty() {
            if self.ready {
                parts.push("ready".to_string());
            } else {
                parts.push("warming".to_string());
            }
        }
        parts.join(", ")
    }
}

fn format_detail_value(value: &Value) -> String {
    match value {
        Value::Bool(v) => v.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => s.clone(),
        Value::Null => "null".to_string(),
        Value::Array(items) => {
            let joined = items.iter().map(format_detail_value).collect::<Vec<_>>().join(", ");
            format!("[{joined}]")
        }
        Value::Object(map) => {
            let joined = map
                .iter()
                .map(|(k, v)| format!("{k}: {}", format_detail_value(v)))
                .collect::<Vec<_>>()
                .join(", ");
            format!("{{{joined}}}")
        }
    }
}

#[derive(Debug, Clone)]
pub struct JobEntry {
    pub prompt: String,
    pub request: GenerationRequest,
    pub status: GenerationStatus,
    pub plan: CompositionPlan,
    pub artifact: Option<LocalArtifact>,
}

impl JobEntry {
    pub fn state_label(&self) -> &'static str {
        match self.status.state {
            JobState::Queued => "Queued",
            JobState::Running => "Running",
            JobState::Succeeded => "Done",
            JobState::Failed => "Failed",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusedPane {
    StatusBar,
    Conversation,
    Jobs,
    Status,
    Prompt,
}

#[derive(Debug, Clone)]
pub struct PlaybackState {
    pub job_id: String,
    pub path: PathBuf,
    pub duration: StdDuration,
    pub started_at: DateTime<Utc>,
    pub is_playing: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputMode {
    Normal,
    Insert,
}

#[derive(Debug)]
pub struct StatusBarState {
    pub progress: f32,
    pub message: String,
    pub last_update: DateTime<Utc>,
}

impl Default for StatusBarState {
    fn default() -> Self {
        Self {
            progress: 0.0,
            message: "Ready for the next idea…".to_string(),
            last_update: Utc::now(),
        }
    }
}

#[derive(Debug)]
pub struct AppState {
    pub input: String,
    pub chat: Vec<ChatEntry>,
    pub jobs: IndexMap<String, JobEntry>,
    pub focused_job: Option<String>,
    pub status_lines: Vec<String>,
    pub status_bar: StatusBarState,
    pub focused_pane: FocusedPane,
    pub chat_scroll: usize,
    pub status_scroll: usize,
    pub chat_following: bool,
    pub status_following: bool,
    pub input_mode: InputMode,
    pub backend_status: Vec<BackendHealthStatus>,
    app_config: AppConfig,
    generation_config: GenerationConfig,
    playback: Option<PlaybackState>,
    planner: CompositionPlanner,
}

impl AppState {
    pub fn new(config: AppConfig) -> Self {
        Self {
            input: String::new(),
            chat: Vec::new(),
            jobs: IndexMap::new(),
            focused_job: None,
            status_lines: Vec::new(),
            status_bar: StatusBarState::default(),
            focused_pane: FocusedPane::Prompt,
            chat_scroll: 0,
            status_scroll: 0,
            chat_following: true,
            status_following: true,
            input_mode: InputMode::Normal,
            backend_status: Vec::new(),
            playback: None,
            generation_config: GenerationConfig::from_app_config(&config),
            app_config: config,
            planner: CompositionPlanner::new(),
        }
    }

    pub fn handle_event(&mut self, event: AppEvent) {
        match event {
            AppEvent::Info(message) => self.push_status_line(message),
            AppEvent::Error(message) => {
                self.push_status_line(format!("Error: {message}"));
                self.append_chat(ChatRole::System, format!("{message}"));
            }
            AppEvent::JobQueued { status, prompt, request, plan } => {
                self.playback = None;
                self.jobs.insert(
                    status.job_id.clone(),
                    JobEntry {
                        prompt: prompt.clone(),
                        request: request.clone(),
                        status: status.clone(),
                        plan: plan.clone(),
                        artifact: None,
                    },
                );
                self.focused_job = Some(status.job_id.clone());
                let summary = format_request_summary(&request);
                self.push_status_line(format!("Job {} queued ({summary})", status.job_id));
                self.append_chat(
                    ChatRole::System,
                    format!("Composition plan for job {}: {}", status.job_id, plan_summary(&plan)),
                );
                self.append_chat(
                    ChatRole::Worker,
                    format!("Queued job {} ({summary})", status.job_id),
                );
                self.focused_pane = FocusedPane::StatusBar;
                self.status_scroll = 0;
                self.status_following = true;
                self.input_mode = InputMode::Normal;
            }
            AppEvent::JobUpdated { status } => {
                if let Some(job) = self.jobs.get_mut(&status.job_id) {
                    job.status = status.clone();
                    if status.state == JobState::Failed {
                        if let Some(message) = status.message.as_deref() {
                            self.push_status_line(format!(
                                "Job {} failed: {message}",
                                status.job_id
                            ));
                            self.append_chat(
                                ChatRole::Worker,
                                format!("Job {} failed: {message}", status.job_id),
                            );
                        } else {
                            self.push_status_line(format!("Job {} failed", status.job_id));
                            self.append_chat(
                                ChatRole::Worker,
                                format!("Job {} failed", status.job_id),
                            );
                        }
                    }
                }
            }
            AppEvent::JobCompleted { status, artifact } => {
                if let Some(job) = self.jobs.get_mut(&status.job_id) {
                    job.status = status.clone();
                    job.artifact = Some(artifact.clone());
                    if let Some(plan) = &artifact.descriptor.metadata.plan {
                        job.plan = plan.clone();
                    }
                    self.focused_job = Some(status.job_id.clone());
                    let (status_line, chat_line) = completion_messages(&status, &artifact);
                    self.push_status_line(status_line);
                    self.append_chat(ChatRole::Worker, chat_line);
                }
            }
            AppEvent::PlaybackStarted { job_id, path, duration } => {
                self.playback = Some(PlaybackState {
                    job_id,
                    path,
                    duration,
                    started_at: Utc::now(),
                    is_playing: true,
                });
                self.status_scroll = 0;
                self.status_following = true;
            }
            AppEvent::PlaybackProgress { is_playing } => {
                if let Some(playback) = &mut self.playback {
                    playback.is_playing = is_playing;
                }
                self.status_bar.last_update = Utc::now();
            }
            AppEvent::PlaybackStopped => {
                if let Some(playback) = &mut self.playback {
                    playback.is_playing = false;
                }
            }
            AppEvent::PollProgress { progress, message } => {
                self.status_bar.progress = progress.clamp(0.0, 1.0);
                self.status_bar.message = message;
                self.status_bar.last_update = Utc::now();
            }
            AppEvent::WorkerNudge { message } => {
                self.append_chat(ChatRole::Worker, message);
            }
        }
    }

    pub fn ingest_health_payload(&mut self, payload: &Value) {
        if let Some(map) = payload.get("backend_status").and_then(Value::as_object) {
            let mut statuses: Vec<BackendHealthStatus> = map
                .iter()
                .map(|(name, value)| BackendHealthStatus::from_json(name, value))
                .collect();
            statuses.sort_by(|a, b| a.name.cmp(&b.name));
            self.backend_status = statuses;
        }
    }

    pub fn append_chat(&mut self, role: ChatRole, content: String) {
        self.chat.push(ChatEntry { role, content, timestamp: Utc::now() });
        if self.chat.len() > MAX_CHAT_ENTRIES {
            let overflow = self.chat.len() - MAX_CHAT_ENTRIES;
            self.chat.drain(0..overflow);
        }
        if self.chat_following {
            self.chat_scroll = 0;
        }
    }

    pub fn push_status_line(&mut self, line: String) {
        self.status_lines.push(line);
        if self.status_lines.len() > MAX_STATUS_LINES {
            let overflow = self.status_lines.len() - MAX_STATUS_LINES;
            self.status_lines.drain(0..overflow);
        }
        if self.status_following {
            self.status_scroll = 0;
        }
    }

    pub fn jobs_iter(&self) -> Iter<'_, String, JobEntry> {
        self.jobs.iter()
    }

    pub fn config_summary(&self) -> String {
        self.generation_config.summary()
    }

    pub fn build_generation_payload(&self, prompt: &str) -> (GenerationRequest, CompositionPlan) {
        let plan = self.planner.build_plan(
            prompt,
            self.generation_config.duration_seconds,
            self.generation_config.seed,
        );
        let request = GenerationRequest {
            prompt: prompt.to_string(),
            seed: self.generation_config.seed,
            duration_seconds: self.generation_config.duration_seconds,
            model_id: self.generation_config.model_id.clone(),
            cfg_scale: self.generation_config.cfg_scale,
            scheduler: None,
            riffusion_num_inference_steps: None,
            riffusion_guidance_scale: None,
            riffusion_scheduler: None,
            musicgen_top_k: None,
            musicgen_top_p: None,
            musicgen_temperature: None,
            musicgen_cfg_coef: None,
            musicgen_two_step_cfg: None,
            output_sample_rate: None,
            output_bit_depth: None,
            output_format: None,
            plan: Some(plan.clone()),
        };
        (request, plan)
    }

    pub fn handle_command(&mut self, input: &str) -> Result<String, String> {
        let trimmed = input.trim_start_matches('/').trim();
        if trimmed.is_empty() {
            return Err("Empty command".to_string());
        }

        let mut parts = trimmed.split_whitespace();
        let command = parts.next().unwrap_or_default();
        let rest: Vec<&str> = parts.collect();

        match command {
            "duration" => {
                let value = rest
                    .first()
                    .ok_or_else(|| "Usage: /duration <seconds>".to_string())?
                    .parse::<u8>()
                    .map_err(|_| "Duration must be an integer (1-30)".to_string())?;
                if !(1..=30).contains(&value) {
                    return Err("Duration must be between 1 and 30 seconds".to_string());
                }
                self.generation_config.duration_seconds = value;
                Ok(format!("Duration set to {value}s"))
            }
            "model" => {
                let model = rest.join(" ").trim().to_string();
                if model.is_empty() {
                    return Err("Usage: /model <model_id>".to_string());
                }
                self.generation_config.model_id = model.clone();
                Ok(format!("Model set to {model}"))
            }
            "cfg" => {
                let arg = rest
                    .first()
                    .ok_or_else(|| "Usage: /cfg <scale|off>".to_string())?;
                if *arg == "off" {
                    self.generation_config.cfg_scale = None;
                    Ok("CFG scale set to auto".to_string())
                } else {
                    let value = arg
                        .parse::<f32>()
                        .map_err(|_| "CFG scale must be a number between 0-20".to_string())?;
                    if !(0.0..=20.0).contains(&value) {
                        return Err("CFG scale must be between 0 and 20".to_string());
                    }
                    self.generation_config.cfg_scale = Some(value);
                    Ok(format!("CFG scale set to {value:.1}"))
                }
            }
            "seed" => {
                let arg = rest
                    .first()
                    .ok_or_else(|| "Usage: /seed <value|off>".to_string())?;
                if *arg == "off" {
                    self.generation_config.seed = None;
                    Ok("Seed set to auto".to_string())
                } else {
                    let value = arg
                        .parse::<u64>()
                        .map_err(|_| "Seed must be a positive integer".to_string())?;
                    self.generation_config.seed = Some(value);
                    Ok(format!("Seed locked to {value}"))
                }
            }
            "reset" => {
                self.generation_config = GenerationConfig::from_app_config(&self.app_config);
                Ok("Generation settings reset to defaults".to_string())
            }
            "show" => Ok(format!("Current config: {}", self.config_summary())),
            "help" => Ok(
                "Commands: /duration <1-30>, /model <id>, /cfg <scale|off>, /seed <value|off>, /show, /reset"
                    .to_string(),
            ),
            other => Err(format!("Unknown command `{other}`")),
        }
    }

    pub fn select_next_job(&mut self) {
        if self.jobs.is_empty() {
            self.focused_job = None;
            return;
        }
        let next_index = self
            .focused_job
            .as_ref()
            .and_then(|id| self.jobs.get_index_of(id))
            .map(|idx| (idx + 1) % self.jobs.len())
            .unwrap_or(0);
        if let Some((job_id, _)) = self.jobs.get_index(next_index) {
            self.focused_job = Some(job_id.clone());
        }
    }

    pub fn select_previous_job(&mut self) {
        if self.jobs.is_empty() {
            self.focused_job = None;
            return;
        }
        let len = self.jobs.len();
        let prev_index = self
            .focused_job
            .as_ref()
            .and_then(|id| self.jobs.get_index_of(id))
            .map(|idx| if idx == 0 { len - 1 } else { idx - 1 })
            .unwrap_or(len - 1);
        if let Some((job_id, _)) = self.jobs.get_index(prev_index) {
            self.focused_job = Some(job_id.clone());
        }
    }

    pub fn selected_job(&self) -> Option<(&String, &JobEntry)> {
        self.focused_job
            .as_ref()
            .and_then(|id| self.jobs.get_full(id))
            .map(|(_, key, value)| (key, value))
    }

    pub fn focus_next(&mut self) {
        self.focused_pane = match self.focused_pane {
            FocusedPane::StatusBar => FocusedPane::Conversation,
            FocusedPane::Conversation => FocusedPane::Jobs,
            FocusedPane::Jobs => FocusedPane::Status,
            FocusedPane::Status => FocusedPane::Prompt,
            FocusedPane::Prompt => FocusedPane::StatusBar,
        };
    }

    pub fn focus_previous(&mut self) {
        self.focused_pane = match self.focused_pane {
            FocusedPane::StatusBar => FocusedPane::Prompt,
            FocusedPane::Conversation => FocusedPane::StatusBar,
            FocusedPane::Jobs => FocusedPane::Conversation,
            FocusedPane::Status => FocusedPane::Jobs,
            FocusedPane::Prompt => FocusedPane::Status,
        };
    }

    pub fn increment_chat_scroll(&mut self, delta: isize) {
        let current = self.chat_scroll as isize;
        let next = (current + delta).max(0);
        self.chat_scroll = next as usize;
        self.chat_following = self.chat_scroll == 0;
    }

    pub fn increment_status_scroll(&mut self, delta: isize) {
        let current = self.status_scroll as isize;
        let next = (current + delta).max(0);
        self.status_scroll = next as usize;
        self.status_following = self.status_scroll == 0;
    }

    pub fn playback_progress(
        &self,
    ) -> Option<(f32, StdDuration, StdDuration, bool, &String, &PathBuf)> {
        let playback = self.playback.as_ref()?;
        let now = Utc::now();
        let elapsed = now.signed_duration_since(playback.started_at);
        let elapsed =
            if elapsed < ChronoDuration::zero() { ChronoDuration::zero() } else { elapsed };
        let elapsed_std = StdDuration::from_millis(elapsed.num_milliseconds().max(0) as u64);
        let capped_elapsed = std::cmp::min(elapsed_std, playback.duration);
        let ratio = if playback.duration.as_millis() == 0 {
            0.0
        } else {
            (capped_elapsed.as_secs_f32() / playback.duration.as_secs_f32()).clamp(0.0, 1.0)
        };
        Some((
            ratio,
            capped_elapsed,
            playback.duration,
            playback.is_playing,
            &playback.job_id,
            &playback.path,
        ))
    }
}

#[derive(Debug, Clone)]
pub enum AppEvent {
    Info(String),
    Error(String),
    JobQueued {
        status: GenerationStatus,
        prompt: String,
        request: GenerationRequest,
        plan: CompositionPlan,
    },
    JobUpdated {
        status: GenerationStatus,
    },
    JobCompleted {
        status: GenerationStatus,
        artifact: LocalArtifact,
    },
    PlaybackStarted {
        job_id: String,
        path: PathBuf,
        duration: StdDuration,
    },
    PlaybackProgress {
        is_playing: bool,
    },
    PlaybackStopped,
    PollProgress {
        progress: f32,
        message: String,
    },
    WorkerNudge {
        message: String,
    },
}

#[derive(Debug, Clone)]
pub enum AppCommand {
    SubmitPrompt { prompt: String, request: GenerationRequest, plan: CompositionPlan },
    PlayJob { job_id: String },
    StopPlayback,
}

pub fn format_request_summary(request: &GenerationRequest) -> String {
    let mut parts: Vec<String> = Vec::new();
    parts.push(request.model_id.clone());
    parts.push(format!("{}s", request.duration_seconds));

    parts.push(
        request
            .cfg_scale
            .map(|value| format!("cfg {:.1}", value))
            .unwrap_or_else(|| "cfg auto".to_string()),
    );

    if let Some(seed) = request.seed {
        parts.push(format!("seed {}", seed));
    }

    if let Some(plan) = &request.plan {
        parts.push(format!("plan {} sections", plan.sections.len()));
    }

    parts.join(", ")
}

fn completion_messages(status: &GenerationStatus, artifact: &LocalArtifact) -> (String, String) {
    let metadata = &artifact.descriptor.metadata;
    let extras = metadata.extras.as_object();

    let backend =
        extras.and_then(|map| map.get("backend")).and_then(Value::as_str).unwrap_or("unknown");
    let placeholder =
        extras.and_then(|map| map.get("placeholder")).and_then(Value::as_bool).unwrap_or(false);
    let guidance = extras.and_then(|map| map.get("guidance_scale")).and_then(Value::as_f64);
    let sample_rate = extras.and_then(|map| map.get("sample_rate")).and_then(Value::as_u64);
    let placeholder_reason =
        extras.and_then(|map| map.get("placeholder_reason")).and_then(Value::as_str);
    let prompt_hash = extras.and_then(|map| map.get("prompt_hash")).and_then(Value::as_str);

    let mut parts: Vec<String> = Vec::new();

    if placeholder {
        let reason = placeholder_reason.unwrap_or("unknown");
        parts.push(format!("placeholder via {backend}"));
        parts.push(format!("reason: {reason}"));
    } else {
        parts.push(backend.to_string());
        if let Some(sr) = sample_rate {
            parts.push(format!("{sr} Hz"));
        }
    }

    parts.push(format!("{}s", metadata.duration_seconds));

    if let Some(cfg) = guidance {
        parts.push(format!("cfg {:.1}", cfg));
    }

    if let Some(hash) = prompt_hash {
        parts.push(format!("hash {hash}"));
    }

    if let Some(seed) = extras.and_then(|map| map.get("seed")).and_then(Value::as_u64) {
        parts.push(format!("seed {}", seed));
    }

    if let Some(plan) = &metadata.plan {
        parts.push(format!("{} sections", plan.sections.len()));
    }

    let detail = parts.join(", ");
    let status_line = format!("Job {} completed ({detail})", status.job_id);
    let chat_line =
        format!("Job {} completed ({detail}) → {}", status.job_id, artifact.local_path.display());
    (status_line, chat_line)
}

fn plan_summary(plan: &CompositionPlan) -> String {
    let mut parts: Vec<String> = Vec::new();
    parts.push(format!("{} sections · {} BPM · {}", plan.sections.len(), plan.tempo_bpm, plan.key));
    let flow = plan
        .sections
        .iter()
        .map(|section| section.label.clone())
        .collect::<Vec<String>>()
        .join(" → ");
    parts.push(format!("flow: {flow}"));
    if plan.sections.iter().any(|section| section.motif_directive.is_some()) {
        let motif_arc = plan
            .sections
            .iter()
            .map(|section| {
                section.motif_directive.clone().unwrap_or_else(|| format!("{:?}", section.role))
            })
            .collect::<Vec<String>>()
            .join(" → ");
        parts.push(format!("motif arc: {motif_arc}"));
    }
    parts.join(" | ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AppConfig;
    use crate::types::GenerationMetadata;
    use chrono::Utc;
    use serde_json::json;

    #[test]
    fn completion_messages_real_pipeline() {
        let status = GenerationStatus {
            job_id: "job123".into(),
            state: JobState::Succeeded,
            progress: 1.0,
            message: None,
            updated_at: Utc::now(),
        };

        let metadata = GenerationMetadata {
            prompt: "dreamy pianos".into(),
            seed: Some(42),
            model_id: "riffusion-v1".into(),
            duration_seconds: 8,
            extras: json!({
                "backend": "riffusion",
                "placeholder": false,
                "guidance_scale": 7.0,
                "sample_rate": 44100,
                "prompt_hash": "abc123ef"
            }),
            plan: None,
        };

        let artifact = LocalArtifact {
            descriptor: GenerationArtifact {
                job_id: "job123".into(),
                artifact_path: "/tmp/job123.wav".into(),
                metadata,
            },
            local_path: PathBuf::from("/tmp/job123/job123.wav"),
        };

        let (status_line, chat_line) = completion_messages(&status, &artifact);
        assert!(status_line.contains("riffusion"));
        assert!(status_line.contains("8s"));
        assert!(status_line.contains("44100 Hz"));
        assert!(status_line.contains("hash abc123ef"));
        assert!(chat_line.contains("→ /tmp/job123/job123.wav"));
    }

    #[test]
    fn completion_messages_placeholder_pipeline() {
        let status = GenerationStatus {
            job_id: "job456".into(),
            state: JobState::Succeeded,
            progress: 1.0,
            message: None,
            updated_at: Utc::now(),
        };

        let metadata = GenerationMetadata {
            prompt: "stormy guitars".into(),
            seed: None,
            model_id: "riffusion-v1".into(),
            duration_seconds: 6,
            extras: json!({
                "backend": "riffusion",
                "placeholder": true,
                "placeholder_reason": "pipeline_unavailable",
                "prompt_hash": "deadbeef"
            }),
            plan: None,
        };

        let artifact = LocalArtifact {
            descriptor: GenerationArtifact {
                job_id: "job456".into(),
                artifact_path: "/tmp/job456.wav".into(),
                metadata,
            },
            local_path: PathBuf::from("/tmp/job456/job456.wav"),
        };

        let (status_line, chat_line) = completion_messages(&status, &artifact);
        assert!(status_line.contains("placeholder via riffusion"));
        assert!(status_line.contains("reason: pipeline_unavailable"));
        assert!(!status_line.contains("Hz"));
        assert!(chat_line.contains("hash deadbeef"));
    }

    #[test]
    fn completion_messages_includes_seed_when_available() {
        let status = GenerationStatus {
            job_id: "job789".into(),
            state: JobState::Succeeded,
            progress: 1.0,
            message: None,
            updated_at: Utc::now(),
        };

        let metadata = GenerationMetadata {
            prompt: "seeded run".into(),
            seed: Some(7),
            model_id: "riffusion-v1".into(),
            duration_seconds: 5,
            extras: json!({
                "backend": "riffusion",
                "placeholder": false,
                "guidance_scale": 4.0,
                "sample_rate": 44100,
                "prompt_hash": "feedcafe",
                "seed": 7
            }),
            plan: None,
        };

        let artifact = LocalArtifact {
            descriptor: GenerationArtifact {
                job_id: "job789".into(),
                artifact_path: "/tmp/job789.wav".into(),
                metadata,
            },
            local_path: PathBuf::from("/tmp/job789/job789.wav"),
        };

        let (status_line, chat_line) = completion_messages(&status, &artifact);
        assert!(status_line.contains("seed 7"));
        assert!(chat_line.contains("seed 7"));
    }

    #[test]
    fn request_summary_formats_values() {
        let request = GenerationRequest {
            prompt: "test".into(),
            seed: Some(99),
            duration_seconds: 12,
            model_id: "riffusion-v1".into(),
            cfg_scale: Some(5.5),
            scheduler: None,
            riffusion_num_inference_steps: None,
            riffusion_guidance_scale: None,
            riffusion_scheduler: None,
            musicgen_top_k: None,
            musicgen_top_p: None,
            musicgen_temperature: None,
            musicgen_cfg_coef: None,
            musicgen_two_step_cfg: None,
            output_sample_rate: None,
            output_bit_depth: None,
            output_format: None,
            plan: None,
        };
        let summary = format_request_summary(&request);
        assert!(summary.contains("riffusion-v1"));
        assert!(summary.contains("12s"));
        assert!(summary.contains("cfg 5.5"));
        assert!(summary.contains("seed 99"));
    }

    #[test]
    fn handle_command_updates_duration() {
        let mut state = AppState::new(AppConfig::default());
        let result = state.handle_command("/duration 12");
        assert!(result.is_ok());
        assert_eq!(state.generation_config.duration_seconds, 12);
    }

    #[test]
    fn handle_command_rejects_invalid_cfg() {
        let mut state = AppState::new(AppConfig::default());
        let result = state.handle_command("/cfg 80");
        assert!(result.is_err());
    }

    #[test]
    fn build_request_reflects_generation_config() {
        let mut state = AppState::new(AppConfig::default());
        let _ = state.handle_command("/duration 10");
        let _ = state.handle_command("/cfg 6.5");
        let _ = state.handle_command("/seed 77");
        let (request, plan) = state.build_generation_payload("lively strings");
        assert_eq!(request.duration_seconds, 10);
        assert_eq!(request.cfg_scale, Some(6.5));
        assert_eq!(request.seed, Some(77));
        assert_eq!(request.model_id, "musicgen-stereo-medium");
        assert!(request.plan.is_some());
        assert_eq!(plan.sections.len(), request.plan.as_ref().unwrap().sections.len());
    }

    #[test]
    fn ingest_health_payload_populates_backend_status() {
        let mut state = AppState::new(AppConfig::default());
        let payload = json!({
            "backend_status": {
                "musicgen": {
                    "ready": true,
                    "device": "cpu",
                    "details": {"sample_rate": 32000},
                    "updated_at": "2024-01-01T00:00:00Z"
                }
            }
        });

        state.ingest_health_payload(&payload);
        assert_eq!(state.backend_status.len(), 1);
        let status = &state.backend_status[0];
        assert_eq!(status.name, "musicgen");
        assert!(status.ready);
        assert_eq!(status.device.as_deref(), Some("cpu"));
        assert!(status.details.iter().any(|(key, value)| key == "sample_rate" && value == "32000"));
    }
}
