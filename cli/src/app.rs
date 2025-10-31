use crate::{
    config::AppConfig,
    planner::CompositionPlanner,
    session::{clip_layer_label, ClipSlotStatus, SessionSnapshot, SessionView},
    session_store,
    types::{
        ClipLayer, CompositionPlan, CompositionSection, GenerationArtifact, GenerationMode,
        GenerationRequest, GenerationStatus, JobState, SectionEnergy, SectionOrchestration,
        SectionRole, SessionCreateRequest, SessionSummary, ThemeDescriptor,
    },
};
use anyhow::Result;
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use indexmap::{map::Iter, IndexMap};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::time::{Duration as StdDuration, Instant};

const MAX_STATUS_LINES: usize = 8;
const MIN_DURATION_SECONDS: u8 = 90;
const MAX_DURATION_SECONDS: u8 = 180;
const MOTIF_PRESET_DURATION_SECONDS: u8 = 16;

#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub model_id: String,
    pub duration_seconds: u8,
    pub cfg_scale: Option<f32>,
    pub seed: Option<u64>,
    pub mode: GenerationMode,
}

impl GenerationConfig {
    fn from_app_config(config: &AppConfig) -> Self {
        Self {
            model_id: config.default_model_id().to_string(),
            duration_seconds: config.default_duration_seconds(),
            cfg_scale: None,
            seed: None,
            mode: GenerationMode::FullTrack,
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
        let mode_text = match self.mode {
            GenerationMode::FullTrack => "Full".to_string(),
            GenerationMode::Motif => "Motif".to_string(),
            GenerationMode::Clip => "Clip".to_string(),
        };
        format!(
            "{} · {}s · {} · {} · {}",
            self.model_id, self.duration_seconds, cfg_text, seed_text, mode_text
        )
    }
}

#[derive(Debug, Default, Clone)]
pub struct GenerationOverrides {
    pub model_id: Option<String>,
    pub duration_seconds: Option<u8>,
    pub mode: Option<GenerationMode>,
}

#[derive(Debug, Clone)]
pub struct InlineCommand {
    pub overrides: GenerationOverrides,
    pub usage_hint: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionCliCommand {
    Start,
    Status,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplashSelection {
    Restore,
    StartNew,
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
    Motif,
    SessionView,
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

#[derive(Debug, Clone)]
pub struct ClipLaunchContext {
    pub layer: ClipLayer,
    pub scene_index: usize,
    pub path: PathBuf,
    pub tempo_bpm: u16,
    pub beats_per_bar: f32,
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
pub struct SplashAnimationState {
    started_at: Instant,
}

impl SplashAnimationState {
    pub fn new() -> Self {
        Self { started_at: Instant::now() }
    }

    pub fn elapsed(&self) -> StdDuration {
        self.started_at.elapsed()
    }

    #[cfg(test)]
    pub fn with_started_at(started_at: Instant) -> Self {
        Self { started_at }
    }
}

impl Default for SplashAnimationState {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct AppState {
    pub input: String,
    pub jobs: IndexMap<String, JobEntry>,
    pub focused_job: Option<String>,
    pub status_lines: Vec<String>,
    pub status_bar: StatusBarState,
    pub focused_pane: FocusedPane,
    pub status_scroll: usize,
    pub status_following: bool,
    pub input_mode: InputMode,
    pub backend_status: Vec<BackendHealthStatus>,
    pub last_submission_model_id: Option<String>,
    pub session_view: SessionView,
    pub session: Option<SessionSummary>,
    restored_session_id: Option<String>,
    app_config: AppConfig,
    generation_config: GenerationConfig,
    playback: Option<PlaybackState>,
    planner: CompositionPlanner,
    musicgen_default_model_id: Option<String>,
    showing_splash: bool,
    splash_selection: SplashSelection,
    pending_snapshot: Option<SessionSnapshot>,
    splash_animation: SplashAnimationState,
}

impl AppState {
    pub fn new(config: AppConfig) -> Self {
        let musicgen_default_model_id = config.default_model_id().to_string();
        let snapshot_result = session_store::load_snapshot();
        let pending_snapshot = snapshot_result.as_ref().ok().and_then(|opt| opt.clone());
        let mut state = Self {
            input: String::new(),
            jobs: IndexMap::new(),
            focused_job: None,
            status_lines: Vec::new(),
            status_bar: StatusBarState::default(),
            focused_pane: FocusedPane::Prompt,
            status_scroll: 0,
            status_following: true,
            input_mode: InputMode::Normal,
            backend_status: Vec::new(),
            last_submission_model_id: None,
            session_view: SessionView::new(),
            session: None,
            restored_session_id: None,
            playback: None,
            generation_config: GenerationConfig::from_app_config(&config),
            app_config: config,
            planner: CompositionPlanner::new(),
            musicgen_default_model_id: Some(musicgen_default_model_id),
            showing_splash: true,
            splash_selection: SplashSelection::StartNew,
            pending_snapshot: pending_snapshot.clone(),
            splash_animation: SplashAnimationState::default(),
        };

        if pending_snapshot.is_some() {
            state.splash_selection = SplashSelection::Restore;
        }

        if let Err(err) = snapshot_result {
            state.status_lines.push(format!("Failed to load session snapshot: {err}"));
        }

        state
    }

    pub fn showing_splash(&self) -> bool {
        self.showing_splash
    }

    pub fn has_pending_snapshot(&self) -> bool {
        self.pending_snapshot.is_some()
    }

    pub fn splash_selection(&self) -> SplashSelection {
        self.splash_selection
    }

    pub fn splash_elapsed(&self) -> StdDuration {
        self.splash_animation.elapsed()
    }

    pub fn select_next_splash_option(&mut self) {
        self.splash_selection = match (self.splash_selection, self.has_pending_snapshot()) {
            (SplashSelection::Restore, _) => SplashSelection::StartNew,
            (SplashSelection::StartNew, true) => SplashSelection::Restore,
            (SplashSelection::StartNew, false) => SplashSelection::StartNew,
        };
    }

    pub fn select_previous_splash_option(&mut self) {
        self.splash_selection = match (self.splash_selection, self.has_pending_snapshot()) {
            (SplashSelection::StartNew, true) => SplashSelection::Restore,
            (SplashSelection::StartNew, false) => SplashSelection::StartNew,
            (SplashSelection::Restore, _) => SplashSelection::StartNew,
        };
    }

    pub fn apply_pending_snapshot(&mut self) -> bool {
        let Some(snapshot) = self.pending_snapshot.clone() else {
            return false;
        };

        self.session_view = SessionView::new();
        self.session_view.restore(&snapshot);
        self.session = None;
        self.restored_session_id = snapshot.session_id.clone();
        self.pending_snapshot = None;
        self.showing_splash = false;
        if let Some(session_id) = snapshot.session_id {
            let label = Self::short_session_id(&session_id);
            self.push_status_line(format!("Restored session layout for {label} (offline)"));
        } else {
            self.push_status_line("Restored session layout (offline)".to_string());
        }
        true
    }

    #[cfg(test)]
    pub fn disable_splash(&mut self) {
        self.showing_splash = false;
    }

    pub fn parse_session_command(&self, command_line: &str) -> Result<SessionCliCommand, String> {
        let trimmed = command_line.trim_start_matches('/').trim();
        let mut parts = trimmed.split_whitespace();
        let Some(keyword) = parts.next() else {
            return Err("Usage: /session <start|status>".to_string());
        };
        if keyword != "session" {
            return Err("not a session command".to_string());
        }
        let Some(action) = parts.next() else {
            return Err("Usage: /session <start|status>".to_string());
        };
        match action {
            "start" => Ok(SessionCliCommand::Start),
            "status" => {
                if self.session.is_none() {
                    Err("No active session. Use /session start first.".to_string())
                } else {
                    Ok(SessionCliCommand::Status)
                }
            }
            other => Err(format!("Unknown session command `{other}`")),
        }
    }

    pub fn handle_event(&mut self, event: AppEvent) {
        match event {
            AppEvent::Info(message) => self.push_status_line(message),
            AppEvent::Error(message) => {
                self.push_status_line(format!("Error: {message}"));
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
                self.last_submission_model_id = Some(request.model_id.clone());
                self.focused_job = Some(status.job_id.clone());
                let summary = format_request_summary(&request);
                self.push_status_line(format!("Job {} queued ({summary})", status.job_id));
                self.push_status_line(plan_summary(&plan));
                self.push_status_line(format!("Queued job {} ({summary})", status.job_id));
                self.focused_pane = FocusedPane::StatusBar;
                self.status_scroll = 0;
                self.status_following = true;
                self.input_mode = InputMode::Normal;
                if matches!(request.mode, Some(GenerationMode::Clip)) {
                    if let Some(layer) = request.clip_layer {
                        let scene_index = request
                            .clip_scene_index
                            .map(|value| value as usize)
                            .unwrap_or(self.session_view.focused().1);
                        self.session_view.mark_pending(
                            layer,
                            scene_index,
                            status.job_id.clone(),
                            prompt.clone(),
                        );
                    }
                }
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
                        } else {
                            self.push_status_line(format!("Job {} failed", status.job_id));
                        }
                        if self.session_view.pending_scene_for_job(&status.job_id).is_some() {
                            self.session_view.mark_failed(&status.job_id);
                            self.session_view.clear_job(&status.job_id);
                        }
                    } else if status.state == JobState::Running {
                        if self.session_view.pending_scene_for_job(&status.job_id).is_some() {
                            self.session_view
                                .mark_status(&status.job_id, ClipSlotStatus::Rendering);
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
                    let status_line = completion_message(&status, &artifact);
                    self.push_status_line(status_line);
                }
                if self.session_view.pending_scene_for_job(&status.job_id).is_some() {
                    let duration = artifact.descriptor.metadata.duration_seconds as f32;
                    let bars = artifact
                        .descriptor
                        .metadata
                        .plan
                        .as_ref()
                        .and_then(|plan| plan.sections.first())
                        .map(|section| section.bars);
                    self.session_view.mark_ready(
                        &status.job_id,
                        &artifact.local_path,
                        duration,
                        bars.map(|b| b as u8),
                    );
                    self.session_view.clear_job(&status.job_id);
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
                self.push_status_line(message);
            }
            AppEvent::SessionStarted { summary } => {
                let label = Self::format_session_label(&summary);
                let tempo = summary
                    .tempo_bpm
                    .map(|bpm| format!("{bpm} BPM"))
                    .unwrap_or_else(|| "tempo pending".to_string());
                self.session = Some(summary.clone());
                self.session_view = SessionView::new();
                self.session_view.apply_summary(&summary);
                self.restored_session_id = Some(summary.session_id.clone());
                self.push_status_line(format!("Session {label} created ({tempo})"));
            }
            AppEvent::SessionUpdated { summary } => {
                let label = Self::format_session_label(&summary);
                if self.session.as_ref().map(|s| s.session_id.clone())
                    != Some(summary.session_id.clone())
                {
                    self.session_view = SessionView::new();
                }
                self.session_view.apply_summary(&summary);
                self.session = Some(summary.clone());
                self.restored_session_id = Some(summary.session_id.clone());
                self.push_status_line(format!(
                    "Session {label} updated ({} clips)",
                    summary.clip_count
                ));
            }
            AppEvent::ClipPlaying { layer, scene_index } => {
                self.session_view.set_playing(layer, scene_index);
            }
            AppEvent::ClipStopped { layer } => {
                self.session_view.clear_playing(layer);
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

        if let Some(default_musicgen) =
            payload.get("musicgen_default_model_id").and_then(Value::as_str)
        {
            self.musicgen_default_model_id = Some(default_musicgen.trim().to_string());
        }

        if let Some(default_model) = payload.get("default_model_id").and_then(Value::as_str) {
            if self.generation_config.model_id == self.app_config.default_model_id() {
                self.generation_config.model_id = default_model.trim().to_string();
            }
            if self.musicgen_default_model_id.is_none() {
                self.musicgen_default_model_id = Some(default_model.trim().to_string());
            }
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
        let base = self.generation_config.summary();
        if let Some(session) = &self.session {
            let short_id = Self::format_session_label(session);
            let tempo = session
                .tempo_bpm
                .map(|bpm| format!("{bpm} BPM"))
                .unwrap_or_else(|| "tempo pending".to_string());
            format!("{base} · Session {short_id} ({tempo})")
        } else {
            base
        }
    }

    pub(crate) fn format_session_label(summary: &SessionSummary) -> String {
        Self::short_session_id(&summary.session_id)
    }

    fn short_session_id(session_id: &str) -> String {
        let trimmed = session_id.trim_start_matches("session-");
        let short: String = trimmed.chars().take(8).collect();
        if short.is_empty() {
            session_id.to_string()
        } else {
            short
        }
    }

    pub fn current_session_id(&self) -> Option<String> {
        self.session.as_ref().map(|summary| summary.session_id.clone())
    }

    pub fn motif_ready(&self) -> bool {
        self.session.as_ref().map(|summary| summary.seed_plan.is_some()).unwrap_or(false)
    }

    pub fn active_clip_target(&self) -> Option<(ClipLayer, usize)> {
        self.session_view.active_target()
    }

    pub fn toggle_active_clip_target(&mut self) -> bool {
        let (layer, scene_index) = self.session_view.focused();
        self.session_view.toggle_active_target(layer, scene_index)
    }

    pub fn prompt_panel_title(&self) -> String {
        if let Some(session) = &self.session {
            let label = Self::format_session_label(session);
            let (layer, scene_index) =
                self.active_clip_target().unwrap_or_else(|| self.session_view.focused());
            let scene_name = self.session_view.scene_name(scene_index);
            let layer_label = clip_layer_label(layer);
            let target_text = if self.active_clip_target().is_some() {
                format!("{scene_name} · {layer_label} (active)")
            } else {
                format!("{scene_name} · {layer_label}")
            };
            format!(
                "Prompt · Session {label} · {target_text} · Model {}",
                self.generation_config.model_id
            )
        } else {
            format!("Prompt · Model {}", self.generation_config.model_id)
        }
    }

    pub fn last_submission_model(&self) -> Option<&str> {
        self.last_submission_model_id.as_deref()
    }

    pub fn default_model_for(&self, alias: &str) -> Option<String> {
        if alias == "musicgen" {
            return Some(Self::sanitized_model(&self.musicgen_default_model_id, || {
                self.app_config.default_model_id().to_string()
            }));
        }
        None
    }

    pub fn inline_command_for(&self, alias: &str) -> Option<InlineCommand> {
        match alias {
            "musicgen" => {
                let model = Self::sanitized_model(&self.musicgen_default_model_id, || {
                    self.app_config.default_model_id().to_string()
                });
                Some(InlineCommand {
                    overrides: GenerationOverrides {
                        model_id: Some(model),
                        ..GenerationOverrides::default()
                    },
                    usage_hint: "Usage: /musicgen <prompt>",
                })
            }
            "small" => Some(InlineCommand {
                overrides: GenerationOverrides {
                    model_id: Some("musicgen-stereo-small".to_string()),
                    ..GenerationOverrides::default()
                },
                usage_hint: "Usage: /small <prompt>",
            }),
            "medium" => Some(InlineCommand {
                overrides: GenerationOverrides {
                    model_id: Some("musicgen-stereo-medium".to_string()),
                    ..GenerationOverrides::default()
                },
                usage_hint: "Usage: /medium <prompt>",
            }),
            "large" => Some(InlineCommand {
                overrides: GenerationOverrides {
                    model_id: Some("musicgen-stereo-large".to_string()),
                    ..GenerationOverrides::default()
                },
                usage_hint: "Usage: /large <prompt>",
            }),
            "motif" => Some(InlineCommand {
                overrides: GenerationOverrides {
                    model_id: Some("musicgen-stereo-medium".to_string()),
                    duration_seconds: Some(MOTIF_PRESET_DURATION_SECONDS),
                    mode: Some(GenerationMode::Motif),
                },
                usage_hint: "Usage: /motif <prompt>",
            }),
            _ => None,
        }
    }

    fn sanitized_model(candidate: &Option<String>, fallback: impl FnOnce() -> String) -> String {
        candidate
            .as_ref()
            .and_then(|value| {
                let trimmed = value.trim();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed.to_string())
                }
            })
            .unwrap_or_else(fallback)
    }

    #[allow(dead_code)]
    pub fn build_generation_payload(&self, prompt: &str) -> (GenerationRequest, CompositionPlan) {
        self.build_generation_payload_with_overrides(prompt, None)
    }

    pub fn build_generation_payload_with_overrides(
        &self,
        prompt: &str,
        overrides: Option<&GenerationOverrides>,
    ) -> (GenerationRequest, CompositionPlan) {
        let duration_seconds = overrides
            .and_then(|opts| opts.duration_seconds)
            .unwrap_or(self.generation_config.duration_seconds);
        let model_id = overrides
            .and_then(|opts| opts.model_id.as_ref())
            .cloned()
            .unwrap_or_else(|| self.generation_config.model_id.clone());
        let mode = overrides.and_then(|opts| opts.mode).unwrap_or(self.generation_config.mode);

        let plan =
            self.planner.build_plan(prompt, duration_seconds, self.generation_config.seed, mode);
        let request = GenerationRequest {
            prompt: prompt.to_string(),
            seed: self.generation_config.seed,
            duration_seconds,
            model_id,
            session_id: self.current_session_id(),
            clip_layer: None,
            clip_scene_index: None,
            clip_bars: None,
            mode: Some(mode),
            cfg_scale: self.generation_config.cfg_scale,
            scheduler: None,
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

    pub fn build_clip_payload_for(
        &self,
        layer: ClipLayer,
        scene_index: usize,
        prompt_override: Option<&str>,
    ) -> Result<(GenerationRequest, CompositionPlan, String), String> {
        let session = self
            .session
            .as_ref()
            .ok_or_else(|| "No active session. Use /session start first.".to_string())?;
        if session.seed_plan.is_none() {
            return Err("Motif not ready yet. Generate a motif before queuing clips.".to_string());
        }
        if !self.session_view.scenes().iter().any(|scene| scene.index == scene_index) {
            return Err("Select a valid scene before submitting a clip.".to_string());
        }

        let bars: u8 = 4;
        let tempo = session.tempo_bpm.unwrap_or(100);
        let time_signature = session.time_signature.clone().unwrap_or_else(|| "4/4".to_string());
        let beats = beats_per_bar(&time_signature);
        let seconds_per_bar = (60.0_f32 / tempo.max(1) as f32) * beats.max(1.0);
        let total_seconds = (bars as f32 * seconds_per_bar).max(1.0);

        let explicit_prompt = prompt_override.and_then(|value| {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        });

        let clip_prompt = explicit_prompt.unwrap_or_else(|| {
            session
                .seed_prompt
                .clone()
                .or_else(|| session.theme.as_ref().map(|theme| theme.motif.clone()))
                .unwrap_or_else(|| "session clip".to_string())
        });

        let descriptor = session.theme.clone().unwrap_or_else(|| ThemeDescriptor {
            motif: clip_prompt.clone(),
            instrumentation: Vec::new(),
            rhythm: "steady pulse".to_string(),
            dynamic_curve: Vec::new(),
            texture: None,
        });

        let section_prompt = clip_prompt_text(&clip_prompt, &descriptor, layer);
        let display_label = clip_layer_label(layer);
        let layer_key = display_label.to_ascii_lowercase();

        let section = CompositionSection {
            section_id: "c00".to_string(),
            role: SectionRole::Development,
            label: format!("{} clip", display_label),
            prompt: section_prompt,
            bars,
            target_seconds: total_seconds,
            energy: clip_energy(layer),
            model_id: None,
            seed_offset: Some(0),
            transition: None,
            motif_directive: Some(clip_directive(layer)),
            variation_axes: vec!["layer".to_string(), layer_key.clone()],
            cadence_hint: None,
            orchestration: clip_orchestration(layer, session.theme.as_ref()),
        };

        let key = session.key.clone().unwrap_or_else(|| "C major".to_string());
        let plan = CompositionPlan {
            version: "v4".to_string(),
            tempo_bpm: tempo,
            time_signature: time_signature.clone(),
            key,
            total_bars: bars as u16,
            total_duration_seconds: total_seconds,
            theme: Some(descriptor.clone()),
            sections: vec![section],
        };

        let duration_seconds = total_seconds.round().clamp(1.0, MAX_DURATION_SECONDS as f32) as u8;
        let request = GenerationRequest {
            prompt: clip_prompt.clone(),
            seed: self.generation_config.seed,
            duration_seconds,
            model_id: self.generation_config.model_id.clone(),
            session_id: self.current_session_id(),
            clip_layer: Some(layer),
            clip_scene_index: Some(scene_index as u16),
            clip_bars: Some(bars),
            mode: Some(GenerationMode::Clip),
            cfg_scale: self.generation_config.cfg_scale,
            scheduler: None,
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

        Ok((request, plan, clip_prompt))
    }

    pub fn build_motif_payload(
        &self,
        prompt: &str,
    ) -> Result<(GenerationRequest, CompositionPlan), String> {
        if prompt.trim().is_empty() {
            return Err("Motif prompt cannot be empty.".to_string());
        }
        if self.session.is_none() {
            return Err("No active session. Use /session start first.".to_string());
        }
        let overrides = GenerationOverrides {
            duration_seconds: Some(MOTIF_PRESET_DURATION_SECONDS),
            mode: Some(GenerationMode::Motif),
            ..GenerationOverrides::default()
        };
        Ok(self.build_generation_payload_with_overrides(prompt, Some(&overrides)))
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
                    .map_err(|_| "Duration must be an integer (90-180)".to_string())?;
                if value < MIN_DURATION_SECONDS || value > MAX_DURATION_SECONDS {
                    return Err(format!(
                        "Duration must be between {MIN_DURATION_SECONDS} and {MAX_DURATION_SECONDS} seconds"
                    ));
                }
                self.generation_config.duration_seconds = value;
                Ok(format!("Duration set to {value}s"))
            }
            "musicgen" => {
                let model = self
                    .default_model_for("musicgen")
                    .unwrap_or_else(|| self.generation_config.model_id.clone());
                self.generation_config.model_id = model.clone();
                Ok(format!("Model set to {model}"))
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
                format!(
                    "Commands: /duration <{MIN_DURATION_SECONDS}-{MAX_DURATION_SECONDS}>, /model <id>, /musicgen, /cfg <scale|off>, /seed <value|off>, /clip <layer> [prompt], /show, /reset · inline prompts: /motif <prompt>, /small|/medium|/large <prompt>"
                ),
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

    pub fn clip_launch_context(&self) -> Option<ClipLaunchContext> {
        let session = self.session.as_ref()?;
        let tempo_bpm = session.tempo_bpm? as u16;
        let time_signature = session.time_signature.clone().unwrap_or_else(|| "4/4".to_string());
        let beats = beats_per_bar(&time_signature);
        let (layer, scene_index) = self.session_view.focused();
        let slot = self.session_view.slot(layer, scene_index);
        if !matches!(slot.status, ClipSlotStatus::Ready) {
            return None;
        }
        let artifact = slot.artifact.as_ref()?;
        Some(ClipLaunchContext {
            layer,
            scene_index,
            path: artifact.clone(),
            tempo_bpm,
            beats_per_bar: beats,
        })
    }

    pub fn focused_clip_layer(&self) -> ClipLayer {
        self.session_view.focused().0
    }

    pub fn reset_session_state(&mut self) {
        self.session = None;
        self.restored_session_id = None;
        self.session_view = SessionView::new();
        self.jobs.clear();
        self.focused_job = None;
        self.playback = None;
        self.pending_snapshot = None;
        self.showing_splash = false;
        self.splash_selection = SplashSelection::StartNew;
    }

    pub fn save_session_snapshot(&self) -> Result<()> {
        if self.showing_splash {
            if let Some(snapshot) = &self.pending_snapshot {
                return session_store::save_snapshot(snapshot);
            }
            return Ok(());
        }
        let session_id = self
            .session
            .as_ref()
            .map(|summary| summary.session_id.clone())
            .or_else(|| self.restored_session_id.clone());
        let snapshot = self.session_view.snapshot(session_id);
        session_store::save_snapshot(&snapshot)
    }

    pub fn focus_next(&mut self) {
        self.focused_pane = match self.focused_pane {
            FocusedPane::StatusBar => FocusedPane::Motif,
            FocusedPane::Motif => FocusedPane::SessionView,
            FocusedPane::SessionView => FocusedPane::Jobs,
            FocusedPane::Jobs => FocusedPane::Status,
            FocusedPane::Status => FocusedPane::Prompt,
            FocusedPane::Prompt => FocusedPane::StatusBar,
        };
    }

    pub fn focus_previous(&mut self) {
        self.focused_pane = match self.focused_pane {
            FocusedPane::StatusBar => FocusedPane::Prompt,
            FocusedPane::Motif => FocusedPane::StatusBar,
            FocusedPane::SessionView => FocusedPane::Motif,
            FocusedPane::Jobs => FocusedPane::SessionView,
            FocusedPane::Status => FocusedPane::Jobs,
            FocusedPane::Prompt => FocusedPane::Status,
        };
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
    SessionStarted {
        summary: SessionSummary,
    },
    SessionUpdated {
        summary: SessionSummary,
    },
    ClipPlaying {
        layer: ClipLayer,
        scene_index: usize,
    },
    ClipStopped {
        layer: ClipLayer,
    },
}

#[derive(Debug, Clone)]
pub enum AppCommand {
    SubmitPrompt {
        prompt: String,
        request: GenerationRequest,
        plan: CompositionPlan,
    },
    SubmitClip {
        prompt: String,
        layer: ClipLayer,
        request: GenerationRequest,
        plan: CompositionPlan,
    },
    PlayJob {
        job_id: String,
    },
    StopPlayback,
    LaunchClip {
        layer: ClipLayer,
        scene_index: usize,
        path: PathBuf,
        tempo_bpm: u16,
        beats_per_bar: f32,
    },
    StopClip {
        layer: ClipLayer,
    },
    StopAllClips,
    CreateSession {
        payload: SessionCreateRequest,
    },
    FetchSession {
        session_id: String,
    },
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

    if let Some(session_id) = &request.session_id {
        let label = AppState::short_session_id(session_id);
        parts.push(format!("session {label}"));
    }

    if let Some(layer) = request.clip_layer {
        let label = clip_layer_label(layer).to_ascii_lowercase();
        parts.push(format!("layer {label}"));
    }
    if let Some(bars) = request.clip_bars {
        parts.push(format!("bars {}", bars));
    }

    if let Some(plan) = &request.plan {
        parts.push(format!("plan {} sections", plan.sections.len()));
    }

    if matches!(request.mode, Some(GenerationMode::Motif)) {
        parts.push("mode motif".to_string());
    }

    parts.join(", ")
}

fn completion_message(status: &GenerationStatus, artifact: &LocalArtifact) -> String {
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
    format!("Job {} completed ({detail}) → {}", status.job_id, artifact.local_path.display())
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

fn beats_per_bar(signature: &str) -> f32 {
    let mut parts = signature.split('/');
    let numerator = parts.next().and_then(|s| s.parse::<f32>().ok()).unwrap_or(4.0);
    let denominator = parts.next().and_then(|s| s.parse::<f32>().ok()).unwrap_or(4.0);
    if denominator <= 0.0 {
        return numerator;
    }
    numerator * (4.0 / denominator)
}

fn clip_prompt_text(prompt: &str, descriptor: &ThemeDescriptor, layer: ClipLayer) -> String {
    let layer_desc = match layer {
        ClipLayer::Rhythm => {
            "Focus on tight percussion and groove-locked drums with tasteful syncopation."
        }
        ClipLayer::Bass => {
            "Deliver a supportive bassline that locks to the kick and outlines the harmony."
        }
        ClipLayer::Harmony => {
            "Layer warm harmonic stabs or pads that reinforce the progression without overcrowding."
        }
        ClipLayer::Lead => {
            "Craft a memorable lead motif that plays call-and-response with the established theme."
        }
        ClipLayer::Textures => "Add evolving textures and atmosphere to widen the stereo field.",
        ClipLayer::Vocals => "Improvise expressive wordless vocals floating above the arrangement.",
    };
    let motif = descriptor.motif.as_str();
    format!(
        "{} {} Reinforce the motif \"{}\" with seamless looping.",
        prompt.trim(),
        layer_desc,
        motif
    )
}

fn clip_energy(layer: ClipLayer) -> SectionEnergy {
    match layer {
        ClipLayer::Rhythm | ClipLayer::Lead => SectionEnergy::High,
        ClipLayer::Harmony | ClipLayer::Bass => SectionEnergy::Medium,
        ClipLayer::Textures | ClipLayer::Vocals => SectionEnergy::Low,
    }
}

fn clip_directive(layer: ClipLayer) -> String {
    match layer {
        ClipLayer::Rhythm => "drive motif groove".to_string(),
        ClipLayer::Bass => "anchor harmonic floor".to_string(),
        ClipLayer::Harmony => "reinforce progression".to_string(),
        ClipLayer::Lead => "embellish motif".to_string(),
        ClipLayer::Textures => "expand atmosphere".to_string(),
        ClipLayer::Vocals => "float vocalise".to_string(),
    }
}

fn clip_orchestration(layer: ClipLayer, theme: Option<&ThemeDescriptor>) -> SectionOrchestration {
    let mut orchestration = SectionOrchestration::default();
    let instrumentation = theme.map(|t| t.instrumentation.clone()).unwrap_or_default();
    match layer {
        ClipLayer::Rhythm => {
            orchestration.rhythm = if instrumentation.is_empty() {
                vec!["tight kit".to_string(), "syncopated percussion".to_string()]
            } else {
                instrumentation
            };
        }
        ClipLayer::Bass => {
            orchestration.bass = if instrumentation.is_empty() {
                vec!["electric bass".to_string(), "synth low-end".to_string()]
            } else {
                instrumentation
            };
        }
        ClipLayer::Harmony => {
            orchestration.harmony = if instrumentation.is_empty() {
                vec!["chord stabs".to_string(), "lush pads".to_string()]
            } else {
                instrumentation
            };
        }
        ClipLayer::Lead => {
            orchestration.lead = if instrumentation.is_empty() {
                vec!["expressive lead".to_string(), "hook synth".to_string()]
            } else {
                instrumentation
            };
        }
        ClipLayer::Textures => {
            orchestration.textures = if instrumentation.is_empty() {
                vec!["ambient wash".to_string(), "granular shimmer".to_string()]
            } else {
                instrumentation
            };
        }
        ClipLayer::Vocals => {
            orchestration.vocals = if instrumentation.is_empty() {
                vec!["wordless vocal".to_string(), "airy choir".to_string()]
            } else {
                instrumentation
            };
        }
    }
    orchestration
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        ClipLayer, CompositionPlan, CompositionSection, GenerationMetadata, GenerationMode,
        SectionEnergy, SectionOrchestration, SectionRole, SessionSummary,
    };
    use crate::{
        config::AppConfig,
        session::{ClipSlotStatus, SceneSnapshot, SessionSnapshot},
    };
    use chrono::Utc;
    use serde_json::json;

    fn app_state_without_splash() -> AppState {
        let mut state = AppState::new(AppConfig::default());
        state.disable_splash();
        state.pending_snapshot = None;
        state
    }

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
            model_id: "musicgen-stereo-medium".into(),
            duration_seconds: 8,
            extras: json!({
                "backend": "musicgen",
                "placeholder": false,
                "guidance_scale": 3.0,
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

        let line = completion_message(&status, &artifact);
        assert!(line.contains("musicgen"));
        assert!(line.contains("8s"));
        assert!(line.contains("44100 Hz"));
        assert!(line.contains("hash abc123ef"));
        assert!(line.contains("→ /tmp/job123/job123.wav"));
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
            model_id: "musicgen-stereo-medium".into(),
            duration_seconds: 6,
            extras: json!({
                "backend": "musicgen",
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

        let line = completion_message(&status, &artifact);
        assert!(line.contains("placeholder via musicgen"));
        assert!(line.contains("reason: pipeline_unavailable"));
        assert!(!line.contains("Hz"));
        assert!(line.contains("hash deadbeef"));
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
            model_id: "musicgen-stereo-medium".into(),
            duration_seconds: 5,
            extras: json!({
                "backend": "musicgen",
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

        let line = completion_message(&status, &artifact);
        assert!(line.contains("seed 7"));
    }

    #[test]
    fn request_summary_formats_values() {
        let request = GenerationRequest {
            prompt: "test".into(),
            seed: Some(99),
            duration_seconds: 12,
            model_id: "musicgen-stereo-medium".into(),
            session_id: None,
            clip_layer: None,
            clip_scene_index: None,
            clip_bars: None,
            mode: Some(GenerationMode::FullTrack),
            cfg_scale: Some(5.5),
            scheduler: None,
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
        assert!(summary.contains("musicgen-stereo-medium"));
        assert!(summary.contains("12s"));
        assert!(summary.contains("cfg 5.5"));
        assert!(summary.contains("seed 99"));
    }

    fn sample_plan() -> CompositionPlan {
        CompositionPlan {
            version: "v4".to_string(),
            tempo_bpm: 120,
            time_signature: "4/4".to_string(),
            key: "C major".to_string(),
            total_bars: 4,
            total_duration_seconds: 8.0,
            theme: None,
            sections: vec![CompositionSection {
                section_id: "s00".to_string(),
                role: SectionRole::Motif,
                label: "Motif".to_string(),
                prompt: "motif".to_string(),
                bars: 4,
                target_seconds: 8.0,
                energy: SectionEnergy::Medium,
                model_id: None,
                seed_offset: None,
                transition: None,
                motif_directive: None,
                variation_axes: Vec::new(),
                cadence_hint: None,
                orchestration: SectionOrchestration::default(),
            }],
        }
    }

    fn session_summary_with_plan(plan: Option<CompositionPlan>) -> SessionSummary {
        SessionSummary {
            session_id: "session-test".to_string(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            name: None,
            tempo_bpm: Some(120),
            key: Some("C major".to_string()),
            time_signature: Some("4/4".to_string()),
            seed_job_id: None,
            seed_prompt: Some("motif seed".to_string()),
            seed_plan: plan,
            theme: None,
            clip_count: 0,
            clips: Vec::new(),
        }
    }

    #[test]
    fn apply_pending_snapshot_restores_layout() {
        let mut state = app_state_without_splash();
        state.showing_splash = true;
        state.pending_snapshot = Some(SessionSnapshot {
            session_id: Some("session-abc123".to_string()),
            scenes: vec![SceneSnapshot { index: 0, name: "Intro".to_string() }],
            focused_layer: ClipLayer::Lead,
            focused_scene: 0,
            active_layer: Some(ClipLayer::Lead),
            active_scene: Some(0),
        });

        state.status_lines.clear();
        assert!(state.apply_pending_snapshot());
        assert!(!state.showing_splash());
        assert_eq!(state.restored_session_id, Some("session-abc123".to_string()));
        assert_eq!(state.session_view.focused(), (ClipLayer::Lead, 0));
        assert!(state
            .status_lines
            .last()
            .expect("expected status line")
            .contains("Restored session layout"));
    }

    #[test]
    fn reset_session_state_clears_session_context() {
        let mut state = app_state_without_splash();
        let plan = sample_plan();
        state.session = Some(session_summary_with_plan(Some(plan.clone())));
        state.restored_session_id = Some("session-old".to_string());

        let request = GenerationRequest {
            prompt: "inspired groove".to_string(),
            seed: None,
            duration_seconds: 12,
            model_id: "musicgen-stereo-medium".to_string(),
            session_id: Some("session-old".to_string()),
            clip_layer: None,
            clip_scene_index: None,
            clip_bars: None,
            mode: Some(GenerationMode::FullTrack),
            cfg_scale: Some(5.5),
            scheduler: None,
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
        let status = GenerationStatus {
            job_id: "job-123".to_string(),
            state: JobState::Queued,
            progress: 0.0,
            message: None,
            updated_at: Utc::now(),
        };

        state.jobs.insert(
            status.job_id.clone(),
            JobEntry {
                prompt: "inspired groove".to_string(),
                request,
                status: status.clone(),
                plan,
                artifact: None,
            },
        );
        state.focused_job = Some(status.job_id.clone());
        state.playback = Some(PlaybackState {
            job_id: status.job_id.clone(),
            path: PathBuf::from("clip.wav"),
            duration: StdDuration::from_secs(12),
            started_at: Utc::now(),
            is_playing: true,
        });

        state.reset_session_state();

        assert!(state.session.is_none());
        assert!(state.restored_session_id.is_none());
        assert!(state.jobs.is_empty());
        assert!(state.focused_job.is_none());
        assert!(state.playback.is_none());
        assert_eq!(state.session_view.focused(), (ClipLayer::Rhythm, 0));
        assert!(!state.showing_splash());
    }

    #[test]
    fn toggle_active_clip_target_stays_selected_once_active() {
        let mut state = app_state_without_splash();
        if let Some((layer, scene)) = state.active_clip_target() {
            // Clear any persisted selection from a prior CLI snapshot on disk.
            assert!(!state.session_view.toggle_active_target(layer, scene));
        }
        let initial_target = state.session_view.focused();
        assert!(state.active_clip_target().is_none());
        assert!(state.toggle_active_clip_target());
        assert_eq!(state.active_clip_target(), Some(initial_target));
        assert!(!state.toggle_active_clip_target());
        assert_eq!(state.active_clip_target(), None);
    }

    #[test]
    fn build_clip_payload_rejects_missing_motif() {
        let mut state = app_state_without_splash();
        state.session = Some(session_summary_with_plan(None));
        let error =
            state.build_clip_payload_for(ClipLayer::Bass, 0, Some("groovy bass")).unwrap_err();
        assert!(error.contains("Motif not ready"), "expected motif readiness error, got {error}");
    }

    #[test]
    fn build_clip_payload_targets_selected_layer() {
        let mut state = app_state_without_splash();
        let plan = sample_plan();
        state.session = Some(session_summary_with_plan(Some(plan.clone())));
        let (request, built_plan, used_prompt) =
            state.build_clip_payload_for(ClipLayer::Lead, 0, Some("lead line")).unwrap();
        assert_eq!(request.clip_layer, Some(ClipLayer::Lead));
        assert_eq!(request.clip_scene_index, Some(0));
        assert_eq!(request.mode, Some(GenerationMode::Clip));
        assert_eq!(used_prompt, "lead line");
        assert_eq!(built_plan.sections.len(), 1);
        assert!(built_plan.sections[0].variation_axes.contains(&"lead".to_string()));
        assert!(plan.sections[0].variation_axes.is_empty());
    }

    #[test]
    fn clip_job_queue_marks_slot_pending() {
        let mut state = app_state_without_splash();
        let plan = sample_plan();
        let summary = session_summary_with_plan(Some(plan));
        state.session = Some(summary.clone());
        state.session_view.apply_summary(&summary);

        let (request, built_plan, prompt) =
            state.build_clip_payload_for(ClipLayer::Lead, 1, Some("lead layer")).unwrap();
        let status = GenerationStatus {
            job_id: "job-clip-1".into(),
            state: JobState::Queued,
            progress: 0.0,
            message: None,
            updated_at: Utc::now(),
        };

        state.handle_event(AppEvent::JobQueued {
            status: status.clone(),
            prompt: prompt.clone(),
            request: request.clone(),
            plan: built_plan.clone(),
        });

        let slot = state.session_view.slot(ClipLayer::Lead, 1);
        assert_eq!(slot.status, ClipSlotStatus::Pending);
        assert_eq!(slot.prompt.as_deref(), Some("lead layer"));
        assert_eq!(slot.job_id.as_deref(), Some("job-clip-1"));
        assert_eq!(
            state.session_view.pending_scene_for_job("job-clip-1"),
            Some((ClipLayer::Lead, 1))
        );
    }

    #[test]
    fn clip_job_completion_marks_slot_ready() {
        let mut state = app_state_without_splash();
        let plan = sample_plan();
        let summary = session_summary_with_plan(Some(plan.clone()));
        state.session = Some(summary.clone());
        state.session_view.apply_summary(&summary);

        let (request, built_plan, prompt) =
            state.build_clip_payload_for(ClipLayer::Bass, 2, Some("bass groove")).unwrap();
        let queued_status = GenerationStatus {
            job_id: "job-clip-2".into(),
            state: JobState::Queued,
            progress: 0.0,
            message: None,
            updated_at: Utc::now(),
        };
        state.handle_event(AppEvent::JobQueued {
            status: queued_status.clone(),
            prompt: prompt.clone(),
            request: request.clone(),
            plan: built_plan.clone(),
        });

        let artifact_plan =
            CompositionPlan { sections: built_plan.sections.clone(), ..built_plan.clone() };
        let metadata = GenerationMetadata {
            prompt: prompt.clone(),
            seed: Some(7),
            model_id: request.model_id.clone(),
            duration_seconds: 12,
            extras: json!({ "backend": "musicgen", "placeholder": false }),
            plan: Some(artifact_plan.clone()),
        };
        let artifact = LocalArtifact {
            descriptor: GenerationArtifact {
                job_id: "job-clip-2".into(),
                artifact_path: "/tmp/job-clip-2.wav".into(),
                metadata,
            },
            local_path: PathBuf::from("/tmp/job-clip-2/job-clip-2.wav"),
        };
        let completed_status =
            GenerationStatus { state: JobState::Succeeded, progress: 1.0, ..queued_status.clone() };

        state.handle_event(AppEvent::JobCompleted {
            status: completed_status,
            artifact: artifact.clone(),
        });

        let slot = state.session_view.slot(ClipLayer::Bass, 2);
        assert_eq!(slot.status, ClipSlotStatus::Ready);
        assert_eq!(slot.artifact.as_deref(), Some(Path::new("/tmp/job-clip-2/job-clip-2.wav")));
        assert_eq!(slot.duration_seconds, Some(12.0));
        assert_eq!(slot.bars, artifact_plan.sections.first().map(|section| section.bars as u8));
        assert!(state.session_view.pending_scene_for_job("job-clip-2").is_none());
    }

    #[test]
    fn handle_command_updates_duration() {
        let mut state = app_state_without_splash();
        let result = state.handle_command("/duration 120");
        assert!(result.is_ok());
        assert_eq!(state.generation_config.duration_seconds, 120);
    }

    #[test]
    fn handle_command_rejects_invalid_cfg() {
        let mut state = app_state_without_splash();
        let result = state.handle_command("/cfg 80");
        assert!(result.is_err());
    }

    #[test]
    fn build_request_reflects_generation_config() {
        let mut state = app_state_without_splash();
        let _ = state.handle_command("/duration 120");
        let _ = state.handle_command("/cfg 6.5");
        let _ = state.handle_command("/seed 77");
        let (request, plan) = state.build_generation_payload("lively strings");
        assert_eq!(request.duration_seconds, 120);
        assert_eq!(request.cfg_scale, Some(6.5));
        assert_eq!(request.seed, Some(77));
        assert_eq!(request.model_id, "musicgen-stereo-medium");
        assert_eq!(request.mode, Some(GenerationMode::FullTrack));
        assert!(request.plan.is_some());
        assert_eq!(plan.sections.len(), request.plan.as_ref().unwrap().sections.len());
    }

    #[test]
    fn build_generation_payload_supports_motif_override() {
        let state = app_state_without_splash();
        let overrides = GenerationOverrides {
            model_id: Some("musicgen-stereo-medium".to_string()),
            duration_seconds: Some(MOTIF_PRESET_DURATION_SECONDS),
            mode: Some(GenerationMode::Motif),
        };
        let (request, plan) =
            state.build_generation_payload_with_overrides("motif spotlight", Some(&overrides));
        assert_eq!(request.mode, Some(GenerationMode::Motif));
        assert_eq!(request.duration_seconds, MOTIF_PRESET_DURATION_SECONDS);
        assert_eq!(plan.sections.len(), 1);
        assert!(matches!(plan.sections[0].role, SectionRole::Motif));
        let summary = format_request_summary(&request);
        assert!(summary.contains("mode motif"));
    }

    #[test]
    fn ingest_health_payload_populates_backend_status() {
        let mut state = app_state_without_splash();
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
