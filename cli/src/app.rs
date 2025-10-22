use crate::{
    config::AppConfig,
    types::{GenerationArtifact, GenerationMetadata, GenerationStatus, JobState},
};
use chrono::{DateTime, Utc};
use indexmap::{map::Iter, IndexMap};
use serde_json::Value;
use std::path::PathBuf;

const MAX_CHAT_ENTRIES: usize = 200;
const MAX_STATUS_LINES: usize = 8;

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
pub struct JobEntry {
    pub prompt: String,
    pub status: GenerationStatus,
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

#[derive(Debug)]
pub struct AppState {
    pub input: String,
    pub chat: Vec<ChatEntry>,
    pub jobs: IndexMap<String, JobEntry>,
    pub focused_job: Option<String>,
    pub status_lines: Vec<String>,
}

impl AppState {
    pub fn new(config: AppConfig) -> Self {
        let _ = config;
        Self {
            input: String::new(),
            chat: Vec::new(),
            jobs: IndexMap::new(),
            focused_job: None,
            status_lines: Vec::new(),
        }
    }

    pub fn handle_event(&mut self, event: AppEvent) {
        match event {
            AppEvent::Info(message) => self.push_status_line(message),
            AppEvent::Error(message) => {
                self.push_status_line(format!("Error: {message}"));
                self.append_chat(ChatRole::System, format!("{message}"));
            }
            AppEvent::JobQueued { status, prompt } => {
                self.jobs.insert(
                    status.job_id.clone(),
                    JobEntry { prompt: prompt.clone(), status: status.clone(), artifact: None },
                );
                self.focused_job = Some(status.job_id.clone());
                self.push_status_line(format!("Job {} queued", status.job_id));
                self.append_chat(ChatRole::Worker, format!("Queued job {}", status.job_id));
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
                    self.focused_job = Some(status.job_id.clone());
                    let (status_line, chat_line) = completion_messages(&status, &artifact);
                    self.push_status_line(status_line);
                    self.append_chat(ChatRole::Worker, chat_line);
                }
            }
        }
    }

    pub fn append_chat(&mut self, role: ChatRole, content: String) {
        self.chat.push(ChatEntry { role, content, timestamp: Utc::now() });
        if self.chat.len() > MAX_CHAT_ENTRIES {
            let overflow = self.chat.len() - MAX_CHAT_ENTRIES;
            self.chat.drain(0..overflow);
        }
    }

    pub fn push_status_line(&mut self, line: String) {
        self.status_lines.push(line);
        if self.status_lines.len() > MAX_STATUS_LINES {
            let overflow = self.status_lines.len() - MAX_STATUS_LINES;
            self.status_lines.drain(0..overflow);
        }
    }

    pub fn jobs_iter(&self) -> Iter<'_, String, JobEntry> {
        self.jobs.iter()
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
}

#[derive(Debug, Clone)]
pub enum AppEvent {
    Info(String),
    Error(String),
    JobQueued { status: GenerationStatus, prompt: String },
    JobUpdated { status: GenerationStatus },
    JobCompleted { status: GenerationStatus, artifact: LocalArtifact },
}

#[derive(Debug, Clone)]
pub enum AppCommand {
    SubmitPrompt { prompt: String },
    PlayJob { job_id: String },
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

    let detail = parts.join(", ");
    let status_line = format!("Job {} completed ({detail})", status.job_id);
    let chat_line =
        format!("Job {} completed ({detail}) → {}", status.job_id, artifact.local_path.display());
    (status_line, chat_line)
}

#[cfg(test)]
mod tests {
    use super::*;
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
}
