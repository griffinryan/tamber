use crate::{
    config::AppConfig,
    types::{GenerationArtifact, GenerationStatus, JobState},
};
use chrono::{DateTime, Utc};
use indexmap::{map::Iter, IndexMap};
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
                    self.push_status_line(format!("Job {} completed", status.job_id));
                    let metadata = &artifact.descriptor.metadata;
                    self.append_chat(
                        ChatRole::Worker,
                        format!(
                            "Job {} completed ({}, {}s) → {}",
                            status.job_id,
                            metadata.model_id,
                            metadata.duration_seconds,
                            artifact.local_path.display()
                        ),
                    );
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
