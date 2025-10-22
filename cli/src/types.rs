#![allow(dead_code)]

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum JobState {
    Queued,
    Running,
    Succeeded,
    Failed,
}

impl JobState {
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Succeeded | Self::Failed)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationRequest {
    pub prompt: String,
    pub seed: Option<u64>,
    pub duration_seconds: u8,
    pub model_id: String,
    pub cfg_scale: Option<f32>,
    pub scheduler: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationStatus {
    pub job_id: String,
    pub state: JobState,
    #[serde(default)]
    pub progress: f32,
    #[serde(default)]
    pub message: Option<String>,
    pub updated_at: DateTime<Utc>,
}

impl GenerationStatus {
    pub fn is_terminal(&self) -> bool {
        self.state.is_terminal()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationMetadata {
    pub prompt: String,
    pub seed: Option<u64>,
    pub model_id: String,
    pub duration_seconds: u8,
    #[serde(default)]
    pub extras: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationArtifact {
    pub job_id: String,
    pub artifact_path: String,
    pub metadata: GenerationMetadata,
}
