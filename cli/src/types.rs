use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JobState {
    Queued,
    Running,
    Succeeded,
    Failed,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GenerationRequest {
    pub prompt: String,
    pub seed: Option<u64>,
    pub duration_seconds: u8,
    pub model_id: String,
    pub cfg_scale: Option<f32>,
    pub scheduler: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GenerationStatus {
    pub job_id: String,
    pub state: JobState,
    pub progress: f32,
    pub message: Option<String>,
    pub updated_at: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GenerationMetadata {
    pub prompt: String,
    pub seed: Option<u64>,
    pub model_id: String,
    pub duration_seconds: u8,
    #[serde(default)]
    pub extras: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GenerationArtifact {
    pub job_id: String,
    pub artifact_path: String,
    pub metadata: GenerationMetadata,
}
