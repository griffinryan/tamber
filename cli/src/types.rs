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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plan: Option<CompositionPlan>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plan: Option<CompositionPlan>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationArtifact {
    pub job_id: String,
    pub artifact_path: String,
    pub metadata: GenerationMetadata,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SectionRole {
    Intro,
    Motif,
    Development,
    Bridge,
    Resolution,
    Outro,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SectionEnergy {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ThemeDescriptor {
    pub motif: String,
    #[serde(default)]
    pub instrumentation: Vec<String>,
    pub rhythm: String,
    #[serde(default)]
    pub dynamic_curve: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub texture: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompositionSection {
    pub section_id: String,
    pub role: SectionRole,
    pub label: String,
    pub prompt: String,
    pub bars: u8,
    pub target_seconds: f32,
    pub energy: SectionEnergy,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed_offset: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transition: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompositionPlan {
    pub version: String,
    pub tempo_bpm: u16,
    pub time_signature: String,
    pub key: String,
    pub total_bars: u16,
    pub total_duration_seconds: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub theme: Option<ThemeDescriptor>,
    pub sections: Vec<CompositionSection>,
}
