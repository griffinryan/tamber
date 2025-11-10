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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clip_layer: Option<ClipLayer>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clip_scene_index: Option<u16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub clip_bars: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<GenerationMode>,
    pub cfg_scale: Option<f32>,
    pub scheduler: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub musicgen_top_k: Option<u16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub musicgen_top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub musicgen_temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub musicgen_cfg_coef: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub musicgen_two_step_cfg: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_sample_rate: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_bit_depth: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plan: Option<CompositionPlan>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GenerationMode {
    FullTrack,
    Motif,
    Clip,
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
    Chorus,
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ClipLayer {
    Rhythm,
    Bass,
    Harmony,
    Lead,
    Textures,
    Vocals,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub motif_directive: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub variation_axes: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cadence_hint: Option<String>,
    #[serde(default)]
    pub orchestration: SectionOrchestration,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct SectionOrchestration {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub rhythm: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bass: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub harmony: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub lead: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub textures: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vocals: Vec<String>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionClipSummary {
    pub job_id: String,
    pub prompt: String,
    pub state: JobState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_seconds: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer: Option<ClipLayer>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scene_index: Option<u16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bars: Option<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SessionCreateRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tempo_bpm: Option<u16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time_signature: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionClipRequest {
    pub layer: ClipLayer,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bars: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scene_index: Option<u16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation: Option<GenerationRequest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    pub session_id: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tempo_bpm: Option<u16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time_signature: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed_job_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed_prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed_plan: Option<CompositionPlan>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub theme: Option<ThemeDescriptor>,
    pub clip_count: usize,
    #[serde(default)]
    pub clips: Vec<SessionClipSummary>,
}
