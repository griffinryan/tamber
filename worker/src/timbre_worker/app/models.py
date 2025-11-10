from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class SectionRole(str, Enum):
    INTRO = "intro"
    MOTIF = "motif"
    CHORUS = "chorus"
    DEVELOPMENT = "development"
    BRIDGE = "bridge"
    RESOLUTION = "resolution"
    OUTRO = "outro"


class SectionEnergy(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class GenerationMode(str, Enum):
    FULL_TRACK = "full_track"
    MOTIF = "motif"
    CLIP = "clip"


class ClipLayer(str, Enum):
    RHYTHM = "rhythm"
    BASS = "bass"
    HARMONY = "harmony"
    LEAD = "lead"
    TEXTURES = "textures"
    VOCALS = "vocals"


class ThemeDescriptor(BaseModel):
    motif: str = Field(..., min_length=1, max_length=128)
    instrumentation: list[str] = Field(default_factory=list)
    rhythm: str = Field(default="steady pulse", min_length=1, max_length=128)
    dynamic_curve: list[str] = Field(default_factory=list)
    texture: Optional[str] = Field(default=None, max_length=128)


class SectionOrchestration(BaseModel):
    rhythm: list[str] = Field(default_factory=list)
    bass: list[str] = Field(default_factory=list)
    harmony: list[str] = Field(default_factory=list)
    lead: list[str] = Field(default_factory=list)
    textures: list[str] = Field(default_factory=list)
    vocals: list[str] = Field(default_factory=list)


class CompositionSection(BaseModel):
    section_id: str = Field(..., min_length=2, max_length=32)
    role: SectionRole
    label: str = Field(..., min_length=1, max_length=64)
    prompt: str = Field(..., min_length=1, max_length=512)
    bars: int = Field(..., ge=1, le=128)
    target_seconds: float = Field(..., gt=0.5, le=120.0)
    energy: SectionEnergy = Field(default=SectionEnergy.MEDIUM)
    model_id: Optional[str] = Field(default=None, max_length=128)
    seed_offset: Optional[int] = Field(default=None)
    transition: Optional[str] = Field(default=None, max_length=128)
    motif_directive: Optional[str] = Field(default=None, max_length=128)
    variation_axes: list[str] = Field(default_factory=list)
    cadence_hint: Optional[str] = Field(default=None, max_length=128)
    orchestration: SectionOrchestration = Field(default_factory=SectionOrchestration)


class CompositionPlan(BaseModel):
    version: str = Field(default="v1", max_length=16)
    tempo_bpm: int = Field(..., ge=40, le=200)
    time_signature: str = Field(default="4/4", min_length=3, max_length=8)
    key: str = Field(default="C major", min_length=3, max_length=32)
    total_bars: int = Field(..., ge=1, le=512)
    total_duration_seconds: float = Field(..., gt=1.0, le=300.0)
    theme: Optional[ThemeDescriptor] = Field(default=None)
    sections: list[CompositionSection]


class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=512)
    seed: Optional[int] = Field(default=None, ge=0)
    duration_seconds: int = Field(default=120, ge=1, le=300)
    model_id: str = Field(default="musicgen-stereo-medium")
    session_id: Optional[str] = Field(default=None, min_length=8, max_length=64)
    mode: GenerationMode = Field(default=GenerationMode.FULL_TRACK)
    clip_layer: Optional[ClipLayer] = Field(default=None)
    clip_scene_index: Optional[int] = Field(default=None, ge=0, le=2)
    clip_bars: Optional[int] = Field(default=None, ge=1, le=64)
    cfg_scale: Optional[float] = Field(default=None, ge=0.0, le=20.0)
    scheduler: Optional[str] = Field(default=None, max_length=64)
    musicgen_top_k: Optional[int] = Field(default=None, ge=0, le=2048)
    musicgen_top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    musicgen_temperature: Optional[float] = Field(default=None, ge=0.0, le=4.0)
    musicgen_cfg_coef: Optional[float] = Field(default=None, ge=0.0, le=20.0)
    musicgen_two_step_cfg: Optional[bool] = Field(default=None)
    output_sample_rate: Optional[int] = Field(default=None, ge=8_000, le=192_000)
    output_bit_depth: Optional[str] = Field(default=None, max_length=16)
    output_format: Optional[str] = Field(default=None, max_length=16)
    plan: Optional[CompositionPlan] = None


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


class GenerationStatus(BaseModel):
    job_id: str
    state: JobState
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    message: Optional[str] = None
    updated_at: datetime = Field(default_factory=_utc_now)


class GenerationMetadata(BaseModel):
    prompt: str
    seed: Optional[int]
    model_id: str
    duration_seconds: int
    extras: dict[str, Any] = Field(default_factory=dict)
    plan: Optional[CompositionPlan] = None


class GenerationArtifact(BaseModel):
    job_id: str
    artifact_path: str
    metadata: GenerationMetadata


class SessionClipSummary(BaseModel):
    job_id: str
    prompt: str
    state: JobState
    duration_seconds: Optional[float] = None
    artifact_path: Optional[str] = None
    layer: Optional[ClipLayer] = None
    scene_index: Optional[int] = None
    bars: Optional[int] = None


class SessionSummary(BaseModel):
    session_id: str
    created_at: datetime
    updated_at: datetime
    name: Optional[str] = None
    tempo_bpm: Optional[int] = None
    key: Optional[str] = None
    time_signature: Optional[str] = None
    seed_job_id: Optional[str] = None
    seed_prompt: Optional[str] = None
    seed_plan: Optional[CompositionPlan] = None
    theme: Optional[ThemeDescriptor] = None
    clip_count: int = 0
    clips: list[SessionClipSummary] = Field(default_factory=list)


class SessionCreateRequest(BaseModel):
    name: Optional[str] = Field(default=None, max_length=64)
    prompt: Optional[str] = Field(default=None, max_length=512)
    tempo_bpm: Optional[int] = Field(default=None, ge=40, le=300)
    key: Optional[str] = Field(default=None, max_length=32)
    time_signature: Optional[str] = Field(default=None, max_length=8)


class SessionClipRequest(BaseModel):
    layer: ClipLayer
    prompt: Optional[str] = Field(default=None, max_length=512)
    bars: Optional[int] = Field(default=None, ge=1, le=64)
    scene_index: Optional[int] = Field(default=None, ge=0, le=2)
    generation: Optional[GenerationRequest] = Field(default=None)
