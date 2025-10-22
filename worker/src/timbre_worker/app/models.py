from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=512)
    seed: Optional[int] = Field(default=None, ge=0)
    duration_seconds: int = Field(default=8, ge=1, le=30)
    model_id: str = Field(default="riffusion-v1")
    cfg_scale: Optional[float] = Field(default=None, ge=0.0, le=20.0)
    scheduler: Optional[str] = Field(default=None, max_length=64)


class GenerationStatus(BaseModel):
    job_id: str
    state: JobState
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    message: Optional[str] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class GenerationMetadata(BaseModel):
    prompt: str
    seed: Optional[int]
    model_id: str
    duration_seconds: int
    extras: dict[str, Any] = Field(default_factory=dict)


class GenerationArtifact(BaseModel):
    job_id: str
    artifact_path: str
    metadata: GenerationMetadata
