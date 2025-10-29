from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_config_dir() -> Path:
    return Path.home() / ".config" / "timbre"


def _default_artifact_root() -> Path:
    return Path.home() / "Music" / "Timbre"


class Settings(BaseSettings):
    """Runtime configuration for the Timbre worker process."""

    model_config = SettingsConfigDict(
        env_prefix="TIMBRE_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    config_dir: Path = Field(default_factory=_default_config_dir)
    artifact_root: Path = Field(default_factory=_default_artifact_root)
    default_model_id: str = "musicgen-stereo-medium"
    default_duration_seconds: int = Field(default=120, ge=1, le=300)
    inference_device: str | None = Field(
        default=None,
        description="Override inference device selection (cpu, mps, cuda).",
    )
    musicgen_default_model_id: str = Field(
        default="musicgen-stereo-medium",
        max_length=128,
        description="Default model identifier for the MusicGen backend.",
    )
    musicgen_top_k: int | None = Field(
        default=250,
        ge=0,
        le=2048,
        description="Default top-k sampling for MusicGen (0 disables).",
    )
    musicgen_top_p: float | None = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Default top-p sampling for MusicGen (0 disables).",
    )
    musicgen_temperature: float | None = Field(
        default=1.15,
        ge=0.0,
        le=4.0,
        description="Default sampling temperature for MusicGen generations.",
    )
    musicgen_cfg_coef: float | None = Field(
        default=3.0,
        ge=0.0,
        le=20.0,
        description="Default classifier-free guidance coefficient for MusicGen.",
    )
    musicgen_two_step_cfg: bool = Field(
        default=True,
        description="Enable MusicGen two-step classifier-free guidance by default.",
    )
    export_sample_rate: int = Field(
        default=48_000,
        ge=8_000,
        le=192_000,
        description="Target sample rate for exported waveforms.",
    )
    export_bit_depth: str = Field(
        default="pcm24",
        description="Default bit depth encoding for exported audio (pcm16, pcm24, pcm32, float32).",
        max_length=16,
    )
    export_format: str = Field(
        default="wav",
        description="Container format for exported audio artifacts.",
        max_length=16,
    )

    @model_validator(mode="after")
    def _align_backend_defaults(self) -> "Settings":
        min_duration = 90
        max_duration = 180
        if self.default_duration_seconds < min_duration:
            self.default_duration_seconds = min_duration
        elif self.default_duration_seconds > max_duration:
            self.default_duration_seconds = max_duration

        if "musicgen_default_model_id" not in self.model_fields_set:
            if self.default_model_id.lower().startswith("musicgen"):
                self.musicgen_default_model_id = self.default_model_id
        elif "default_model_id" not in self.model_fields_set:
            self.default_model_id = self.musicgen_default_model_id

        return self

    def ensure_directories(self) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_root.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
