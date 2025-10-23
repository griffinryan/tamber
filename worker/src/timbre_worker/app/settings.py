from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_config_dir() -> Path:
    return Path.home() / ".config" / "timbre"


def _default_artifact_root() -> Path:
    return Path.home() / "Music" / "Timbre"


class Settings(BaseSettings):
    """Runtime configuration for the Timbre worker process."""

    model_config = SettingsConfigDict(env_prefix="TIMBRE_", env_nested_delimiter="__", extra="ignore")

    config_dir: Path = Field(default_factory=_default_config_dir)
    artifact_root: Path = Field(default_factory=_default_artifact_root)
    default_model_id: str = "riffusion-v1"
    default_duration_seconds: int = Field(default=8, ge=1, le=30)
    inference_device: str | None = Field(
        default=None,
        description="Override inference device selection (cpu, mps, cuda).",
    )
    riffusion_allow_inference: bool = Field(
        default=True,
        description="Enable Riffusion pipeline loading; disable to force placeholder audio.",
    )

    def ensure_directories(self) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_root.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
