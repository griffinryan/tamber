from __future__ import annotations

from pathlib import Path

import pytest

from timbre_worker.app.models import GenerationRequest
from timbre_worker.app.settings import Settings
from timbre_worker.services.riffusion import RiffusionService


@pytest.mark.asyncio
async def test_riffusion_service_placeholder_generation(tmp_path: Path) -> None:
    settings = Settings(artifact_root=tmp_path / "artifacts", config_dir=tmp_path / "config")
    settings.ensure_directories()
    service = RiffusionService(settings)

    request = GenerationRequest(prompt="soft synthwave", duration_seconds=2, model_id="riffusion-v1")
    artifact = await service.generate("job123", request)

    path = Path(artifact.artifact_path)
    assert path.exists()
    assert artifact.metadata.extras.get("placeholder") is True
    assert artifact.metadata.prompt == "soft synthwave"
