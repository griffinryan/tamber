from __future__ import annotations

import hashlib
from pathlib import Path
import types
from typing import Any, Dict

import numpy as np
import pytest

from timbre_worker.app.models import GenerationRequest
from timbre_worker.app.settings import Settings
from timbre_worker.services.riffusion import DEFAULT_GUIDANCE_SCALE, PipelineHandle, RiffusionService


@pytest.mark.asyncio
async def test_riffusion_service_placeholder_generation(tmp_path: Path) -> None:
    settings = Settings(artifact_root=tmp_path / "artifacts", config_dir=tmp_path / "config")
    settings.ensure_directories()
    service = RiffusionService(settings)

    request = GenerationRequest(prompt="soft synthwave", duration_seconds=2, model_id="riffusion-v1")
    artifact = await service.generate("job123", request)

    path = Path(artifact.artifact_path)
    assert path.exists()
    extras = artifact.metadata.extras
    assert extras.get("placeholder") is True
    assert extras.get("placeholder_reason") is not None
    assert extras.get("backend") == "riffusion"
    assert extras.get("prompt_hash") == hashlib.blake2s(b"soft synthwave", digest_size=8).hexdigest()
    assert artifact.metadata.prompt == "soft synthwave"


@pytest.mark.asyncio
async def test_placeholder_audio_varies_by_prompt(tmp_path: Path) -> None:
    settings = Settings(artifact_root=tmp_path / "artifacts", config_dir=tmp_path / "config")
    settings.ensure_directories()
    service = RiffusionService(settings)

    prompt_a = "minimal ambient drones"
    prompt_b = "energetic drum and bass"

    request_a = GenerationRequest(prompt=prompt_a, duration_seconds=2, model_id="riffusion-v1")
    request_b = GenerationRequest(prompt=prompt_b, duration_seconds=2, model_id="riffusion-v1")

    artifact_a = await service.generate("job-a", request_a)
    artifact_b = await service.generate("job-b", request_b)

    data_a = Path(artifact_a.artifact_path).read_bytes()
    data_b = Path(artifact_b.artifact_path).read_bytes()
    assert data_a != data_b

    extras_a = artifact_a.metadata.extras
    extras_b = artifact_b.metadata.extras
    assert extras_a.get("placeholder") is True
    assert extras_b.get("placeholder") is True
    assert extras_a.get("prompt_hash") != extras_b.get("prompt_hash")


@pytest.mark.asyncio
async def test_riffusion_service_pipeline_receives_prompt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _StubGenerator:
        def manual_seed(self, seed: int) -> "_StubGenerator":
            return self

    stub_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
        float16=np.float16,
        float32=np.float32,
        Generator=lambda device=None: _StubGenerator(),
    )

    monkeypatch.setattr("timbre_worker.services.riffusion.torch", stub_torch, raising=False)
    monkeypatch.setattr("timbre_worker.services.riffusion.TORCH_IMPORT_ERROR", None, raising=False)

    calls: list[Dict[str, Any]] = []

    class _DummyPipeline:
        def __init__(self, store: list[Dict[str, Any]], sample_rate: int = 44100):
            self.sample_rate = sample_rate
            self._store = store

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            self._store.append(dict(kwargs))
            duration = kwargs.get("audio_length_in_s", 1)
            samples = max(1, int(self.sample_rate * duration))
            waveform = np.linspace(-0.5, 0.5, samples, dtype=np.float32)
            audio = np.stack([waveform, waveform], axis=0)
            return types.SimpleNamespace(audios=[audio], sample_rate=self.sample_rate)

    def loader(resolved_model_id: str) -> PipelineHandle:
        return PipelineHandle(pipeline=_DummyPipeline(calls), sample_rate=44100)

    settings = Settings(artifact_root=tmp_path / "artifacts", config_dir=tmp_path / "config")
    settings.ensure_directories()
    service = RiffusionService(settings, pipeline_loader=loader)

    request = GenerationRequest(prompt="vintage jazz trio", duration_seconds=3, model_id="riffusion-v1")
    artifact = await service.generate("job-real", request)

    assert Path(artifact.artifact_path).exists()
    extras = artifact.metadata.extras
    assert extras.get("placeholder") is False
    expected_hash = hashlib.blake2s(request.prompt.encode("utf-8"), digest_size=8).hexdigest()
    assert extras.get("prompt_hash") == expected_hash
    assert extras.get("sample_rate") == 44100
    assert extras.get("guidance_scale") == DEFAULT_GUIDANCE_SCALE

    assert calls, "pipeline should have been invoked"
    first_call = calls[0]
    assert first_call.get("prompt") == request.prompt
    assert first_call.get("guidance_scale") == DEFAULT_GUIDANCE_SCALE
    assert first_call.get("num_inference_steps") == 50
    assert first_call.get("audio_length_in_s") == request.duration_seconds

    request_cfg = GenerationRequest(
        prompt="dreamy modular sequence",
        duration_seconds=4,
        model_id="riffusion-v1",
        cfg_scale=4.5,
    )
    artifact_cfg = await service.generate("job-real-2", request_cfg)
    assert Path(artifact_cfg.artifact_path).exists()
    extras_cfg = artifact_cfg.metadata.extras
    assert extras_cfg.get("placeholder") is False
    assert extras_cfg.get("guidance_scale") == pytest.approx(4.5)
    assert calls[-1].get("guidance_scale") == pytest.approx(4.5)
