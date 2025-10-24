from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from timbre_worker.app.models import (
    CompositionPlan,
    CompositionSection,
    GenerationRequest,
    SectionEnergy,
    SectionRole,
)
from timbre_worker.app.settings import Settings
from timbre_worker.services.riffusion import (
    DEFAULT_GUIDANCE_SCALE,
    MIN_RENDER_SECONDS,
    PipelineHandle,
    RiffusionService,
)


@pytest.mark.asyncio
async def test_riffusion_service_placeholder_generation(tmp_path: Path) -> None:
    settings = Settings(
        artifact_root=tmp_path / "artifacts",
        config_dir=tmp_path / "config",
        riffusion_allow_inference=False,
    )
    settings.ensure_directories()
    service = RiffusionService(settings)

    request = GenerationRequest(
        prompt="soft synthwave",
        duration_seconds=2,
        model_id="riffusion-v1",
    )
    artifact = await service.generate("job123", request)

    path = Path(artifact.artifact_path)
    assert path.exists()
    extras = artifact.metadata.extras
    assert extras.get("placeholder") is True
    assert extras.get("placeholder_reason") is not None
    assert extras.get("backend") == "riffusion"
    assert extras.get("sample_rate") == settings.export_sample_rate
    assert extras.get("bit_depth") == settings.export_bit_depth
    assert extras.get("format") == settings.export_format
    assert extras.get("prompt_hash") == hashlib.blake2s(
        b"soft synthwave",
        digest_size=8,
    ).hexdigest()
    assert extras.get("num_inference_steps") == settings.riffusion_num_inference_steps
    assert extras.get("scheduler") in (None, settings.riffusion_scheduler)
    assert extras.get("render_seconds") == pytest.approx(
        MIN_RENDER_SECONDS,
        rel=0.0,
        abs=0.5,
    )
    assert artifact.metadata.plan is not None
    assert artifact.metadata.prompt == "soft synthwave"


@pytest.mark.asyncio
async def test_riffusion_warmup_reports_status(tmp_path: Path) -> None:
    settings = Settings(
        artifact_root=tmp_path / "artifacts",
        config_dir=tmp_path / "config",
        riffusion_allow_inference=False,
    )
    settings.ensure_directories()
    service = RiffusionService(settings)

    status = await service.warmup()
    assert status.name == "riffusion"
    assert status.ready is False
    assert status.details.get("pipeline_loaded") is False


@pytest.mark.asyncio
async def test_placeholder_audio_varies_by_prompt(tmp_path: Path) -> None:
    settings = Settings(
        artifact_root=tmp_path / "artifacts",
        config_dir=tmp_path / "config",
        riffusion_allow_inference=False,
    )
    settings.ensure_directories()
    service = RiffusionService(settings)

    prompt_a = "minimal ambient drones"
    prompt_b = "energetic drum and bass"

    request_a = GenerationRequest(
        prompt=prompt_a,
        duration_seconds=2,
        model_id="riffusion-v1",
    )
    request_b = GenerationRequest(
        prompt=prompt_b,
        duration_seconds=2,
        model_id="riffusion-v1",
    )

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
    assert artifact_a.metadata.plan is not None
    assert artifact_b.metadata.plan is not None


@pytest.mark.asyncio
async def test_placeholder_audio_respects_seed(tmp_path: Path) -> None:
    settings = Settings(
        artifact_root=tmp_path / "artifacts",
        config_dir=tmp_path / "config",
        riffusion_allow_inference=False,
    )
    settings.ensure_directories()
    service = RiffusionService(settings)

    prompt = "noisy drones"
    request_seed_a = GenerationRequest(
        prompt=prompt,
        duration_seconds=3,
        model_id="riffusion-v1",
        seed=1234,
    )
    request_seed_b = GenerationRequest(
        prompt=prompt,
        duration_seconds=3,
        model_id="riffusion-v1",
        seed=5678,
    )

    artifact_a = await service.generate("job-seed-a", request_seed_a)
    artifact_b = await service.generate("job-seed-b", request_seed_b)

    data_a = Path(artifact_a.artifact_path).read_bytes()
    data_b = Path(artifact_b.artifact_path).read_bytes()
    assert data_a != data_b

    extras_a = artifact_a.metadata.extras
    extras_b = artifact_b.metadata.extras
    assert extras_a.get("seed") == 1234
    assert extras_b.get("seed") == 5678
    assert artifact_a.metadata.plan is not None
    assert artifact_b.metadata.plan is not None


@pytest.mark.asyncio
async def test_riffusion_service_pipeline_receives_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    monkeypatch.setattr(
        "timbre_worker.services.riffusion.torch",
        stub_torch,
        raising=False,
    )
    monkeypatch.setattr(
        "timbre_worker.services.riffusion.TORCH_IMPORT_ERROR",
        None,
        raising=False,
    )

    calls: list[Dict[str, Any]] = []

    class _DummyPipeline:
        def __init__(
            self,
            store: list[Dict[str, Any]],
            sample_rate: int = 44100,
        ):
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

    settings = Settings(
        artifact_root=tmp_path / "artifacts",
        config_dir=tmp_path / "config",
    )
    settings.ensure_directories()
    service = RiffusionService(settings, pipeline_loader=loader)

    request = GenerationRequest(
        prompt="vintage jazz trio",
        duration_seconds=3,
        model_id="riffusion-v1",
    )
    artifact = await service.generate("job-real", request)

    assert Path(artifact.artifact_path).exists()
    extras = artifact.metadata.extras
    assert extras.get("placeholder") is False
    assert artifact.metadata.plan is not None
    expected_hash = hashlib.blake2s(
        request.prompt.encode("utf-8"),
        digest_size=8,
    ).hexdigest()
    assert extras.get("prompt_hash") == expected_hash
    assert extras.get("sample_rate") == settings.export_sample_rate
    assert extras.get("bit_depth") == settings.export_bit_depth
    assert extras.get("format") == settings.export_format
    assert extras.get("guidance_scale") == DEFAULT_GUIDANCE_SCALE
    assert extras.get("num_inference_steps") == settings.riffusion_num_inference_steps
    assert extras.get("scheduler") in (
        None,
        settings.riffusion_scheduler,
        "DPMSolverMultistepScheduler",
    )
    assert extras.get("render_seconds") == pytest.approx(
        MIN_RENDER_SECONDS,
        rel=0.0,
        abs=0.5,
    )

    assert calls, "pipeline should have been invoked"
    first_call = calls[0]
    assert first_call.get("prompt") == request.prompt


@pytest.mark.asyncio
async def test_riffusion_inference_falls_back_to_placeholder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    monkeypatch.setattr(
        "timbre_worker.services.riffusion.torch",
        stub_torch,
        raising=False,
    )
    monkeypatch.setattr(
        "timbre_worker.services.riffusion.TORCH_IMPORT_ERROR",
        None,
        raising=False,
    )

    gradient = np.tile(np.linspace(0, 255, 512, dtype=np.uint8), (512, 1))
    image = np.stack([gradient, gradient, gradient], axis=-1)

    class _DummyPipeline:
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return types.SimpleNamespace(images=[image], sample_rate=44100)

    def loader(resolved_model_id: str) -> PipelineHandle:
        return PipelineHandle(pipeline=_DummyPipeline(), sample_rate=44100)

    monkeypatch.setattr(
        "timbre_worker.services.riffusion.RiffusionService._resolve_spectrogram_decoder",
        lambda self, _: None,
    )

    settings = Settings(
        artifact_root=tmp_path / "artifacts",
        config_dir=tmp_path / "config",
    )
    settings.ensure_directories()
    service = RiffusionService(settings, pipeline_loader=loader)

    request = GenerationRequest(
        prompt="ambient pads",
        duration_seconds=4,
        model_id="riffusion-v1",
    )

    section = CompositionSection(
        section_id="s00",
        role=SectionRole.MOTIF,
        label="Test",
        prompt=request.prompt,
        bars=4,
        target_seconds=4.0,
        energy=SectionEnergy.MEDIUM,
    )

    plan = CompositionPlan(
        version="test",
        tempo_bpm=90,
        time_signature="4/4",
        key="C major",
        total_bars=4,
        total_duration_seconds=4.0,
        sections=[section],
    )

    render = await service.render_section(
        request,
        section,
        plan=plan,
    )

    assert render.extras["placeholder"] is True
    assert "inference_error" in render.extras["placeholder_reason"]


def test_load_pipeline_uses_trust_remote_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: list[dict[str, Any]] = []

    class _StubTorch:
        float16 = np.float16
        float32 = np.float32

        class cuda:  # type: ignore[valid-type]
            @staticmethod
            def is_available() -> bool:
                return False

        class backends:  # type: ignore[valid-type]
            class mps:  # type: ignore[valid-type]
                @staticmethod
                def is_available() -> bool:
                    return False

    class _StubPipeline:
        def __init__(self) -> None:
            self.sample_rate = 44100

        def to(self, _device: str) -> "_StubPipeline":
            return self

        def set_progress_bar_config(
            self,
            disable: bool = True,
        ) -> None:  # pragma: no cover - noop
            return None

    def fake_from_pretrained(model_id: str, **kwargs: Any) -> _StubPipeline:
        captured_kwargs.append({"model_id": model_id, **kwargs})
        return _StubPipeline()

    class _StubDiffusionPipeline:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> _StubPipeline:
            return fake_from_pretrained(model_id, **kwargs)

    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.DiffusionPipeline = _StubDiffusionPipeline

    monkeypatch.setattr(
        "timbre_worker.services.riffusion.torch",
        _StubTorch,
        raising=False,
    )
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    settings = Settings(
        artifact_root=tmp_path / "artifacts",
        config_dir=tmp_path / "config",
    )
    settings.ensure_directories()
    service = RiffusionService(settings)
    handle = service._load_pipeline("riffusion/riffusion-model-v1")

    assert isinstance(handle, PipelineHandle)
    assert captured_kwargs, "from_pretrained should have been invoked"
    first_call = captured_kwargs[0]
    assert first_call.get("trust_remote_code") is True


def test_device_selection_prefers_mps_float32(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StubTorch:
        float16 = np.float16
        float32 = np.float32

        class cuda:  # type: ignore[valid-type]
            @staticmethod
            def is_available() -> bool:
                return False

        class backends:  # type: ignore[valid-type]
            class mps:  # type: ignore[valid-type]
                @staticmethod
                def is_available() -> bool:
                    return True

    monkeypatch.setattr(
        "timbre_worker.services.riffusion.torch",
        _StubTorch,
        raising=False,
    )
    monkeypatch.setattr(
        "timbre_worker.services.riffusion.TORCH_IMPORT_ERROR",
        None,
        raising=False,
    )

    settings = Settings(
        artifact_root=tmp_path / "artifacts",
        config_dir=tmp_path / "config",
    )
    settings.ensure_directories()
    service = RiffusionService(settings)

    assert service._device == "mps"
    assert service._dtype == np.float32


def test_device_override_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StubTorch:
        float16 = np.float16
        float32 = np.float32

        class cuda:  # type: ignore[valid-type]
            @staticmethod
            def is_available() -> bool:
                return True

        class backends:  # type: ignore[valid-type]
            class mps:  # type: ignore[valid-type]
                @staticmethod
                def is_available() -> bool:
                    return True

    monkeypatch.setattr(
        "timbre_worker.services.riffusion.torch",
        _StubTorch,
        raising=False,
    )
    monkeypatch.setattr(
        "timbre_worker.services.riffusion.TORCH_IMPORT_ERROR",
        None,
        raising=False,
    )

    settings = Settings(
        artifact_root=tmp_path / "artifacts",
        config_dir=tmp_path / "config",
        inference_device="cpu",
    )
    settings.ensure_directories()
    service = RiffusionService(settings)

    assert service._device == "cpu"
    assert service._dtype == np.float32
