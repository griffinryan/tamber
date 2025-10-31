from pathlib import Path

import numpy as np
import pytest

from timbre_worker.app.settings import Settings
from timbre_worker.services.musicgen import MusicGenService


def test_build_generation_kwargs_handles_two_step_flag() -> None:
    service = MusicGenService()

    kwargs, applied = service._build_generation_kwargs(  # type: ignore[protected-access]
        max_new_tokens=64,
        top_k=128,
        top_p=0.85,
        temperature=1.1,
        cfg_coef=3.0,
        two_step_cfg=True,
    )

    assert "do_classifier_free_guidance" not in kwargs
    assert kwargs["guidance_scale"] == 3.0
    assert applied is True


def test_placeholder_waveform_reports_two_step_support() -> None:
    service = MusicGenService()

    waveform, sample_rate, extras = service._placeholder_waveform(  # type: ignore[protected-access]
        prompt="ambient arpeggios under moonlight",
        duration_seconds=2.0,
        reason="musicgen_unavailable",
        seed=42,
        top_k=64,
        top_p=0.9,
        temperature=1.0,
        cfg_coef=2.0,
        two_step_cfg=True,
    )

    assert isinstance(waveform, np.ndarray)
    assert sample_rate == 32000
    assert extras["two_step_cfg"] is True
    assert extras["two_step_cfg_applied"] is True
    assert extras["two_step_cfg_supported"] is True


@pytest.mark.asyncio
async def test_musicgen_warmup_reports_status(monkeypatch: pytest.MonkeyPatch) -> None:
    service = MusicGenService()

    async def _fake_ensure(model_id: str):  # type: ignore[override]
        return None, "load_skipped"

    monkeypatch.setattr(service, "_ensure_model", _fake_ensure)

    status = await service.warmup()
    assert status.name == "musicgen"
    assert status.ready is False
    assert status.error == "load_skipped"


def test_device_override_falls_back_to_cpu(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "cfg"
    artifacts_dir = tmp_path / "artifacts"
    settings = Settings(
        config_dir=cfg_dir,
        artifact_root=artifacts_dir,
        inference_device="cpu",
    )
    service = MusicGenService(settings=settings)
    assert service._device == "cpu"  # type: ignore[attr-defined]
