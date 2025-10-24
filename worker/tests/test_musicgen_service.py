import numpy as np

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
    assert applied is False
    assert service._two_step_warning_emitted is True  # type: ignore[protected-access]


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
    assert extras["two_step_cfg_applied"] is False
    assert extras["two_step_cfg_supported"] is False
