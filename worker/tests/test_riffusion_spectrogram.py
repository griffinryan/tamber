from __future__ import annotations

import types

import torch

from timbre_worker.services import riffusion_spectrogram as rs


def test_decoder_handles_driver_based_inverse_mel(monkeypatch):
    calls: dict[str, object] = {}

    class _StubInverse:
        def __init__(
            self,
            *,
            n_stft: int,
            n_mels: int = 128,
            sample_rate: int = 16000,
            f_min: float = 0.0,
            f_max: float | None = None,
            norm: str | None = None,
            mel_scale: str = "htk",
            driver: str = "gels",
        ) -> None:
            calls["kwargs"] = {
                "n_stft": n_stft,
                "n_mels": n_mels,
                "sample_rate": sample_rate,
                "f_min": f_min,
                "f_max": f_max,
                "norm": norm,
                "mel_scale": mel_scale,
                "driver": driver,
            }

        def to(self, device: torch.device) -> "_StubInverse":
            calls["device"] = device
            return self

    class _StubGriffin:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def to(self, device: torch.device) -> "_StubGriffin":
            return self

    monkeypatch.setattr(
        rs.torchaudio,
        "transforms",
        types.SimpleNamespace(InverseMelScale=_StubInverse, GriffinLim=_StubGriffin),
        raising=True,
    )

    decoder = rs.SpectrogramImageDecoder(rs.SpectrogramParams())

    assert decoder._inverse_mel.decoder_mode == "gels"
    assert "kwargs" in calls
    assert "device" in calls and str(calls["device"]) == "cpu"
    assert "driver" in calls["kwargs"]
    assert "max_iter" not in calls["kwargs"]
