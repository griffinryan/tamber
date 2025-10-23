"""Audio utilities shared across backends."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

try:  # pragma: no cover
    import soundfile as sf
except Exception:  # noqa: BLE001
    sf = None  # type: ignore[assignment]


def ensure_waveform_channels(waveform: np.ndarray) -> np.ndarray:
    """Normalise waveform orientation to shape (samples, channels)."""

    data = np.clip(waveform, -1.0, 1.0)
    if data.ndim == 1:
        return data.astype(np.float32)
    if data.ndim == 2 and data.shape[0] in (1, 2):
        return data.T.astype(np.float32)
    if data.ndim == 3:
        return data[0].T.astype(np.float32)
    return data.squeeze().astype(np.float32)


def write_waveform(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    """Persist waveform to disk, preferring soundfile when available."""

    data = ensure_waveform_channels(waveform)

    if sf is not None:
        sf.write(path, data, sample_rate, subtype="PCM_16")
        return

    pcm = (data * 32767).astype(np.int16)
    if pcm.ndim == 1:
        channels = 1
        frames = pcm
    else:
        channels = pcm.shape[1]
        frames = pcm

    with wave.open(str(path), "wb") as wav_file:  # type: ignore[attr-defined]
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames.tobytes())
