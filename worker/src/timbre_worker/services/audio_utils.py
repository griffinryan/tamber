"""Audio utilities shared across backends."""

from __future__ import annotations

import math
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


def _as_two_dimensional(waveform: np.ndarray) -> tuple[np.ndarray, bool]:
    data = ensure_waveform_channels(waveform)
    if data.ndim == 1:
        return data.reshape(-1, 1), True
    return data, False


def trim_silence(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    threshold_db: float = -45.0,
    window_ms: float = 25.0,
    pre_roll_ms: float = 35.0,
    post_roll_ms: float = 60.0,
) -> np.ndarray:
    """Remove leading and trailing silence using an RMS gate."""

    data, was_mono = _as_two_dimensional(waveform)
    if data.size == 0:
        return waveform

    window_samples = max(1, int(sample_rate * window_ms / 1000.0))
    if window_samples >= data.shape[0]:
        return waveform

    power = np.mean(np.square(data), axis=1)
    kernel = np.ones(window_samples, dtype=np.float32) / window_samples
    smoothed = np.convolve(power, kernel, mode="same")
    smoothed = np.maximum(smoothed, 1e-9)
    db = 10.0 * np.log10(smoothed)
    mask = db > threshold_db
    if not np.any(mask):
        return waveform

    start_index = int(np.argmax(mask))
    end_index = int(len(mask) - np.argmax(mask[::-1]) - 1)

    pre_roll = int(sample_rate * pre_roll_ms / 1000.0)
    post_roll = int(sample_rate * post_roll_ms / 1000.0)

    start = max(0, start_index - pre_roll)
    end = min(data.shape[0], end_index + post_roll)
    trimmed = data[start:end]
    if was_mono:
        return trimmed.reshape(-1)
    return trimmed


def rms_level(waveform: np.ndarray) -> float:
    data, _was_mono = _as_two_dimensional(waveform)
    if data.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(data), axis=0)).mean())


def normalise_loudness(
    waveform: np.ndarray,
    target_rms: float = 0.18,
    *,
    max_gain: float = 4.0,
) -> np.ndarray:
    """Scale waveform RMS toward target with a gain ceiling."""

    data, was_mono = _as_two_dimensional(waveform)
    current = rms_level(data)
    if current <= 1e-6:
        return waveform
    gain = min(target_rms / current, max_gain)
    adjusted = data * gain
    if was_mono:
        return adjusted.reshape(-1)
    return adjusted


def soft_limiter(waveform: np.ndarray, *, threshold: float = 0.9) -> np.ndarray:
    """Apply a simple soft limiter using tanh compression above threshold."""

    if threshold <= 0.0:
        return np.clip(waveform, -1.0, 1.0)

    data = ensure_waveform_channels(waveform)
    over = np.abs(data) > threshold
    if np.any(over):
        exceeded = data[over]
        data = data.astype(np.float32, copy=True)
        data[over] = threshold * np.tanh(exceeded / threshold)
    return data


def crossfade_append(
    left: np.ndarray,
    right: np.ndarray,
    fade_samples: int,
) -> np.ndarray:
    """Append right waveform to left using a linear crossfade."""

    if fade_samples <= 1:
        return np.concatenate((left, right), axis=0)

    left_data, left_mono = _as_two_dimensional(left)
    right_data, right_mono = _as_two_dimensional(right)

    fade_samples = min(
        fade_samples,
        max(1, left_data.shape[0] - 1),
        max(1, right_data.shape[0] - 1),
    )
    if fade_samples <= 1:
        merged = np.vstack((left_data, right_data))
    else:
        fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)[:, None]
        fade_in = 1.0 - fade_out
        left_main = left_data[:-fade_samples]
        left_tail = left_data[-fade_samples:]
        right_head = right_data[:fade_samples]
        right_rest = right_data[fade_samples:]
        blended = left_tail * fade_out + right_head * fade_in
        merged = np.vstack((left_main, blended, right_rest))

    if left_mono and right_mono:
        return merged.reshape(-1)
    return merged


def fit_to_length(
    waveform: np.ndarray,
    target_samples: int,
    sample_rate: int,
    *,
    tempo_bpm: int,
) -> np.ndarray:
    """Trim or extend a waveform to the target length while preserving energy."""

    if target_samples <= 0:
        return np.zeros((0,), dtype=np.float32)

    trimmed = trim_silence(waveform, sample_rate)
    trimmed = normalise_loudness(trimmed)
    data, was_mono = _as_two_dimensional(trimmed)

    if data.shape[0] > target_samples:
        start = max(0, (data.shape[0] - target_samples) // 2)
        end = start + target_samples
        data = data[start:end]
    elif data.shape[0] < target_samples:
        seconds_per_beat = 60.0 / max(tempo_bpm, 1)
        loop_samples = int(max(seconds_per_beat * sample_rate, sample_rate * 0.5))
        loop_samples = min(loop_samples, data.shape[0])
        loop_samples = max(loop_samples, 1)
        while data.shape[0] < target_samples:
            remaining = target_samples - data.shape[0]
            segment = data[-loop_samples:]
            if remaining < segment.shape[0]:
                segment = segment[:remaining]
            fade = int(max(1, math.floor(sample_rate * 0.04)))
            data = crossfade_append(data, segment, fade)
            if data.shape[0] > target_samples:
                data = data[:target_samples]
                break

    if was_mono:
        return data.reshape(-1)
    return data


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
