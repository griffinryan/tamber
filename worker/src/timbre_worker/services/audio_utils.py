"""Audio utilities shared across backends."""

from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
from loguru import logger

try:  # pragma: no cover
    import soundfile as sf
except Exception:  # noqa: BLE001
    sf = None  # type: ignore[assignment]

try:  # pragma: no cover
    import torchaudio
except Exception:  # noqa: BLE001
    torchaudio = None  # type: ignore[assignment]


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
        start = max(0, data.shape[0] - target_samples)
        data = data[start:]
    elif data.shape[0] < target_samples:
        loop_window = min(data.shape[0], max(1, int(sample_rate * 0.5)))
        if loop_window <= 1:
            pad = np.zeros(
                (target_samples - data.shape[0], data.shape[1]),
                dtype=np.float32,
            )
            data = np.vstack((data, pad))
        else:
            fade_samples = int(max(1, math.floor(sample_rate * 0.05)))
            fade_samples = min(fade_samples, loop_window - 1)
            loop_segment = data[-loop_window:].copy()
            extended = data.copy()
            while extended.shape[0] < target_samples:
                remaining = target_samples - extended.shape[0]
                chunk = loop_segment
                if remaining < loop_segment.shape[0]:
                    chunk = loop_segment[-remaining:].copy()
                extended = crossfade_append(
                    extended,
                    chunk,
                    max(1, fade_samples),
                )
            data = extended[:target_samples]

    if was_mono:
        return data.reshape(-1)
    return data


def resample_waveform(
    waveform: np.ndarray,
    src_rate: int,
    dst_rate: int,
) -> np.ndarray:
    """Resample waveform to a new sample rate using polyphase when available."""

    if src_rate <= 0 or dst_rate <= 0 or src_rate == dst_rate:
        return ensure_waveform_channels(waveform)

    data = ensure_waveform_channels(waveform)
    mono = False
    if data.ndim == 1:
        data = data[:, None]
        mono = True

    if torchaudio is not None:  # pragma: no cover - optional dependency
        try:
            import torch

            tensor = torch.as_tensor(data.T, dtype=torch.float32)
            resampled = torchaudio.functional.resample(  # type: ignore[attr-defined]
                tensor,
                orig_freq=src_rate,
                new_freq=dst_rate,
            )
            result = resampled.T.cpu().numpy().astype(np.float32)
            if mono:
                return result[:, 0]
            return result
        except Exception:  # noqa: BLE001
            pass

    scale = dst_rate / src_rate
    target_length = int(round(data.shape[0] * scale))
    if target_length <= 0:
        if mono:
            return np.zeros((0,), dtype=np.float32)
        return np.zeros((0, data.shape[1]), dtype=np.float32)

    x_old = np.linspace(0.0, 1.0, num=data.shape[0], endpoint=False, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, num=target_length, endpoint=False, dtype=np.float64)
    resampled = np.empty((target_length, data.shape[1]), dtype=np.float32)
    for channel in range(data.shape[1]):
        resampled[:, channel] = np.interp(x_new, x_old, data[:, channel])
    if mono:
        return resampled[:, 0].astype(np.float32)
    return resampled.astype(np.float32)


def tilt_highs(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    gain_db: float = 2.5,
    exponent: float = 1.2,
) -> np.ndarray:
    """Apply a gentle high-frequency tilt to restore brilliance."""

    if gain_db <= 0.0 or sample_rate <= 0:
        return ensure_waveform_channels(waveform)

    data = ensure_waveform_channels(waveform).astype(np.float32)
    mono = False
    if data.ndim == 1:
        data = data[:, None]
        mono = True
    if data.shape[0] < 4:
        return data[:, 0] if mono else data

    spectrum = np.fft.rfft(data, axis=0)
    freqs = np.fft.rfftfreq(data.shape[0], d=1.0 / sample_rate)
    if freqs.size == 0:
        return data[:, 0] if mono else data

    nyquist = float(freqs[-1]) if freqs[-1] > 0.0 else float(sample_rate) / 2.0
    ratio = np.clip(freqs / max(nyquist, 1.0), 0.0, 1.0)
    high_gain = 10.0 ** (gain_db / 20.0)
    curve = 1.0 + (high_gain - 1.0) * np.power(ratio, exponent, where=ratio > 0.0)
    curve = curve.astype(np.float32)
    if spectrum.ndim == 1:
        spectrum *= curve
    else:
        spectrum *= curve[:, None]
    boosted = np.fft.irfft(spectrum, n=data.shape[0], axis=0)
    boosted = boosted.real.astype(np.float32)

    peak = float(np.max(np.abs(boosted))) if boosted.size else 0.0
    if peak > 1.0:
        boosted = boosted / peak
    if mono:
        return boosted[:, 0]
    return boosted


def _bit_depth_to_int(bit_depth: str) -> int:
    mapping = {
        "pcm16": 16,
        "pcm24": 24,
        "pcm32": 32,
        "float32": 32,
    }
    return mapping.get(bit_depth.lower(), 16)


def _soundfile_subtype(bit_depth: str) -> str:
    mapping = {
        "pcm16": "PCM_16",
        "pcm24": "PCM_24",
        "pcm32": "PCM_32",
        "float32": "FLOAT",
    }
    return mapping.get(bit_depth.lower(), "PCM_16")


def _apply_tpdf_dither(data: np.ndarray, bit_depth: int) -> np.ndarray:
    if bit_depth <= 0:
        return data
    step = 1.0 / float(2 ** (bit_depth - 1))
    rng = np.random.default_rng()
    noise = (rng.random(data.shape, dtype=np.float32) - rng.random(data.shape, dtype=np.float32)) * step
    return np.clip(data + noise, -1.0, 1.0).astype(np.float32)


def write_waveform(
    path: Path,
    waveform: np.ndarray,
    sample_rate: int,
    *,
    bit_depth: str = "pcm24",
    audio_format: str = "wav",
    dither: bool = True,
) -> None:
    """Persist waveform to disk, preferring soundfile when available."""

    data = ensure_waveform_channels(waveform)

    format_token = audio_format.lower()
    depth_token = bit_depth.lower()

    if sf is not None and format_token == "wav":  # pragma: no cover - depends on optional lib
        subtype = _soundfile_subtype(depth_token)
        export = data.astype(np.float32)
        if dither and subtype != "FLOAT":
            export = _apply_tpdf_dither(export, _bit_depth_to_int(depth_token))
        sf.write(path, export, sample_rate, subtype=subtype)
        return

    if format_token != "wav":
        logger.warning(
            "Falling back to WAV output for %s; soundfile dependency required for other formats",
            audio_format,
        )

    target_bit_depth = _bit_depth_to_int(depth_token)
    if target_bit_depth != 16:
        logger.warning(
            "PCM %s export requires soundfile; emitting 16-bit WAV instead",
            bit_depth,
        )
        target_bit_depth = 16

    export = data.astype(np.float32, copy=True)
    if dither:
        export = _apply_tpdf_dither(export, target_bit_depth)
    scale = float(2 ** (target_bit_depth - 1) - 1)
    pcm = np.clip(export, -1.0, 1.0)
    pcm = (pcm * scale).astype(np.int32)
    if target_bit_depth == 16:
        frames = pcm.astype(np.int16)
    else:
        frames = pcm.astype(np.int32)

    if frames.ndim == 1:
        channels = 1
    else:
        channels = frames.shape[1]

    sampwidth = 2 if target_bit_depth == 16 else 4
    with wave.open(str(path), "wb") as wav_file:  # type: ignore[attr-defined]
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames.tobytes())
