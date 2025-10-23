"""Helpers to reconstruct audio waveforms from Riffusion spectrogram images.

This module adapts the open-source Riffusion project (MIT License) to provide a small
subset of its spectrogram decoding utilities in a form that returns NumPy waveforms.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

try:  # pragma: no cover - optional dependency
    import torchaudio
except Exception as exc:  # noqa: BLE001
    torchaudio = None  # type: ignore[assignment]
    TORCHAUDIO_IMPORT_ERROR = exc
else:  # pragma: no cover
    TORCHAUDIO_IMPORT_ERROR = None


ImageLike = Union[Image.Image, np.ndarray]


@dataclass(frozen=True)
class SpectrogramParams:
    """Parameter bundle describing the training-time spectrogram configuration."""

    stereo: bool = True
    sample_rate: int = 44100
    step_size_ms: int = 10
    window_duration_ms: int = 100
    padded_duration_ms: int = 400
    num_frequencies: int = 512
    min_frequency: int = 0
    max_frequency: int = 10_000
    mel_scale_norm: Optional[str] = None
    mel_scale_type: str = "htk"
    max_mel_iters: int = 200
    num_griffin_lim_iters: int = 48
    power_for_image: float = 0.3
    max_image_value: float = 30e6

    @property
    def n_fft(self) -> int:
        return int(self.padded_duration_ms / 1000 * self.sample_rate)

    @property
    def hop_length(self) -> int:
        return int(self.step_size_ms / 1000 * self.sample_rate)

    @property
    def win_length(self) -> int:
        return int(self.window_duration_ms / 1000 * self.sample_rate)


class SpectrogramImageDecoder:
    """Decode Riffusion spectrogram images into time-domain waveforms."""

    def __init__(self, params: SpectrogramParams, *, device: str = "cpu") -> None:
        if torchaudio is None:  # pragma: no cover - exercised in runtime checks
            raise RuntimeError(
                f"torchaudio_unavailable:{TORCHAUDIO_IMPORT_ERROR}"
            ) from TORCHAUDIO_IMPORT_ERROR

        self._params = params
        # Torchaudio's spectral ops are most stable on CPU and inexpensive at clip length.
        self._device = torch.device("cpu")
        self._dtype = torch.float32

        self._inverse_mel = self._build_inverse_mel(params)
        self._griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=params.n_fft,
            n_iter=params.num_griffin_lim_iters,
            win_length=params.win_length,
            hop_length=params.hop_length,
            power=1.0,
        ).to(self._device)

    @property
    def sample_rate(self) -> int:
        return self._params.sample_rate

    def decode(self, image: ImageLike) -> Tuple[np.ndarray, int]:
        """Convert a spectrogram image into a waveform and sample rate."""

        mel = self._spectrogram_from_image(image)
        waveform = self._waveform_from_mel(mel)
        return waveform, self._params.sample_rate

    def encode(self, waveform: np.ndarray, sample_rate: int) -> Image.Image:
        """Convert a waveform into the spectrogram image used by Riffusion."""

        if torchaudio is None:
            raise RuntimeError("torchaudio_unavailable")
        if waveform.ndim == 1:
            waveform = waveform[:, None]
        if waveform.shape[0] < waveform.shape[1]:
            waveform = waveform.T

        resampled = waveform
        if sample_rate != self._params.sample_rate:
            factor = self._params.sample_rate / sample_rate
            indices = np.arange(0, waveform.shape[0] * factor, factor)
            indices = indices[indices < waveform.shape[0]].astype(np.int32)
            resampled = waveform[indices]

        tensor = torch.as_tensor(resampled.T, dtype=self._dtype, device=self._device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)

        stft = torch.stft(
            tensor,
            n_fft=self._params.n_fft,
            hop_length=self._params.hop_length,
            win_length=self._params.win_length,
            window=torch.hann_window(self._params.win_length).to(self._device),
            center=True,
            return_complex=True,
        )
        magnitude = torch.abs(stft)
        mel = torch.nn.functional.linear(
            magnitude,
            torch.tensor(
                torchaudio.functional.create_fb_matrix(
                    self._params.n_fft // 2 + 1,
                    self._params.sample_rate,
                    self._params.num_frequencies,
                    self._params.min_frequency,
                    self._params.max_frequency,
                ),
                device=self._device,
                dtype=self._dtype,
            ),
        )
        mel = mel.cpu().numpy()

        mel = np.power(np.maximum(mel, 1e-6), self._params.power_for_image)
        mel = mel / mel.max(initial=1.0)
        mel = mel * self._params.max_image_value

        if self._params.stereo and mel.shape[0] >= 2:
            channels = np.stack([
                mel[0],
                mel[1],
                mel[1],
            ], axis=0)
        else:
            channels = np.stack([mel[0], mel[0], mel[0]], axis=0)

        image_array = np.clip(channels, 0.0, self._params.max_image_value)
        image_array = 255.0 - (image_array / self._params.max_image_value) * 255.0
        image_array = image_array.transpose(1, 2, 0)
        image_array = np.flip(image_array, axis=0)
        image = Image.fromarray(image_array.astype(np.uint8))
        return image

    def _build_inverse_mel(self, params: SpectrogramParams) -> torch.nn.Module:
        """Construct an InverseMelScale compatible with the installed torchaudio."""

        inverse_cls = torchaudio.transforms.InverseMelScale
        signature = inspect.signature(inverse_cls.__init__)
        kwargs: dict[str, object] = {
            "n_stft": params.n_fft // 2 + 1,
            "n_mels": params.num_frequencies,
            "sample_rate": params.sample_rate,
            "f_min": params.min_frequency,
            "f_max": params.max_frequency,
            "norm": params.mel_scale_norm,
            "mel_scale": params.mel_scale_type,
        }

        if "max_iter" in signature.parameters:
            kwargs.update(
                {
                    "max_iter": params.max_mel_iters,
                    "tolerance_loss": 1e-5,
                    "tolerance_change": 1e-8,
                    "sgdargs": None,
                }
            )
            mode = "legacy"
        elif "driver" in signature.parameters:
            kwargs["driver"] = "gels"
            mode = "gels"
        else:
            mode = "unknown"

        module = inverse_cls(**kwargs).to(self._device)
        module.decoder_mode = mode
        return module

    def _spectrogram_from_image(self, image: ImageLike) -> np.ndarray:
        if isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
            exif = pil_image.getexif()
            max_value = self._params.max_image_value
            if exif:
                # 11080 is SpectrogramParams.ExifTags.MAX_VALUE.value in upstream project.
                max_value = float(exif.get(11080, max_value))
            array = np.asarray(pil_image, dtype=np.float32)
        else:
            array = np.asarray(image, dtype=np.float32)
            max_value = self._params.max_image_value
            if array.ndim == 2:
                array = np.stack([array] * 3, axis=-1)

        array = np.flip(array, axis=0)  # Flip vertical axis back to spectrogram orientation.
        channels_first = array.transpose(2, 0, 1)

        if self._params.stereo and channels_first.shape[0] >= 3:
            data = channels_first[[1, 2], :, :]
        else:
            data = channels_first[0:1, :, :]

        data = 255.0 - data
        data = np.clip(data / 255.0, 0.0, 1.0)
        data = np.power(data, 1.0 / self._params.power_for_image)
        data = data * max_value
        return data.astype(np.float32)

    def _waveform_from_mel(self, mel: np.ndarray) -> np.ndarray:
        amplitudes = torch.as_tensor(mel, dtype=self._dtype, device=self._device)
        linear = self._inverse_mel(amplitudes)
        # Griffin-Lim expects (batch, freq, time); ensure batch dimension is present.
        if linear.ndim == 2:
            linear = linear.unsqueeze(0)
        waveform = self._griffin_lim(linear)
        waveform = waveform.cpu().numpy()  # (batch, samples)
        waveform = np.transpose(waveform, (1, 0))  # (samples, channels)

        peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
        if peak > 0.0:
            waveform = waveform / peak * 0.95

        return waveform.astype(np.float32)


def spectrogram_decoder_available() -> bool:
    """Return True if the torchaudio-backed decoder can be constructed."""

    return torchaudio is not None
