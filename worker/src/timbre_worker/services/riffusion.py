from __future__ import annotations

import asyncio
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from loguru import logger

from ..app.models import GenerationArtifact, GenerationMetadata, GenerationRequest
from ..app.settings import Settings

try:  # pragma: no cover - optional dependency imports are validated at runtime
    import torch
except Exception as exc:  # noqa: BLE001
    torch = None  # type: ignore[assignment]
    TORCH_IMPORT_ERROR = exc
else:  # pragma: no cover
    TORCH_IMPORT_ERROR = None

try:  # pragma: no cover
    from riffusion.spectrogram_image_converter import SpectrogramImageConverter
except Exception as exc:  # noqa: BLE001
    SpectrogramImageConverter = None  # type: ignore[misc,assignment]
    SPECTROGRAM_IMPORT_ERROR = exc
else:  # pragma: no cover
    SPECTROGRAM_IMPORT_ERROR = None

try:  # pragma: no cover
    import soundfile as sf
except Exception as exc:  # noqa: BLE001
    sf = None  # type: ignore[assignment]
    SOUND_FILE_IMPORT_ERROR = exc
else:  # pragma: no cover
    SOUND_FILE_IMPORT_ERROR = None


DEFAULT_GUIDANCE_SCALE = 7.0
MODEL_REGISTRY = {
    "riffusion-v1": "riffusion/riffusion-model-v1",
}


class GenerationFailure(Exception):
    """Expected failure during audio generation."""


@dataclass
class PipelineHandle:
    pipeline: Any
    sample_rate: int


class RiffusionService:
    """Service responsible for loading Riffusion pipelines and generating audio."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._artifact_root = settings.artifact_root
        self._default_model_id = settings.default_model_id
        self._pipelines: Dict[str, Optional[PipelineHandle]] = {}
        self._placeholder_reasons: Dict[str, str] = {}
        self._pipeline_lock = asyncio.Lock()
        self._device = self._select_device()
        self._spectrogram_converter: Optional[SpectrogramImageConverter] = (
            SpectrogramImageConverter() if SpectrogramImageConverter is not None else None
        )

    @property
    def default_model_id(self) -> str:
        return self._default_model_id

    async def warmup(self) -> None:
        await self._ensure_pipeline(self._default_model_id)

    async def generate(self, job_id: str, request: GenerationRequest) -> GenerationArtifact:
        model_key = request.model_id or self._default_model_id
        pipeline_handle, placeholder_reason = await self._ensure_pipeline(model_key)

        artifact_path: Path
        extras: Dict[str, Any]

        if pipeline_handle is None:
            artifact_path, extras = await asyncio.to_thread(
                self._write_placeholder_audio,
                job_id,
                request.prompt,
                request.duration_seconds,
                placeholder_reason or "pipeline_unavailable",
            )
        else:
            artifact_path, extras = await asyncio.to_thread(
                self._run_pipeline,
                pipeline_handle,
                job_id,
                request,
            )

        metadata = GenerationMetadata(
            prompt=request.prompt,
            seed=request.seed,
            model_id=model_key,
            duration_seconds=request.duration_seconds,
            extras=extras,
        )

        return GenerationArtifact(
            job_id=job_id,
            artifact_path=str(artifact_path),
            metadata=metadata,
        )

    async def _ensure_pipeline(
        self, model_id: str
    ) -> Tuple[Optional[PipelineHandle], Optional[str]]:
        async with self._pipeline_lock:
            if model_id in self._pipelines:
                return self._pipelines[model_id], self._placeholder_reasons.get(model_id)

        resolved = MODEL_REGISTRY.get(model_id, model_id)

        if torch is None:
            reason = self._missing_dependency_reason()
            async with self._pipeline_lock:
                self._pipelines[model_id] = None
                if reason:
                    self._placeholder_reasons[model_id] = reason
            logger.warning("Riffusion pipeline unavailable: {}", reason)
            return None, reason

        try:
            pipeline_handle = await asyncio.to_thread(self._load_pipeline, resolved)
        except GenerationFailure as exc:
            logger.warning("Pipeline prerequisites missing: %s", exc)
            async with self._pipeline_lock:
                self._pipelines[model_id] = None
                self._placeholder_reasons[model_id] = str(exc)
            return None, str(exc)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load pipeline %s", resolved)
            async with self._pipeline_lock:
                self._pipelines[model_id] = None
                self._placeholder_reasons[model_id] = f"load_error:{exc.__class__.__name__}"
            return None, f"load_error:{exc.__class__.__name__}"

        async with self._pipeline_lock:
            self._pipelines[model_id] = pipeline_handle
            self._placeholder_reasons.pop(model_id, None)
        return pipeline_handle, None

    def _load_pipeline(self, resolved_model_id: str) -> PipelineHandle:
        assert torch is not None  # For type checkers
        try:
            from diffusers import DiffusionPipeline
        except Exception as exc:  # noqa: BLE001
            raise GenerationFailure(f"diffusers_unavailable:{exc}") from exc

        dtype = torch.float16 if self._device in {"cuda", "mps"} else torch.float32
        logger.info(
            "Loading Riffusion pipeline %s on %s with dtype %s",
            resolved_model_id,
            self._device,
            dtype,
        )
        try:
            pipeline = DiffusionPipeline.from_pretrained(
                resolved_model_id,
                custom_pipeline="riffusion",
                torch_dtype=dtype,
                safety_checker=None,
            )
        except Exception:
            pipeline = DiffusionPipeline.from_pretrained(
                resolved_model_id,
                torch_dtype=dtype,
                safety_checker=None,
            )
        pipeline = pipeline.to(self._device)
        if hasattr(pipeline, "set_progress_bar_config"):
            pipeline.set_progress_bar_config(disable=True)

        sample_rate = getattr(pipeline, "sample_rate", None)
        if sample_rate is None and hasattr(pipeline, "audio_processor"):
            sample_rate = getattr(pipeline.audio_processor, "sample_rate", None)
        if sample_rate is None:
            sample_rate = 44100

        return PipelineHandle(pipeline=pipeline, sample_rate=sample_rate)

    def _run_pipeline(
        self,
        handle: PipelineHandle,
        job_id: str,
        request: GenerationRequest,
    ) -> Tuple[Path, Dict[str, Any]]:
        assert torch is not None

        guidance_scale = request.cfg_scale if request.cfg_scale is not None else DEFAULT_GUIDANCE_SCALE
        generator = None
        if request.seed is not None:
            generator = torch.Generator(device=self._device).manual_seed(request.seed)

        pipeline = handle.pipeline
        try:
            result = pipeline(
                prompt=request.prompt,
                num_inference_steps=50,
                guidance_scale=guidance_scale,
                audio_length_in_s=request.duration_seconds,
                generator=generator,
            )
        except TypeError:
            result = pipeline(
                prompt=request.prompt,
                num_inference_steps=50,
                guidance_scale=guidance_scale,
                generator=generator,
            )

        audio_list = getattr(result, "audios", None)
        sample_rate = getattr(result, "sample_rate", None) or handle.sample_rate

        if audio_list:
            waveform = np.asarray(audio_list[0], dtype=np.float32)
        else:
            waveform, sample_rate = self._audio_from_images(result, sample_rate)

        waveform = self._prepare_waveform(waveform)

        artifact_path = self._artifact_path(job_id, placeholder=False)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_audio_file(artifact_path, waveform, sample_rate)

        extras: Dict[str, Any] = {
            "backend": "riffusion",
            "device": self._device,
            "sample_rate": sample_rate,
            "guidance_scale": guidance_scale,
            "placeholder": False,
        }
        if request.seed is not None:
            extras["seed"] = request.seed

        return artifact_path, extras

    def _write_placeholder_audio(
        self,
        job_id: str,
        prompt: str,
        duration_seconds: int,
        reason: str,
    ) -> Tuple[Path, Dict[str, Any]]:
        sample_rate = 44100
        duration_seconds = max(1, min(duration_seconds, 30))
        total_samples = duration_seconds * sample_rate
        t = np.linspace(0, duration_seconds, total_samples, endpoint=False, dtype=np.float32)

        base_freq = 220 + (abs(hash(prompt)) % 360)
        waveform = 0.2 * np.sin(2 * np.pi * base_freq * t)
        waveform += 0.05 * np.sin(2 * np.pi * base_freq * 0.5 * t)
        waveform += 0.02 * np.random.default_rng(seed=abs(hash(prompt)) % (2**32)).standard_normal(
            size=total_samples
        )
        waveform = np.clip(waveform, -1.0, 1.0).astype(np.float32)

        artifact_path = self._artifact_path(job_id, placeholder=True)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_audio_file(artifact_path, waveform, sample_rate)

        extras = {
            "backend": "riffusion",
            "device": "placeholder",
            "sample_rate": sample_rate,
            "guidance_scale": DEFAULT_GUIDANCE_SCALE,
            "placeholder": True,
            "placeholder_reason": reason,
        }

        return artifact_path, extras

    def _write_audio_file(self, path: Path, waveform: np.ndarray, sample_rate: int) -> None:
        waveform = np.clip(waveform, -1.0, 1.0)
        if waveform.ndim == 1:
            data = waveform
        elif waveform.ndim == 2 and waveform.shape[0] in (1, 2):
            data = waveform.T
        else:
            data = waveform.squeeze()

        if sf is not None:
            sf.write(path, data, sample_rate, subtype="PCM_16")
            return

        pcm = (data * 32767).astype(np.int16)
        channels = 1 if pcm.ndim == 1 else pcm.shape[1]
        with wave.open(str(path), "wb") as wav_file:  # type: ignore[attr-defined]
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm.tobytes())

    def _artifact_path(self, job_id: str, *, placeholder: bool) -> Path:
        filename = f"{job_id}{'-placeholder' if placeholder else ''}.wav"
        return self._artifact_root / filename

    def _prepare_waveform(self, waveform: np.ndarray) -> np.ndarray:
        if waveform.ndim == 3:
            waveform = waveform[0]
        if waveform.ndim == 1:
            return waveform
        if waveform.ndim == 2 and waveform.shape[0] < waveform.shape[1]:
            return waveform.T
        return waveform

    def _select_device(self) -> str:
        if torch is None:
            return "cpu"
        if torch.cuda.is_available():  # pragma: no cover - depends on hardware
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # pragma: no cover
            return "mps"
        return "cpu"

    def _missing_dependency_reason(self) -> str:
        if torch is None:
            return f"torch_unavailable:{TORCH_IMPORT_ERROR}"
        if SpectrogramImageConverter is None:
            return f"spectrogram_converter_unavailable:{SPECTROGRAM_IMPORT_ERROR}"
        return "unknown"

    def _audio_from_images(
        self,
        result: Any,
        default_sample_rate: int,
    ) -> Tuple[np.ndarray, int]:
        if self._spectrogram_converter is None:
            raise GenerationFailure(
                f"spectrogram_converter_unavailable:{SPECTROGRAM_IMPORT_ERROR}"
            )

        images = getattr(result, "images", None)
        if not images:
            raise GenerationFailure("pipeline returned empty audio result")

        image = images[0]
        segment = self._spectrogram_converter.audio_from_spectrogram_image(image)
        waveform = np.asarray(segment.samples, dtype=np.float32)
        sample_rate = getattr(segment, "sample_rate", default_sample_rate)
        return waveform, sample_rate
