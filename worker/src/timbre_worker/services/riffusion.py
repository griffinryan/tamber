from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from PIL import Image
from loguru import logger

from ..app.models import (
    CompositionPlan,
    CompositionSection,
    GenerationArtifact,
    GenerationMetadata,
    GenerationRequest,
    SectionEnergy,
    SectionRole,
)
from ..app.settings import Settings
from .audio_utils import ensure_waveform_channels, write_waveform
from .riffusion_spectrogram import SpectrogramImageDecoder, SpectrogramParams
from .types import SectionRender

try:  # pragma: no cover - optional dependency imports are validated at runtime
    import torch
except Exception as exc:  # noqa: BLE001
    torch = None  # type: ignore[assignment]
    TORCH_IMPORT_ERROR = exc
else:  # pragma: no cover
    TORCH_IMPORT_ERROR = None


DEFAULT_GUIDANCE_SCALE = 7.0
MIN_RENDER_SECONDS = 5.0
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

    def __init__(
        self,
        settings: Settings,
        *,
        pipeline_loader: Optional[Callable[[str], PipelineHandle]] = None,
    ):
        self._settings = settings
        self._artifact_root = settings.artifact_root
        self._default_model_id = settings.default_model_id
        self._pipelines: Dict[str, Optional[PipelineHandle]] = {}
        self._placeholder_reasons: Dict[str, str] = {}
        self._pipeline_lock = asyncio.Lock()
        self._allow_inference = settings.riffusion_allow_inference
        self._device, self._dtype = self._select_device_and_dtype()
        self._pipeline_loader = pipeline_loader
        self._spectrogram_decoder: Optional[SpectrogramImageDecoder] = None
        self._spectrogram_params = SpectrogramParams()
        self._spectrogram_error: Optional[str] = None
        self._spectrogram_mode: Optional[str] = None

    @property
    def default_model_id(self) -> str:
        return self._default_model_id

    async def warmup(self) -> None:
        await self._ensure_pipeline(self._default_model_id)
        self._resolve_spectrogram_decoder(44100)

    async def render_section(
        self,
        request: GenerationRequest,
        section: CompositionSection,
        *,
        plan: CompositionPlan,
        model_id: Optional[str] = None,
        render_seconds: Optional[float] = None,
    ) -> SectionRender:
        model_key = model_id or section.model_id or request.model_id or self._default_model_id
        pipeline_handle, placeholder_reason = await self._ensure_pipeline(model_key)

        duration_hint = (
            render_seconds if render_seconds is not None else float(section.target_seconds)
        )
        duration = max(MIN_RENDER_SECONDS, max(1.0, float(duration_hint)))
        guidance_scale = (
            request.cfg_scale if request.cfg_scale is not None else DEFAULT_GUIDANCE_SCALE
        )
        section_seed: Optional[int] = None
        if request.seed is not None:
            offset = section.seed_offset or 0
            section_seed = request.seed + offset

        dtype_str = (
            str(self._dtype) if self._dtype is not None and pipeline_handle is not None else None
        )

        used_placeholder = pipeline_handle is None
        device_token = self._device if pipeline_handle is not None else "placeholder"

        if pipeline_handle is None:
            waveform, sample_rate, extras = await asyncio.to_thread(
                self._placeholder_waveform,
                section.prompt,
                duration,
                placeholder_reason or "pipeline_unavailable",
                section_seed,
            )
        else:
            try:
                waveform, sample_rate, extras = await asyncio.to_thread(
                    self._run_inference,
                    pipeline_handle,
                    section.prompt,
                    duration,
                    guidance_scale,
                    section_seed,
                )
            except GenerationFailure as exc:
                logger.warning(
                    "Riffusion inference failed for section %s (%s); using placeholder audio",
                    section.section_id,
                    exc,
                )
                used_placeholder = True
                device_token = "placeholder"
                placeholder_reason = f"inference_error:{exc}"
                waveform, sample_rate, extras = await asyncio.to_thread(
                    self._placeholder_waveform,
                    section.prompt,
                    duration,
                    placeholder_reason,
                    section_seed,
                )
        actual_seconds = len(ensure_waveform_channels(waveform)) / float(sample_rate)

        extras.update(
            {
                "backend": "riffusion",
                "device": device_token,
                "guidance_scale": guidance_scale,
                "placeholder": used_placeholder,
                "section_id": section.section_id,
                "section_label": section.label,
                "section_role": section.role.value,
                "plan_version": plan.version,
                "render_seconds": actual_seconds,
                "target_seconds": float(section.target_seconds),
            }
        )
        if dtype_str is not None:
            extras["dtype"] = dtype_str
        if placeholder_reason:
            extras["placeholder_reason"] = placeholder_reason
        if section_seed is not None:
            extras["seed"] = section_seed
        decoder_mode = self._spectrogram_mode if not used_placeholder else "placeholder"
        if decoder_mode is not None:
            extras["spectrogram_decoder"] = decoder_mode
        extras["prompt_hash"] = self._prompt_hash(section.prompt)

        return SectionRender(waveform=waveform, sample_rate=sample_rate, extras=extras)

    async def generate(self, job_id: str, request: GenerationRequest) -> GenerationArtifact:
        plan = request.plan or self._fallback_plan(request)
        renders: list[SectionRender] = []
        for section in plan.sections:
            render = await self.render_section(
                request,
                section,
                plan=plan,
                model_id=request.model_id,
            )
            renders.append(render)

        waveform, sample_rate = self._combine_renders(renders)
        artifact_path = self._artifact_path(job_id, placeholder=False)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        write_waveform(artifact_path, waveform, sample_rate)

        placeholder_flags = [
            render.extras.get("placeholder", False) for render in renders
        ]
        placeholder_reasons = [
            render.extras.get("placeholder_reason")
            for render in renders
            if render.extras.get("placeholder_reason")
        ]
        guidance_values = [
            render.extras.get("guidance_scale")
            for render in renders
            if render.extras.get("guidance_scale") is not None
        ]
        prompt_hashes = [
            render.extras.get("prompt_hash")
            for render in renders
            if render.extras.get("prompt_hash") is not None
        ]

        extras: Dict[str, Any] = {
            "backend": "riffusion",
            "device": self._device if not all(placeholder_flags) else "placeholder",
            "sample_rate": sample_rate,
            "placeholder": any(placeholder_flags),
            "sections": [render.extras for render in renders],
        }
        seed_values = [
            render.extras.get("seed")
            for render in renders
            if render.extras.get("seed") is not None
        ]
        if seed_values:
            unique_seeds = {value for value in seed_values}
            extras["seed"] = (
                seed_values[0] if len(unique_seeds) == 1 else seed_values
            )
        render_lengths = [
            render.extras.get("render_seconds")
            for render in renders
            if render.extras.get("render_seconds") is not None
        ]
        if render_lengths:
            extras["render_seconds"] = (
                render_lengths[0] if len(render_lengths) == 1 else render_lengths
            )
        if self._dtype is not None and not all(placeholder_flags):
            extras["dtype"] = str(self._dtype)
        if guidance_values:
            extras["guidance_scale"] = guidance_values[0]
        if prompt_hashes:
            extras["prompt_hash"] = prompt_hashes[0]
            extras["prompt_hashes"] = prompt_hashes
        if placeholder_reasons:
            extras["placeholder_reason"] = (
                placeholder_reasons[0]
                if len(placeholder_reasons) == 1
                else placeholder_reasons
            )

        metadata = GenerationMetadata(
            prompt=request.prompt,
            seed=request.seed,
            model_id=request.model_id or self._default_model_id,
            duration_seconds=request.duration_seconds,
            extras=extras,
            plan=plan,
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

        if not self._allow_inference:
            reason = "inference_disabled"
            async with self._pipeline_lock:
                self._pipelines[model_id] = None
                self._placeholder_reasons[model_id] = reason
            logger.info("Riffusion inference disabled via settings; using placeholder audio")
            return None, reason

        if torch is None:
            reason = self._missing_dependency_reason()
            async with self._pipeline_lock:
                self._pipelines[model_id] = None
                if reason:
                    self._placeholder_reasons[model_id] = reason
            logger.warning("Riffusion pipeline unavailable: {}", reason)
            return None, reason

        try:
            if self._pipeline_loader is None:
                pipeline_handle = await asyncio.to_thread(self._load_pipeline, resolved)
            else:
                pipeline_handle = await asyncio.to_thread(self._pipeline_loader, resolved)
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
        dtype = self._dtype or torch.float32
        logger.info(
            "Loading Riffusion pipeline %s on %s with dtype %s",
            resolved_model_id,
            self._device,
            dtype,
        )
        try:
            pipeline = DiffusionPipeline.from_pretrained(
                resolved_model_id,
                torch_dtype=dtype,
                safety_checker=None,
                trust_remote_code=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise GenerationFailure(f"pipeline_load_failed:{exc}") from exc
        pipeline = pipeline.to(self._device)
        if hasattr(pipeline, "set_progress_bar_config"):
            pipeline.set_progress_bar_config(disable=True)

        sample_rate = getattr(pipeline, "sample_rate", None)
        if sample_rate is None and hasattr(pipeline, "audio_processor"):
            sample_rate = getattr(pipeline.audio_processor, "sample_rate", None)
        if sample_rate is None:
            sample_rate = 44100

        return PipelineHandle(pipeline=pipeline, sample_rate=sample_rate)

    def _run_inference(
        self,
        handle: PipelineHandle,
        prompt: str,
        duration_seconds: float,
        guidance_scale: float,
        seed: Optional[int],
    ) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        assert torch is not None

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._device).manual_seed(seed)

        audio_length = max(1, int(round(duration_seconds)))

        pipeline = handle.pipeline
        try:
            result = pipeline(
                prompt=prompt,
                num_inference_steps=50,
                guidance_scale=guidance_scale,
                audio_length_in_s=audio_length,
                generator=generator,
            )
        except TypeError:
            result = pipeline(
                prompt=prompt,
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

        extras: Dict[str, Any] = {
            "sample_rate": sample_rate,
            "guidance_scale": guidance_scale,
            "placeholder": False,
            "prompt_hash": self._prompt_hash(prompt),
        }
        if seed is not None:
            extras["seed"] = seed

        return self._prepare_waveform(waveform), sample_rate, extras

    def _placeholder_waveform(
        self,
        prompt: str,
        duration_seconds: float,
        reason: str,
        seed: Optional[int],
    ) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        sample_rate = 44100
        duration_seconds = max(1.0, min(duration_seconds, 30.0))
        total_samples = int(duration_seconds * sample_rate)
        t = np.linspace(0, duration_seconds, total_samples, endpoint=False, dtype=np.float32)

        seed_value = seed if seed is not None else abs(hash((prompt, duration_seconds))) % (2**32)
        base_freq_source = (hash(prompt), seed_value)
        base_freq = 220 + (abs(hash(base_freq_source)) % 360)
        waveform = 0.2 * np.sin(2 * np.pi * base_freq * t)
        waveform += 0.05 * np.sin(2 * np.pi * base_freq * 0.5 * t)
        rng_seed = seed_value % (2**32)
        waveform += 0.02 * np.random.default_rng(seed=rng_seed).standard_normal(size=total_samples)
        waveform = np.clip(waveform, -1.0, 1.0).astype(np.float32)
        extras = {
            "sample_rate": sample_rate,
            "guidance_scale": DEFAULT_GUIDANCE_SCALE,
            "placeholder_reason": reason,
            "prompt_hash": self._prompt_hash(prompt),
            "render_seconds": duration_seconds,
        }
        if seed is not None:
            extras["seed"] = seed

        return waveform, sample_rate, extras

    def _artifact_path(self, job_id: str, *, placeholder: bool) -> Path:
        filename = f"{job_id}{'-placeholder' if placeholder else ''}.wav"
        return self._artifact_root / filename

    def _combine_renders(self, renders: list[SectionRender]) -> Tuple[np.ndarray, int]:
        if not renders:
            raise GenerationFailure("no sections rendered")
        sample_rate = renders[0].sample_rate
        combined = ensure_waveform_channels(renders[0].waveform)
        for render in renders[1:]:
            if render.sample_rate != sample_rate:
                raise GenerationFailure("sample_rate_mismatch")
            combined = self._crossfade_append(
                combined,
                ensure_waveform_channels(render.waveform),
                sample_rate,
            )

        peak = float(np.max(np.abs(combined))) if combined.size else 0.0
        if peak > 0.99:
            combined = combined / peak * 0.99
        return combined.astype(np.float32), sample_rate

    def _crossfade_append(
        self,
        left: np.ndarray,
        right: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        crossfade_seconds = 0.25
        fade_samples = min(
            int(sample_rate * crossfade_seconds),
            max(1, left.shape[0] // 3),
            max(1, right.shape[0] // 3),
        )
        if fade_samples <= 1:
            return np.concatenate((left, right), axis=0)

        if left.ndim == 1:
            left = left[:, None]
        if right.ndim == 1:
            right = right[:, None]

        fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)[:, None]
        fade_in = 1.0 - fade_out

        left_main = left[:-fade_samples]
        left_tail = left[-fade_samples:]
        right_head = right[:fade_samples]
        right_rest = right[fade_samples:]

        blended = left_tail * fade_out + right_head * fade_in
        concatenated = np.vstack((left_main, blended, right_rest))
        return concatenated.reshape(-1, concatenated.shape[-1])

    def _prepare_waveform(self, waveform: np.ndarray) -> np.ndarray:
        if waveform.ndim == 3:
            waveform = waveform[0]
        if waveform.ndim == 1:
            return waveform
        if waveform.ndim == 2 and waveform.shape[0] < waveform.shape[1]:
            return waveform.T
        return waveform

    def _select_device_and_dtype(self) -> Tuple[str, Optional[Any]]:
        if torch is None:
            return "cpu", None

        override = (self._settings.inference_device or "").strip().lower()
        if override:
            if override == "cuda" and torch.cuda.is_available():  # pragma: no cover - hardware dependent
                return "cuda", torch.float16
            if (
                override == "mps"
                and getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()  # pragma: no cover
            ):
                return "mps", torch.float32
            if override == "cpu":
                return "cpu", torch.float32
            logger.warning(
                "Requested inference device %s unavailable; falling back to auto-selection",
                override,
            )

        if torch.cuda.is_available():  # pragma: no cover - depends on hardware
            return "cuda", torch.float16
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # pragma: no cover
            # Riffusion outputs distort on MPS with float16; prefer float32 even if slower.
            return "mps", torch.float32
        return "cpu", torch.float32

    def _missing_dependency_reason(self) -> str:
        if torch is None:
            guidance = "run `uv sync --project worker --extra inference` to install torch/diffusers"
            detail = TORCH_IMPORT_ERROR or "not installed"
            return f"torch_unavailable:{detail};{guidance}"
        return "unknown"

    def _resolve_spectrogram_decoder(
        self,
        expected_sample_rate: int,
    ) -> Optional[SpectrogramImageDecoder]:
        params = self._spectrogram_params
        if params.sample_rate != expected_sample_rate:
            params = replace(params, sample_rate=expected_sample_rate)
            self._spectrogram_params = params
            self._spectrogram_decoder = None

        if self._spectrogram_decoder is not None:
            return self._spectrogram_decoder

        try:
            decoder = SpectrogramImageDecoder(params, device="cpu")
        except RuntimeError as exc:
            self._spectrogram_error = str(exc)
            logger.warning("Spectrogram decoder unavailable: %s", exc)
            return None

        self._spectrogram_decoder = decoder
        self._spectrogram_error = None
        self._spectrogram_mode = getattr(decoder._inverse_mel, "decoder_mode", "unknown")
        logger.info(
            "Spectrogram decoder initialised (mode=%s, sample_rate=%s)",
            self._spectrogram_mode,
            params.sample_rate,
        )
        return decoder

    def _audio_from_images(
        self,
        result: Any,
        default_sample_rate: int,
    ) -> Tuple[np.ndarray, int]:
        images = getattr(result, "images", None)
        if not images:
            raise GenerationFailure("pipeline returned empty audio result")

        raw_image = images[0]
        if isinstance(raw_image, Image.Image):
            pil_image = raw_image
        else:
            array = np.asarray(raw_image)
            if array.dtype != np.uint8:
                upper_bound = float(np.max(array)) if array.size else 0.0
                if upper_bound <= 1.0:
                    array = np.clip(array, 0.0, 1.0) * 255.0
                array = np.clip(array, 0.0, 255.0).astype(np.uint8)
            if array.ndim == 2:
                pil_image = Image.fromarray(array, mode="L").convert("RGB")
            else:
                pil_image = Image.fromarray(array)

        decoder = self._resolve_spectrogram_decoder(default_sample_rate)
        if decoder is None:
            reason = self._spectrogram_error or "spectrogram_decoder_unavailable"
            raise GenerationFailure(reason)

        waveform, sample_rate = decoder.decode(pil_image)
        return waveform, sample_rate

    def _prompt_hash(self, prompt: str) -> str:
        digest = hashlib.blake2s(prompt.encode("utf-8"), digest_size=8)
        return digest.hexdigest()

    def _fallback_plan(self, request: GenerationRequest) -> CompositionPlan:
        duration = max(request.duration_seconds, 4)
        bars = max(4, duration // 2)
        tempo = int(np.clip(round(240 * bars / duration), 60, 120))
        section = CompositionSection(
            section_id="s00",
            role=SectionRole.MOTIF,
            label="Primary",
            prompt=request.prompt,
            bars=bars,
            target_seconds=float(request.duration_seconds),
            energy=SectionEnergy.MEDIUM,
            model_id=request.model_id,
            seed_offset=0,
            transition=None,
        )
        return CompositionPlan(
            version="fallback-v1",
            tempo_bpm=tempo,
            time_signature="4/4",
            key="C major",
            total_bars=bars,
            total_duration_seconds=float(request.duration_seconds),
            sections=[section],
        )
