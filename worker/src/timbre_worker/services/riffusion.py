from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np
from loguru import logger

try:  # pragma: no cover - optional dependency import
    from PIL import Image
except Exception as exc:  # noqa: BLE001
    Image = None  # type: ignore[assignment]
    PIL_IMPORT_ERROR = exc
else:  # pragma: no cover
    PIL_IMPORT_ERROR = None

if TYPE_CHECKING:  # pragma: no cover - typing aid only
    from PIL import Image as PILImage  # noqa: F401
else:
    PILImage = Any

from ..app.models import (
    CompositionPlan,
    CompositionSection,
    GenerationArtifact,
    GenerationMetadata,
    GenerationRequest,
    SectionEnergy,
    SectionRole,
    ThemeDescriptor,
)
from ..app.settings import Settings
from .audio_utils import (
    ensure_waveform_channels,
    normalise_loudness,
    resample_waveform,
    soft_limiter,
    tilt_highs,
    write_waveform,
)
from .riffusion_spectrogram import SpectrogramImageDecoder, SpectrogramParams
from .types import SectionRender

try:  # pragma: no cover - optional dependency imports are validated at runtime
    import torch
except Exception as exc:  # noqa: BLE001
    torch = None  # type: ignore[assignment]
    TORCH_IMPORT_ERROR = exc
else:  # pragma: no cover
    TORCH_IMPORT_ERROR = None


DEFAULT_GUIDANCE_SCALE = 8.5
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
        self._default_model_id = settings.riffusion_default_model_id
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
        self._spectrogram_warned = False
        self._default_guidance = settings.riffusion_guidance_scale
        self._default_steps = settings.riffusion_num_inference_steps
        self._default_scheduler = settings.riffusion_scheduler
        self._enable_phase_refinement = settings.riffusion_enable_phase_refinement

    @property
    def default_model_id(self) -> str:
        return self._default_model_id

    async def warmup(self) -> None:
        await self._ensure_pipeline(self._default_model_id)

    async def render_section(
        self,
        request: GenerationRequest,
        section: CompositionSection,
        *,
        plan: CompositionPlan,
        model_id: Optional[str] = None,
        render_seconds: Optional[float] = None,
        theme: ThemeDescriptor | None = None,
        previous_render: SectionRender | None = None,
        motif_seed: SectionRender | None = None,
    ) -> SectionRender:
        model_key = model_id or section.model_id or request.model_id or self._default_model_id
        pipeline_handle, placeholder_reason = await self._ensure_pipeline(model_key)

        duration_hint = (
            render_seconds if render_seconds is not None else float(section.target_seconds)
        )
        duration = max(MIN_RENDER_SECONDS, max(1.0, float(duration_hint)))
        guidance_source = (
            request.riffusion_guidance_scale
            if request.riffusion_guidance_scale is not None
            else (request.cfg_scale if request.cfg_scale is not None else self._default_guidance)
        )
        guidance_scale = float(
            guidance_source if guidance_source is not None else DEFAULT_GUIDANCE_SCALE
        )
        num_inference_steps = int(
            request.riffusion_num_inference_steps
            if request.riffusion_num_inference_steps is not None
            else self._default_steps
        )
        num_inference_steps = max(1, num_inference_steps)
        scheduler_name = (
            request.riffusion_scheduler
            or request.scheduler
            or self._default_scheduler
            or None
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

        prompt_text = self._compose_prompt(
            section.prompt,
            theme=theme,
            previous=previous_render,
        )

        init_image = None
        init_strength = 0.55
        if previous_render is not None:
            init_image = self._prepare_init_image(previous_render)
            if init_image is not None:
                extras_hint = previous_render.extras.copy()
            else:
                extras_hint = None
        else:
            extras_hint = None

        if pipeline_handle is None:
            waveform, sample_rate, extras = await asyncio.to_thread(
                self._placeholder_waveform,
                prompt_text,
                duration,
                placeholder_reason or "pipeline_unavailable",
                section_seed,
                guidance_scale,
                num_inference_steps,
                scheduler_name,
            )
        else:
            try:
                waveform, sample_rate, extras = await asyncio.to_thread(
                    self._run_inference,
                    pipeline_handle,
                    prompt_text,
                    duration,
                    guidance_scale,
                    section_seed,
                    init_image,
                    init_strength,
                    num_inference_steps,
                    scheduler_name,
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
                    prompt_text,
                    duration,
                    placeholder_reason,
                    section_seed,
                    guidance_scale,
                    num_inference_steps,
                    scheduler_name,
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
                "num_inference_steps": num_inference_steps,
            }
        )
        if dtype_str is not None:
            extras["dtype"] = dtype_str
        if placeholder_reason:
            extras["placeholder_reason"] = placeholder_reason
        if scheduler_name and extras.get("scheduler") is None:
            extras["scheduler"] = scheduler_name
        if section_seed is not None:
            extras["seed"] = section_seed
        if theme is not None:
            extras["theme"] = theme.model_dump()
        if init_image is not None:
            extras.setdefault("continuation", {})
            extras["continuation"].update(
                {
                    "from_section": previous_render.extras.get("section_id")
                    if previous_render is not None
                    else None,
                    "strength": init_strength,
                }
            )
        if extras_hint is not None:
            extras.setdefault("continuation", {})
            extras["continuation"].setdefault(
                "source_metadata",
                extras_hint,
            )
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
        waveform = normalise_loudness(waveform)
        waveform = tilt_highs(waveform, sample_rate)
        waveform = soft_limiter(waveform, threshold=0.98)

        export_rate = self._settings.export_sample_rate
        if export_rate and sample_rate != export_rate:
            waveform = resample_waveform(waveform, sample_rate, export_rate)
            sample_rate = export_rate
            waveform = soft_limiter(waveform, threshold=0.98)

        artifact_path = self._artifact_path(job_id, placeholder=False)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        write_waveform(
            artifact_path,
            waveform,
            sample_rate,
            bit_depth=self._settings.export_bit_depth,
            audio_format=self._settings.export_format,
        )

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
        step_counts = [
            render.extras.get("num_inference_steps")
            for render in renders
            if render.extras.get("num_inference_steps") is not None
        ]

        extras: Dict[str, Any] = {
            "backend": "riffusion",
            "device": self._device if not all(placeholder_flags) else "placeholder",
            "sample_rate": sample_rate,
            "placeholder": any(placeholder_flags),
            "sections": [render.extras for render in renders],
        }
        extras["bit_depth"] = self._settings.export_bit_depth
        extras["format"] = self._settings.export_format
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
        if step_counts:
            unique_steps = {value for value in step_counts}
            extras["num_inference_steps"] = (
                step_counts[0] if len(unique_steps) == 1 else step_counts
            )
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

    def _configure_scheduler(
        self,
        pipeline: Any,
        scheduler_name: Optional[str],
    ) -> Optional[str]:
        if scheduler_name is None or not scheduler_name.strip():
            return None
        if not hasattr(pipeline, "scheduler"):
            return None
        normalized = scheduler_name.strip().lower().replace("-", "_")
        current = pipeline.scheduler.__class__.__name__.lower()
        if normalized == current:
            return pipeline.scheduler.__class__.__name__

        try:
            from diffusers import (  # type: ignore
                DDIMScheduler,
                DPMSolverMultistepScheduler,
                EulerAncestralDiscreteScheduler,
                EulerDiscreteScheduler,
                KDPM2AncestralDiscreteScheduler,
                PNDMScheduler,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Scheduler import failed for %s: %s", scheduler_name, exc)
            return None

        config = getattr(pipeline.scheduler, "config", None)
        if config is None:
            return None

        scheduler = None
        try:
            if normalized in {"dpmpp_2m", "dpmpp_2m_sde", "dpm++_2m", "dpmsolver++"}:
                scheduler = DPMSolverMultistepScheduler.from_config(config)
                setattr(scheduler, "algorithm_type", "dpmsolver++")
                if hasattr(scheduler, "use_karras_sigmas"):
                    setattr(scheduler, "use_karras_sigmas", False)
            elif normalized in {
                "dpmpp_2m_karras",
                "dpm++_2m_karras",
                "dpmsolver++_karras",
            }:
                scheduler = DPMSolverMultistepScheduler.from_config(config)
                setattr(scheduler, "algorithm_type", "dpmsolver++")
                if hasattr(scheduler, "use_karras_sigmas"):
                    setattr(scheduler, "use_karras_sigmas", True)
            elif normalized in {"euler_a", "k_euler_a", "euler_ancestral"}:
                scheduler = EulerAncestralDiscreteScheduler.from_config(config)
            elif normalized in {"euler", "k_euler"}:
                scheduler = EulerDiscreteScheduler.from_config(config)
            elif normalized in {"ddim"}:
                scheduler = DDIMScheduler.from_config(config)
            elif normalized in {"pndm"}:
                scheduler = PNDMScheduler.from_config(config)
            elif normalized in {"kdpm_2_a", "kdpm2_a", "k_dpm_2_a"}:
                scheduler = KDPM2AncestralDiscreteScheduler.from_config(config)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to apply scheduler %s: %s", scheduler_name, exc)
            scheduler = None

        if scheduler is None:
            return None

        pipeline.scheduler = scheduler
        return scheduler.__class__.__name__

    def _run_inference(
        self,
        handle: PipelineHandle,
        prompt: str,
        duration_seconds: float,
        guidance_scale: float,
        seed: Optional[int],
        init_image: Optional[PILImage],
        init_strength: Optional[float],
        num_inference_steps: int,
        scheduler_name: Optional[str],
    ) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        assert torch is not None

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._device).manual_seed(seed)

        audio_length = max(1.0, float(duration_seconds))

        pipeline = handle.pipeline
        scheduler_actual = self._configure_scheduler(pipeline, scheduler_name)
        try:
            kwargs = {
                "prompt": prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "audio_length_in_s": audio_length,
                "generator": generator,
            }
            if init_image is not None:
                kwargs["image"] = init_image
                if init_strength is not None:
                    kwargs["strength"] = init_strength
            result = pipeline(**kwargs)
        except TypeError:
            kwargs = {
                "prompt": prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "audio_length_in_s": int(round(audio_length)),
                "generator": generator,
            }
            if init_image is not None:
                kwargs["image"] = init_image
                if init_strength is not None:
                    kwargs["strength"] = init_strength
            result = pipeline(**kwargs)

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
            "num_inference_steps": num_inference_steps,
        }
        if seed is not None:
            extras["seed"] = seed
        if scheduler_actual is not None:
            extras["scheduler"] = scheduler_actual
        elif scheduler_name is not None:
            extras["scheduler"] = scheduler_name

        return self._prepare_waveform(waveform), sample_rate, extras

    def _placeholder_waveform(
        self,
        prompt: str,
        duration_seconds: float,
        reason: str,
        seed: Optional[int],
        guidance_scale: float,
        num_inference_steps: int,
        scheduler_name: Optional[str],
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
            "guidance_scale": guidance_scale,
            "placeholder_reason": reason,
            "prompt_hash": self._prompt_hash(prompt),
            "render_seconds": duration_seconds,
            "num_inference_steps": num_inference_steps,
        }
        if seed is not None:
            extras["seed"] = seed
        if scheduler_name is not None:
            extras["scheduler"] = scheduler_name

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
            if override == "cuda" and torch.cuda.is_available():  # pragma: no cover
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
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            # pragma: no cover - hardware dependent
            # Riffusion outputs distort on MPS with float16; prefer float32 even if slower.
            return "mps", torch.float32
        return "cpu", torch.float32

    def _missing_dependency_reason(self) -> str:
        if torch is None:
            guidance = "run `uv sync --project worker --extra inference` to install torch/diffusers"
            detail = TORCH_IMPORT_ERROR or "not installed"
            return f"torch_unavailable:{detail};{guidance}"
        return "unknown"

    def _pillow_reason(self) -> str:
        guidance = "run `uv sync --project worker --extra inference` to install pillow"
        detail = PIL_IMPORT_ERROR or "not installed"
        return f"pillow_unavailable:{detail};{guidance}"

    def _compose_prompt(
        self,
        base_prompt: str,
        *,
        theme: ThemeDescriptor | None,
        previous: SectionRender | None,
    ) -> str:
        segments: list[str] = [base_prompt.strip()]
        if theme is not None:
            instrumentation = (
                ", ".join(theme.instrumentation)
                if theme.instrumentation
                else "blended instrumentation"
            )
            descriptor_parts = [
                f"Maintain the {theme.motif} motif",
                f"with {instrumentation}",
                f"following a {theme.rhythm} feel",
            ]
            if theme.texture:
                descriptor_parts.append(f"and preserve the {theme.texture}")
            segments.append(" ".join(descriptor_parts) + ".")
        if previous is not None:
            prev_label = previous.extras.get("section_label") or previous.extras.get("section_id")
            descriptor = prev_label or "section"
            segments.append(
                "Flow smoothly from the previous "
                f"{descriptor}, evolving its ideas without changing the instrumentation."
            )
        return " ".join(segment.strip() for segment in segments if segment.strip())

    def _resolve_spectrogram_decoder(
        self,
        expected_sample_rate: int,
    ) -> Optional[SpectrogramImageDecoder]:
        if not self._allow_inference:
            self._spectrogram_error = "inference_disabled"
            return None
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
            if not self._spectrogram_warned:
                logger.warning("Spectrogram decoder unavailable: %s", exc)
                self._spectrogram_warned = True
            return None

        self._spectrogram_decoder = decoder
        self._spectrogram_error = None
        self._spectrogram_mode = getattr(decoder._inverse_mel, "decoder_mode", "unknown")
        self._spectrogram_warned = False
        logger.info(
            "Spectrogram decoder initialised (mode=%s, sample_rate=%s)",
            self._spectrogram_mode,
            params.sample_rate,
        )
        return decoder

    def _prepare_init_image(self, render: SectionRender) -> Optional[PILImage]:
        if Image is None:
            return None
        if render.waveform.size == 0:
            return None
        decoder = self._resolve_spectrogram_decoder(render.sample_rate)
        if decoder is None or not hasattr(decoder, "encode"):
            return None
        waveform = ensure_waveform_channels(render.waveform)
        total_seconds = waveform.shape[0] / render.sample_rate
        if total_seconds <= 0.5:
            return None
        tail_seconds = min(4.0, total_seconds)
        tail_samples = max(1, int(tail_seconds * render.sample_rate))
        tail = waveform[-tail_samples:]
        try:
            return decoder.encode(tail, render.sample_rate)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to encode continuation spectrogram: %s", exc)
            return None

    def _audio_from_images(
        self,
        result: Any,
        default_sample_rate: int,
    ) -> Tuple[np.ndarray, int]:
        if Image is None:
            raise GenerationFailure(self._pillow_reason())

        images = getattr(result, "images", None)
        if not images:
            raise GenerationFailure("pipeline returned empty audio result")

        raw_image = images[0]
        if Image is not None and isinstance(raw_image, Image.Image):
            pil_image = raw_image
        else:
            array = np.asarray(raw_image)
            if array.dtype != np.uint8:
                upper_bound = float(np.max(array)) if array.size else 0.0
                if upper_bound <= 1.0:
                    array = np.clip(array, 0.0, 1.0) * 255.0
                array = np.clip(array, 0.0, 255.0).astype(np.uint8)
            if Image is None:
                raise GenerationFailure(self._pillow_reason())
            if array.ndim == 2:
                pil_image = Image.fromarray(array, mode="L").convert("RGB")
            else:
                pil_image = Image.fromarray(array)

        decoder = self._resolve_spectrogram_decoder(default_sample_rate)
        if decoder is None:
            reason = self._spectrogram_error or "spectrogram_decoder_unavailable"
            raise GenerationFailure(reason)

        waveform, sample_rate = decoder.decode(
            pil_image,
            refine_phase=self._enable_phase_refinement,
        )
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
