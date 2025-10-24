"""MusicGen backend service built on Hugging Face transformers."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from loguru import logger

from ..app.models import (
    CompositionPlan,
    CompositionSection,
    GenerationRequest,
    ThemeDescriptor,
)
from ..app.settings import Settings
from .audio_utils import ensure_waveform_channels, resample_waveform
from .types import SectionRender

try:  # pragma: no cover - deferred dependency import
    import torch
except Exception as exc:  # noqa: BLE001
    torch = None  # type: ignore[assignment]
    TORCH_IMPORT_ERROR = exc
else:  # pragma: no cover
    TORCH_IMPORT_ERROR = None

try:  # pragma: no cover
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
except Exception as exc:  # noqa: BLE001
    AutoProcessor = None  # type: ignore[assignment]
    MusicgenForConditionalGeneration = None  # type: ignore[assignment]
    TRANSFORMERS_IMPORT_ERROR = exc
else:  # pragma: no cover
    TRANSFORMERS_IMPORT_ERROR = None


MODEL_REGISTRY = {
    "musicgen-small": "facebook/musicgen-small",
    "musicgen-medium": "facebook/musicgen-medium",
    "musicgen-large": "facebook/musicgen-large",
    "musicgen-stereo-small": "facebook/musicgen-stereo-small",
    "musicgen-stereo-medium": "facebook/musicgen-stereo-medium",
    "musicgen-stereo-large": "facebook/musicgen-stereo-large",
}


@dataclass
class ModelHandle:
    model: MusicgenForConditionalGeneration
    processor: AutoProcessor
    sample_rate: int
    frame_rate: int


class MusicGenService:
    """Text-to-music generation via transformers MusicGen checkpoints."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        default_model_id: Optional[str] = None,
    ) -> None:
        self._settings = settings
        if default_model_id is not None:
            self._default_model_id = default_model_id
        elif settings is not None:
            self._default_model_id = settings.musicgen_default_model_id
        else:
            self._default_model_id = "musicgen-small"

        self._handles: Dict[str, Optional[ModelHandle]] = {}
        self._lock = asyncio.Lock()
        self._device = self._select_device()

        self._top_k = settings.musicgen_top_k if settings is not None else None
        self._top_p = settings.musicgen_top_p if settings is not None else None
        self._temperature = settings.musicgen_temperature if settings is not None else None
        self._cfg_coef = settings.musicgen_cfg_coef if settings is not None else None
        self._two_step_cfg = settings.musicgen_two_step_cfg if settings is not None else None
        self._supports_two_step_cfg = True

    @property
    def default_model_id(self) -> str:
        return self._default_model_id

    async def warmup(self) -> None:
        await self._ensure_model(self._default_model_id)

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
        handle, placeholder_reason = await self._ensure_model(
            model_id or section.model_id or self._default_model_id
        )

        duration_hint = (
            render_seconds if render_seconds is not None else float(section.target_seconds)
        )
        duration = float(max(1.0, duration_hint))
        section_seed: Optional[int] = None
        if request.seed is not None:
            offset = section.seed_offset or 0
            section_seed = request.seed + offset

        prompt_text = self._compose_prompt(section.prompt, theme=theme, previous=previous_render)
        conditioning_requested = motif_seed is not None or previous_render is not None
        top_k = request.musicgen_top_k if request.musicgen_top_k is not None else self._top_k
        top_p = request.musicgen_top_p if request.musicgen_top_p is not None else self._top_p
        temperature = (
            request.musicgen_temperature
            if request.musicgen_temperature is not None
            else self._temperature
        )
        cfg_coef = request.musicgen_cfg_coef or self._cfg_coef
        two_step_cfg = request.musicgen_two_step_cfg
        if two_step_cfg is None:
            two_step_cfg = self._two_step_cfg

        if handle is None:
            conditioning_extras: Dict[str, Any] = {
                "audio_conditioning_requested": conditioning_requested,
                "audio_conditioning_applied": False,
                "conditioning_reason": placeholder_reason or "musicgen_unavailable",
                "audio_prompt_segments": [],
                "audio_prompt_seconds": 0.0,
            }
            waveform, sample_rate, extras = await asyncio.to_thread(
                self._placeholder_waveform,
                prompt_text,
                duration,
                placeholder_reason or "musicgen_unavailable",
                section_seed,
                top_k,
                top_p,
                temperature,
                cfg_coef,
                two_step_cfg,
            )
            extras["placeholder"] = True
            extras["backend"] = "musicgen"
            extras["device"] = "placeholder"
        else:
            inputs, conditioning_extras = self._prepare_model_inputs(
                handle,
                prompt_text,
                motif_seed=motif_seed,
                previous_render=previous_render,
                conditioning_requested=conditioning_requested,
            )
            waveform, sample_rate, extras = await asyncio.to_thread(
                self._generate_waveform,
                handle,
                inputs,
                prompt_text,
                duration,
                section_seed,
                top_k,
                top_p,
                temperature,
                cfg_coef,
                two_step_cfg,
            )
            extras["placeholder"] = False
            extras["backend"] = "musicgen"
            extras["device"] = self._device

        actual_seconds = len(ensure_waveform_channels(waveform)) / float(sample_rate)
        extras.update(
            {
                "section_id": section.section_id,
                "section_label": section.label,
                "section_role": section.role.value,
                "plan_version": plan.version,
                "target_seconds": float(section.target_seconds),
                "render_seconds": actual_seconds,
            }
        )
        if placeholder_reason:
            extras["placeholder_reason"] = placeholder_reason
        if section_seed is not None:
            extras["seed"] = section_seed
        if theme is not None:
            extras["theme"] = theme.model_dump()
        extras.update(conditioning_extras)
        if motif_seed is not None:
            seed_section = motif_seed.extras.get("section_id") if isinstance(motif_seed.extras, dict) else None
            if seed_section is not None:
                extras["motif_seed_section"] = seed_section
        for key, value in (
            ("top_k", top_k),
            ("top_p", top_p),
            ("temperature", temperature),
            ("cfg_coef", cfg_coef),
            ("two_step_cfg", two_step_cfg),
        ):
            if value is not None:
                extras[key] = value

        return SectionRender(waveform=waveform, sample_rate=sample_rate, extras=extras)

    async def _ensure_model(
        self,
        model_id: str,
    ) -> Tuple[Optional[ModelHandle], Optional[str]]:
        async with self._lock:
            if model_id in self._handles:
                return self._handles[model_id], None

        resolved = MODEL_REGISTRY.get(model_id, model_id)

        if torch is None:
            reason = self._missing_dependency_reason()
            async with self._lock:
                self._handles[model_id] = None
            logger.warning("MusicGen unavailable: {}", reason)
            return None, reason

        if AutoProcessor is None or MusicgenForConditionalGeneration is None:
            reason = self._missing_dependency_reason()
            async with self._lock:
                self._handles[model_id] = None
            logger.warning("MusicGen unavailable: {}", reason)
            return None, reason

        try:
            processor = AutoProcessor.from_pretrained(resolved)
            model = MusicgenForConditionalGeneration.from_pretrained(resolved)
            model = model.to(self._device)
            model.eval()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load MusicGen model %s", resolved)
            async with self._lock:
                self._handles[model_id] = None
            return None, f"load_error:{exc.__class__.__name__}"

        sample_rate = getattr(model.config, "sampling_rate", 32000)
        frame_rate = getattr(model.config, "frame_rate", 50)
        handle = ModelHandle(
            model=model,
            processor=processor,
            sample_rate=int(sample_rate),
            frame_rate=int(frame_rate),
        )
        async with self._lock:
            self._handles[model_id] = handle
        return handle, None

    def _build_generation_kwargs(
        self,
        *,
        max_new_tokens: int,
        top_k: Optional[int],
        top_p: Optional[float],
        temperature: Optional[float],
        cfg_coef: Optional[float],
        two_step_cfg: Optional[bool],
    ) -> Tuple[Dict[str, Any], bool]:
        """Assemble arguments for MusicGen.generate and report two-step CFG usage."""
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
        }
        if top_k is not None and top_k > 0:
            generation_kwargs["top_k"] = int(top_k)
        if top_p is not None and 0.0 < top_p <= 1.0:
            generation_kwargs["top_p"] = float(top_p)
        if temperature is not None and temperature > 0.0:
            generation_kwargs["temperature"] = float(temperature)
        if cfg_coef is not None and cfg_coef > 0.0:
            generation_kwargs["guidance_scale"] = float(cfg_coef)

        requested_two_step = bool(two_step_cfg) if two_step_cfg is not None else False
        applied_two_step = requested_two_step and self._supports_two_step_cfg

        return generation_kwargs, applied_two_step

    def _generate_waveform(
        self,
        handle: ModelHandle,
        inputs: Any,
        prompt: str,
        duration_seconds: float,
        seed: Optional[int],
        top_k: Optional[int],
        top_p: Optional[float],
        temperature: Optional[float],
        cfg_coef: Optional[float],
        two_step_cfg: Optional[bool],
    ) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        assert torch is not None
        if hasattr(inputs, "to"):
            inputs = inputs.to(self._device)
        else:
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
        if hasattr(inputs, "data"):
            model_inputs = dict(inputs.data)
        else:
            model_inputs = inputs

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._device).manual_seed(seed)

        tokens = self._seconds_to_tokens(duration_seconds, handle)
        generation_kwargs, two_step_applied = self._build_generation_kwargs(
            max_new_tokens=tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            cfg_coef=cfg_coef,
            two_step_cfg=two_step_cfg,
        )

        with torch.no_grad():
            audio_tokens = handle.model.generate(
                **model_inputs,
                generator=generator,
                **generation_kwargs,
            )
        decode_tokens = audio_tokens
        if hasattr(audio_tokens, "device"):
            device_type = getattr(audio_tokens.device, "type", None)
            if device_type == "mps":
                decode_tokens = audio_tokens.cpu()

        decoded_batch = handle.processor.batch_decode(decode_tokens, return_tensors=True)
        decoded = decoded_batch[0]
        if hasattr(decoded, "cpu"):
            decoded = decoded.cpu()
        if hasattr(decoded, "numpy"):
            decoded = decoded.numpy()

        waveform = ensure_waveform_channels(np.asarray(decoded).T)
        sample_rate = handle.sample_rate
        if waveform.ndim == 1:
            waveform = waveform.reshape(-1, 1)
        if waveform.shape[1] == 1:
            waveform = np.repeat(waveform, 2, axis=1)
        waveform = waveform.astype(np.float32)

        if sample_rate != 44100:
            waveform = resample_waveform(waveform, sample_rate, 44100)
            sample_rate = 44100

        extras: Dict[str, Any] = {
            "sample_rate": sample_rate,
            "prompt_hash": self._prompt_hash(prompt),
        }
        if seed is not None:
            extras["seed"] = seed
        if two_step_cfg is not None:
            extras["two_step_cfg"] = bool(two_step_cfg)
            extras["two_step_cfg_applied"] = two_step_applied
            extras["two_step_cfg_supported"] = self._supports_two_step_cfg

        return waveform, sample_rate, extras

    def _placeholder_waveform(
        self,
        prompt: str,
        duration_seconds: float,
        reason: str,
        seed: Optional[int],
        top_k: Optional[int],
        top_p: Optional[float],
        temperature: Optional[float],
        cfg_coef: Optional[float],
        two_step_cfg: Optional[bool],
    ) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        sample_rate = 32000
        duration_seconds = max(1.0, min(duration_seconds, 30.0))
        total_samples = int(duration_seconds * sample_rate)
        t = np.linspace(0, duration_seconds, total_samples, endpoint=False, dtype=np.float32)

        seed_value = (
            seed if seed is not None else abs(hash((prompt, duration_seconds))) % (2**32)
        )
        base_freq_source = (hash(prompt), seed_value)
        base_freq = 110 + (abs(hash(base_freq_source)) % 440)
        waveform = 0.18 * np.sin(2 * np.pi * base_freq * t)
        waveform += 0.06 * np.sin(2 * np.pi * base_freq * 0.5 * t)
        waveform += 0.03 * np.sin(2 * np.pi * (base_freq * 1.5) * t)
        waveform += 0.02 * np.random.default_rng(seed=seed_value).standard_normal(
            size=total_samples
        )
        waveform = np.clip(waveform, -1.0, 1.0).astype(np.float32)

        extras = {
            "sample_rate": sample_rate,
            "prompt_hash": self._prompt_hash(prompt),
            "placeholder_reason": reason,
        }
        if seed is not None:
            extras["seed"] = seed
        for key, value in (
            ("top_k", top_k),
            ("top_p", top_p),
            ("temperature", temperature),
            ("cfg_coef", cfg_coef),
        ):
            if value is not None:
                extras[key] = value
        if two_step_cfg is not None:
            extras["two_step_cfg"] = bool(two_step_cfg)
            extras["two_step_cfg_applied"] = bool(two_step_cfg) and self._supports_two_step_cfg
            extras["two_step_cfg_supported"] = self._supports_two_step_cfg

        return waveform, sample_rate, extras

    def _seconds_to_tokens(self, duration: float, handle: ModelHandle) -> int:
        frame_rate = max(handle.frame_rate, 1)
        return max(int(round(duration * frame_rate)), frame_rate)

    def _select_device(self) -> str:
        if torch is None:
            return "cpu"
        if torch.cuda.is_available():  # pragma: no cover
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _missing_dependency_reason(self) -> str:
        if torch is None:
            detail = TORCH_IMPORT_ERROR or "torch_not_installed"
            return f"torch_unavailable:{detail}"
        if AutoProcessor is None or MusicgenForConditionalGeneration is None:
            detail = TRANSFORMERS_IMPORT_ERROR or "transformers_not_installed"
            return f"transformers_unavailable:{detail}"
        return "unknown"

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
            parts = [
                f"Keep the {theme.motif} motif",
                f"with {instrumentation}",
                f"locked to a {theme.rhythm}",
            ]
            if theme.texture:
                parts.append(f"inside a {theme.texture}")
            segments.append(" ".join(parts) + ".")
        if previous is not None:
            prev_label = previous.extras.get("section_label") or previous.extras.get("section_id")
            descriptor = prev_label or "section"
            segments.append(
                "Continue seamlessly from the previous "
                f"{descriptor}, preserving tone and instrumentation while evolving the motif."
            )
        return " ".join(fragment.strip() for fragment in segments if fragment.strip())

    def _prompt_hash(self, prompt: str) -> str:
        digest = hashlib.blake2s(prompt.encode("utf-8"), digest_size=8)
        return digest.hexdigest()

    def _prepare_model_inputs(
        self,
        handle: ModelHandle,
        prompt: str,
        *,
        motif_seed: SectionRender | None,
        previous_render: SectionRender | None,
        conditioning_requested: bool,
    ) -> Tuple[Any, Dict[str, Any]]:
        audio_prompt, audio_meta = self._build_audio_prompt(
            handle,
            motif_seed=motif_seed,
            previous_render=previous_render,
        )
        extras: Dict[str, Any] = {
            "audio_conditioning_requested": conditioning_requested,
        }
        extras.update(audio_meta)

        processor_kwargs: Dict[str, Any] = {
            "text": [prompt],
            "padding": True,
            "return_tensors": "pt",
        }
        if audio_prompt is not None:
            processor_kwargs["audio"] = [audio_prompt]
            processor_kwargs["sampling_rate"] = handle.sample_rate

        try:
            inputs = handle.processor(**processor_kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MusicGen conditioning failed during preprocessing: %s", exc)
            extras["audio_conditioning_applied"] = False
            extras["audio_conditioning_error"] = str(exc)
            inputs = handle.processor(
                text=[prompt],
                padding=True,
                return_tensors="pt",
            )
        return inputs, extras

    def _build_audio_prompt(
        self,
        handle: ModelHandle,
        *,
        motif_seed: SectionRender | None,
        previous_render: SectionRender | None,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        segments: list[np.ndarray] = []
        segment_meta: list[Dict[str, Any]] = []
        if motif_seed is not None:
            motif_audio = self._prepare_audio_prompt_segment(
                motif_seed,
                handle.sample_rate,
                max_seconds=6.0,
                segment="head",
            )
            if motif_audio.size > 0:
                segments.append(motif_audio)
                segment_meta.append(
                    {
                        "source": "motif",
                        "seconds": float(motif_audio.size / handle.sample_rate),
                    }
                )
        if previous_render is not None:
            tail_audio = self._prepare_audio_prompt_segment(
                previous_render,
                handle.sample_rate,
                max_seconds=4.0,
                segment="tail",
            )
            if tail_audio.size > 0:
                segments.append(tail_audio)
                segment_meta.append(
                    {
                        "source": "previous_tail",
                        "seconds": float(tail_audio.size / handle.sample_rate),
                    }
                )

        if not segments:
            return None, {
                "audio_conditioning_applied": False,
                "audio_prompt_segments": [],
                "audio_prompt_seconds": 0.0,
            }

        combined = np.concatenate(segments).astype(np.float32)
        return combined, {
            "audio_conditioning_applied": True,
            "audio_prompt_seconds": float(combined.size / handle.sample_rate),
            "audio_prompt_segments": segment_meta,
        }

    def _prepare_audio_prompt_segment(
        self,
        render: SectionRender,
        target_rate: int,
        *,
        max_seconds: float,
        segment: str,
    ) -> np.ndarray:
        waveform = ensure_waveform_channels(render.waveform)
        if waveform.ndim == 1:
            mono = waveform.astype(np.float32)
        else:
            mono = waveform.mean(axis=1).astype(np.float32)
        if render.sample_rate != target_rate:
            resampled = resample_waveform(waveform, render.sample_rate, target_rate)
            if resampled.ndim == 1:
                mono = resampled.astype(np.float32)
            else:
                mono = resampled.mean(axis=1).astype(np.float32)

        max_samples = int(max_seconds * target_rate)
        if max_samples <= 0:
            return np.zeros(0, dtype=np.float32)
        if mono.size > max_samples:
            if segment == "tail":
                mono = mono[-max_samples:]
            else:
                mono = mono[:max_samples]

        mono = mono.astype(np.float32, copy=True)
        if mono.size == 0:
            return mono
        fade_samples = min(256, max(1, mono.size // 24))
        if fade_samples > 0:
            fade = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
            if segment == "tail":
                mono[:fade_samples] *= fade
            else:
                mono[-fade_samples:] *= fade[::-1]
        mono = np.clip(mono, -1.0, 1.0)
        return mono
