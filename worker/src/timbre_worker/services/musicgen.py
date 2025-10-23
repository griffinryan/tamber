"""MusicGen backend service providing melodiic phrase generation."""

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
from .audio_utils import ensure_waveform_channels
from .types import SectionRender

try:  # pragma: no cover - optional dependency imports are validated at runtime
    import torch
except Exception as exc:  # noqa: BLE001
    torch = None  # type: ignore[assignment]
    TORCH_IMPORT_ERROR = exc
else:  # pragma: no cover
    TORCH_IMPORT_ERROR = None

try:  # pragma: no cover
    from audiocraft.models import MusicGen
except Exception as exc:  # noqa: BLE001
    MusicGen = None  # type: ignore[assignment]
    MUSICGEN_IMPORT_ERROR = exc
else:  # pragma: no cover
    MUSICGEN_IMPORT_ERROR = None


MODEL_REGISTRY = {
    "musicgen-small": "facebook/musicgen-small",
    "musicgen-medium": "facebook/musicgen-medium",
}


@dataclass
class ModelHandle:
    model: Any
    sample_rate: int


class MusicGenService:
    """Adapter around Audiocraft MusicGen with graceful fallbacks."""

    def __init__(self, *, default_model_id: str = "musicgen-small") -> None:
        self._default_model_id = default_model_id
        self._models: Dict[str, Optional[ModelHandle]] = {}
        self._lock = asyncio.Lock()
        self._device = self._select_device()

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
    ) -> SectionRender:
        model_key = model_id or section.model_id or self._default_model_id
        handle, placeholder_reason = await self._ensure_model(model_key)

        duration_hint = (
            render_seconds if render_seconds is not None else float(section.target_seconds)
        )
        duration = max(1.0, float(duration_hint))
        section_seed: Optional[int] = None
        if request.seed is not None:
            offset = section.seed_offset or 0
            section_seed = request.seed + offset

        prompt_text = self._compose_prompt(section.prompt, theme=theme, previous=previous_render)

        if handle is None:
            waveform, sample_rate, extras = await asyncio.to_thread(
                self._placeholder_waveform,
                prompt_text,
                duration,
                placeholder_reason or "musicgen_unavailable",
                section_seed,
            )
            extras["placeholder"] = True
            extras["backend"] = "musicgen"
            extras["device"] = "placeholder"
        else:
            waveform, sample_rate, extras = await asyncio.to_thread(
                self._render_waveform,
                handle,
                prompt_text,
                duration,
                section_seed,
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

        return SectionRender(waveform=waveform, sample_rate=sample_rate, extras=extras)

    async def _ensure_model(
        self,
        model_id: str,
    ) -> Tuple[Optional[ModelHandle], Optional[str]]:
        async with self._lock:
            if model_id in self._models:
                return self._models[model_id], None

        resolved = MODEL_REGISTRY.get(model_id, model_id)

        if torch is None or MusicGen is None:
            reason = self._missing_dependency_reason()
            async with self._lock:
                self._models[model_id] = None
            logger.warning("MusicGen unavailable: {}", reason)
            return None, reason

        try:
            model = await asyncio.to_thread(MusicGen.get_pretrained, resolved, device=self._device)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load MusicGen model %s", resolved)
            async with self._lock:
                self._models[model_id] = None
            return None, f"load_error:{exc.__class__.__name__}"

        sample_rate = getattr(model, "sample_rate", 32000)
        handle = ModelHandle(model=model, sample_rate=sample_rate)
        async with self._lock:
            self._models[model_id] = handle
        return handle, None

    def _render_waveform(
        self,
        handle: ModelHandle,
        prompt: str,
        duration_seconds: float,
        seed: Optional[int],
    ) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        assert MusicGen is not None and torch is not None
        model = handle.model

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._device).manual_seed(seed)

        try:
            model.set_generation_params(duration=duration_seconds, use_sampling=True)
            audio = model.generate(
                descriptions=[prompt],
                progress=False,
                return_tokens=False,
                generator=generator,
            )
        except TypeError:
            audio = model.generate(
                descriptions=[prompt],
                progress=False,
                generator=generator,
            )

        if isinstance(audio, list):
            tensor = audio[0]
        else:
            tensor = audio

        if hasattr(tensor, "detach"):
            tensor = tensor.detach()
        if hasattr(tensor, "cpu"):
            tensor = tensor.cpu()

        waveform = np.asarray(tensor, dtype=np.float32)
        if waveform.ndim == 2:
            waveform = waveform.T

        extras: Dict[str, Any] = {
            "sample_rate": handle.sample_rate,
            "prompt_hash": self._prompt_hash(prompt),
        }
        if seed is not None:
            extras["seed"] = seed

        return waveform, handle.sample_rate, extras

    def _placeholder_waveform(
        self,
        prompt: str,
        duration_seconds: float,
        reason: str,
        seed: Optional[int],
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
        return waveform, sample_rate, extras

    def _select_device(self) -> str:
        if torch is None:
            return "cpu"
        if torch.cuda.is_available():  # pragma: no cover
            return "cuda"
        if (
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()  # pragma: no cover
        ):
            return "mps"
        return "cpu"

    def _missing_dependency_reason(self) -> str:
        if torch is None:
            detail = TORCH_IMPORT_ERROR or "torch_not_installed"
            return f"torch_unavailable:{detail}"
        if MusicGen is None:
            detail = MUSICGEN_IMPORT_ERROR or "audiocraft_not_installed"
            return f"musicgen_unavailable:{detail}"
        return "unknown"

    def _prompt_hash(self, prompt: str) -> str:
        digest = hashlib.blake2s(prompt.encode("utf-8"), digest_size=8)
        return digest.hexdigest()

    def _compose_prompt(
        self,
        base_prompt: str,
        *,
        theme: ThemeDescriptor | None,
        previous: SectionRender | None,
    ) -> str:
        segments: list[str] = [base_prompt.strip()]
        if theme is not None:
            instrumentation = ", ".join(theme.instrumentation) if theme.instrumentation else "blended instrumentation"
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
            segments.append(
                f"Continue seamlessly from the previous {prev_label or 'section'}, preserving tone and instrumentation while evolving the motif."
            )
        return " ".join(fragment.strip() for fragment in segments if fragment.strip())
