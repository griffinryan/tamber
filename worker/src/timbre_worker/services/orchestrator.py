"""High-level composition orchestrator coordinating backends."""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from ..app.models import (
    CompositionSection,
    CompositionPlan,
    GenerationArtifact,
    GenerationMetadata,
    GenerationRequest,
)
from ..app.settings import Settings
from .audio_utils import ensure_waveform_channels, write_waveform
from .musicgen import MusicGenService
from .planner import CompositionPlanner
from .riffusion import RiffusionService
from .types import SectionRender


class ComposerOrchestrator:
    """Coordinates planning, backend selection, and audio assembly."""

    def __init__(
        self,
        settings: Settings,
        planner: CompositionPlanner,
        riffusion: RiffusionService,
        musicgen: MusicGenService,
    ) -> None:
        self._settings = settings
        self._planner = planner
        self._riffusion = riffusion
        self._musicgen = musicgen
        self._artifact_root = settings.artifact_root

    async def warmup(self) -> None:
        await asyncio.gather(
            self._riffusion.warmup(),
            self._musicgen.warmup(),
        )

    async def generate(
        self,
        job_id: str,
        request: GenerationRequest,
        progress_cb: Optional[
            Callable[[int, int, SectionRender], Awaitable[None]]
        ] = None,
        mix_cb: Optional[Callable[[float], Awaitable[None]]] = None,
    ) -> GenerationArtifact:
        plan = request.plan or self._planner.build_plan(request)

        renders: List[SectionRender] = []
        for index, section in enumerate(plan.sections):
            backend = self._select_backend(section.model_id or request.model_id or "")
            render_hint = self._render_duration_hint(index, section, len(plan.sections))
            logger.debug(
                "Rendering section %s via %s (target %.2fs, render %.2fs)",
                section.section_id,
                backend.__class__.__name__,
                section.target_seconds,
                render_hint,
            )
            render = await backend.render_section(
                request,
                section,
                plan=plan,
                model_id=section.model_id or request.model_id,
                render_seconds=render_hint,
            )
            renders.append(render)
            if progress_cb is not None:
                await progress_cb(index + 1, len(plan.sections), render)

        trimmed, sample_rate = self._prepare_section_waveforms(plan, renders)
        waveform, crossfades = self._combine_sections(plan, trimmed, sample_rate)
        if mix_cb is not None:
            await mix_cb(len(waveform) / sample_rate)

        artifact_path = self._artifact_root / f"{job_id}.wav"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        write_waveform(artifact_path, waveform, sample_rate)

        extras: Dict[str, object] = {
            "backend": "composer",
            "placeholder": any(
                render.extras.get("placeholder", False) for render in renders
            ),
            "sections": [render.extras for render in renders],
            "mix": {
                "sample_rate": sample_rate,
                "duration_seconds": len(waveform) / sample_rate,
                "crossfades": crossfades,
            },
            "sample_rate": sample_rate,
        }

        metadata = GenerationMetadata(
            prompt=request.prompt,
            seed=request.seed,
            model_id=request.model_id,
            duration_seconds=int(round(len(waveform) / sample_rate)),
            extras=extras,
            plan=plan,
        )

        return GenerationArtifact(
            job_id=job_id,
            artifact_path=str(artifact_path),
            metadata=metadata,
        )

    def _select_backend(self, model_id: str) -> object:
        token = (model_id or "").lower()
        if "musicgen" in token or "audiocraft" in token or "facebook/musicgen" in token:
            return self._musicgen
        return self._riffusion

    def _render_duration_hint(
        self,
        index: int,
        section: CompositionSection,
        total_sections: int,
    ) -> float:
        padding = 0.0
        if total_sections > 1:
            if index == 0 or index == total_sections - 1:
                padding = 0.25
            else:
                padding = 0.5
        return max(
            float(section.target_seconds) + padding,
            float(section.target_seconds),
        )

    def _prepare_section_waveforms(
        self,
        plan: CompositionPlan,
        renders: List[SectionRender],
    ) -> Tuple[List[np.ndarray], int]:
        if not renders:
            raise RuntimeError("composition produced no sections")
        prepared: List[np.ndarray] = []
        sample_rate = renders[0].sample_rate
        for section, render in zip(plan.sections, renders, strict=True):
            if render.sample_rate != sample_rate:
                raise RuntimeError("section sample rates diverged")
            waveform = ensure_waveform_channels(render.waveform)
            target_seconds = float(section.target_seconds)
            target_samples = max(1, int(round(target_seconds * sample_rate)))
            if waveform.shape[0] > target_samples:
                start = max(0, (waveform.shape[0] - target_samples) // 2)
                end = start + target_samples
                prepared.append(waveform[start:end])
            elif waveform.shape[0] < target_samples:
                normalized = waveform if waveform.ndim == 2 else waveform.reshape(-1, 1)
                channels = normalized.shape[1] if normalized.ndim == 2 else 1
                pad = target_samples - normalized.shape[0]
                padding = np.zeros((pad, channels), dtype=np.float32)
                padded = np.vstack((normalized, padding))
                if channels == 1:
                    prepared.append(padded.reshape(-1))
                else:
                    prepared.append(padded)
            else:
                prepared.append(waveform)
        if len(prepared) != len(plan.sections):
            raise RuntimeError("prepared waveforms do not match plan sections")
        return prepared, sample_rate

    def _combine_sections(
        self,
        plan: CompositionPlan,
        waveforms: List[np.ndarray],
        sample_rate: int,
    ) -> Tuple[np.ndarray, List[Dict[str, object]]]:
        if not waveforms:
            raise RuntimeError("composition produced no sections")

        combined = waveforms[0]
        crossfade_records: List[Dict[str, object]] = []

        for index, waveform in enumerate(waveforms[1:], start=1):
            left = combined
            right = waveform
            fade_seconds = self._crossfade_seconds(left, right, sample_rate)
            combined = self._crossfade_append(left, right, sample_rate, fade_seconds)
            crossfade_records.append(
                {
                    "from": plan.sections[index - 1].section_id,
                    "to": plan.sections[index].section_id,
                    "seconds": fade_seconds,
                }
            )

        peak = float(np.max(np.abs(combined))) if combined.size else 0.0
        if peak > 0.98:
            combined = combined / peak * 0.98

        return combined.astype(np.float32), crossfade_records

    def _crossfade_seconds(
        self,
        left: np.ndarray,
        right: np.ndarray,
        sample_rate: int,
    ) -> float:
        left_duration = left.shape[0] / sample_rate
        right_duration = right.shape[0] / sample_rate
        base = min(0.5, max(0.1, min(left_duration, right_duration) * 0.2))
        return float(base)

    def _crossfade_append(
        self,
        left: np.ndarray,
        right: np.ndarray,
        sample_rate: int,
        fade_seconds: float,
    ) -> np.ndarray:
        fade_samples = int(
            max(
                1,
                min(
                    sample_rate * fade_seconds,
                    left.shape[0] - 1,
                    right.shape[0] - 1,
                ),
            )
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
        result = np.vstack((left_main, blended, right_rest))
        return result.reshape(-1, result.shape[-1])
