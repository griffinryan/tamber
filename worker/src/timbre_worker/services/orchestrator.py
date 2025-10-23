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
from .audio_utils import (
    crossfade_append,
    ensure_waveform_channels,
    fit_to_length,
    normalise_loudness,
    rms_level,
    soft_limiter,
    write_waveform,
)
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
        total_sections = len(plan.sections)
        for index, section in enumerate(plan.sections):
            backend = self._select_backend(section.model_id or request.model_id or "")
            render_hint = self._render_duration_hint(plan, index, section, total_sections)
            previous_render = renders[-1] if renders else None
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
                theme=plan.theme,
                previous_render=previous_render,
            )
            renders.append(render)
            if progress_cb is not None:
                await progress_cb(index + 1, len(plan.sections), render)

        trimmed, sample_rate, section_rms = self._prepare_section_waveforms(plan, renders)
        waveform, crossfades = self._combine_sections(plan, trimmed, sample_rate)
        waveform = normalise_loudness(waveform, target_rms=0.2)
        waveform = soft_limiter(waveform, threshold=0.98)
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
                "section_rms": section_rms,
                "target_rms": 0.2,
            },
            "sample_rate": sample_rate,
        }
        if plan.theme is not None:
            extras["theme"] = plan.theme.model_dump()

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
        plan: CompositionPlan,
        index: int,
        section: CompositionSection,
        total_sections: int,
    ) -> float:
        tempo = max(plan.tempo_bpm, 1)
        seconds_per_beat = 60.0 / tempo
        interior_padding = seconds_per_beat * 1.25
        edge_padding = seconds_per_beat * 0.75
        padding = edge_padding if index in (0, total_sections - 1) else interior_padding
        padding = float(min(1.5, max(0.35, padding)))
        return max(float(section.target_seconds) + padding, float(section.target_seconds))

    def _prepare_section_waveforms(
        self,
        plan: CompositionPlan,
        renders: List[SectionRender],
    ) -> Tuple[List[np.ndarray], int, List[float]]:
        if not renders:
            raise RuntimeError("composition produced no sections")
        prepared: List[np.ndarray] = []
        loudness: List[float] = []
        sample_rate = renders[0].sample_rate
        tempo = max(plan.tempo_bpm, 1)
        for section, render in zip(plan.sections, renders, strict=True):
            if render.sample_rate != sample_rate:
                raise RuntimeError("section sample rates diverged")
            waveform = ensure_waveform_channels(render.waveform)
            target_seconds = float(section.target_seconds)
            target_samples = max(1, int(round(target_seconds * sample_rate)))
            fitted = fit_to_length(waveform, target_samples, sample_rate, tempo_bpm=tempo)
            loudness.append(rms_level(fitted))
            prepared.append(fitted)
        if len(prepared) != len(plan.sections):
            raise RuntimeError("prepared waveforms do not match plan sections")
        return prepared, sample_rate, loudness

    def _combine_sections(
        self,
        plan: CompositionPlan,
        waveforms: List[np.ndarray],
        sample_rate: int,
    ) -> Tuple[np.ndarray, List[Dict[str, object]]]:
        if not waveforms:
            raise RuntimeError("composition produced no sections")

        combined = ensure_waveform_channels(waveforms[0])
        crossfade_records: List[Dict[str, object]] = []

        for index, waveform in enumerate(waveforms[1:], start=1):
            left = combined
            right = ensure_waveform_channels(waveform)
            fade_seconds = self._crossfade_seconds(plan, left, right, sample_rate)
            fade_samples = int(max(1, round(sample_rate * fade_seconds)))
            combined = crossfade_append(left, right, fade_samples)
            crossfade_records.append(
                {
                    "from": plan.sections[index - 1].section_id,
                    "to": plan.sections[index].section_id,
                    "seconds": fade_seconds,
                }
            )

        return combined.astype(np.float32), crossfade_records

    def _crossfade_seconds(
        self,
        plan: CompositionPlan,
        left: np.ndarray,
        right: np.ndarray,
        sample_rate: int,
    ) -> float:
        left_duration = left.shape[0] / sample_rate
        right_duration = right.shape[0] / sample_rate
        seconds_per_beat = 60.0 / max(plan.tempo_bpm, 1)
        base = min(left_duration, right_duration) * 0.35
        base = max(base, seconds_per_beat * 0.5)
        base = min(base, seconds_per_beat * 1.5)
        return float(max(0.1, min(base, 1.5)))
