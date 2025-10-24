"""High-level composition orchestrator coordinating backends."""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from ..app.models import (
    CompositionPlan,
    CompositionSection,
    GenerationArtifact,
    GenerationMetadata,
    GenerationRequest,
    SectionRole,
)
from ..app.settings import Settings
from .audio_utils import (
    crossfade_append,
    ensure_waveform_channels,
    fit_to_length,
    normalise_loudness,
    resample_waveform,
    rms_level,
    soft_limiter,
    tilt_highs,
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
        motif_seed_render: SectionRender | None = None
        motif_seed_section: CompositionSection | None = None
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
                motif_seed=motif_seed_render,
            )
            if motif_seed_render is None and self._is_motif_seed_section(section):
                motif_seed_render = render
                motif_seed_section = section
            renders.append(render)
            if progress_cb is not None:
                await progress_cb(index + 1, len(plan.sections), render)

        motif_metadata = self._finalise_motif_seed(
            job_id, plan, motif_seed_section, motif_seed_render
        )

        trimmed, sample_rate, section_rms = self._prepare_section_waveforms(plan, renders)
        waveform, crossfades = self._combine_sections(plan, trimmed, sample_rate, renders)
        waveform = normalise_loudness(waveform, target_rms=0.2)
        waveform = tilt_highs(waveform, sample_rate)
        waveform = soft_limiter(waveform, threshold=0.98)

        export_rate = self._settings.export_sample_rate
        if export_rate and sample_rate != export_rate:
            waveform = resample_waveform(waveform, sample_rate, export_rate)
            sample_rate = export_rate
            waveform = soft_limiter(waveform, threshold=0.98)

        if mix_cb is not None:
            await mix_cb(len(waveform) / sample_rate)

        artifact_path = self._artifact_root / f"{job_id}.wav"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        write_waveform(
            artifact_path,
            waveform,
            sample_rate,
            bit_depth=self._settings.export_bit_depth,
            audio_format=self._settings.export_format,
        )

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
        if motif_metadata is not None:
            extras["motif_seed"] = motif_metadata
        else:
            extras["motif_seed"] = {"captured": False}
        extras["bit_depth"] = self._settings.export_bit_depth
        extras["format"] = self._settings.export_format
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

    def _is_motif_seed_section(self, section: CompositionSection) -> bool:
        directive = (section.motif_directive or "").lower()
        if "state motif" in directive:
            return True
        return section.role == SectionRole.MOTIF

    def _finalise_motif_seed(
        self,
        job_id: str,
        plan: CompositionPlan,
        section: CompositionSection | None,
        render: SectionRender | None,
    ) -> Dict[str, object] | None:
        if section is None or render is None:
            return None
        try:
            motif_path = self._artifact_root / f"{job_id}_motif.wav"
            motif_path.parent.mkdir(parents=True, exist_ok=True)
            write_waveform(
                motif_path,
                render.waveform,
                render.sample_rate,
                bit_depth="pcm16",
                audio_format="wav",
            )
            features = self._compute_motif_features(
                render.waveform, render.sample_rate, plan.key
            )
            payload: Dict[str, object] = {
                "captured": True,
                "path": str(motif_path),
                "section_id": section.section_id,
                "section_label": section.label,
                "section_role": section.role.value,
                "sample_rate": render.sample_rate,
            }
            payload.update(features)
            if plan.theme is not None:
                payload["motif_text"] = plan.theme.motif
                payload["motif_rhythm"] = plan.theme.rhythm
            return payload
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to capture motif seed for job %s (%s)", job_id, exc
            )
            return {"captured": False, "reason": f"error:{exc.__class__.__name__}"}

    def _compute_motif_features(
        self, waveform: np.ndarray, sample_rate: int, plan_key: str
    ) -> Dict[str, object]:
        data = ensure_waveform_channels(waveform)
        if data.ndim == 1:
            mono = data.astype(np.float64)
        else:
            mono = data.mean(axis=1).astype(np.float64)
        duration = float(len(mono) / max(sample_rate, 1))
        rms = float(rms_level(data))
        centroid = self._spectral_centroid(mono, sample_rate)
        chroma = self._chroma_vector(mono, sample_rate)
        dominant_index = int(np.argmax(chroma)) if chroma.sum() > 0 else None
        pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        features: Dict[str, object] = {
            "duration_seconds": duration,
            "rms": rms,
            "spectral_centroid_hz": centroid,
            "chroma_vector": chroma.tolist(),
        }
        if dominant_index is not None:
            features["dominant_pitch_class"] = pitch_names[dominant_index]
        key_pc = self._pitch_class_from_key(plan_key)
        if key_pc is not None and chroma.sum() > 0:
            features["plan_key_alignment"] = float(chroma[key_pc])
        return features

    def _spectral_centroid(self, mono: np.ndarray, sample_rate: int) -> float:
        if mono.size == 0:
            return 0.0
        window = np.hanning(mono.size)
        if np.allclose(window.sum(), 0.0):
            window = np.ones_like(mono)
        spectrum = np.abs(np.fft.rfft(mono * window)) ** 2
        freqs = np.fft.rfftfreq(mono.size, d=1.0 / max(sample_rate, 1))
        energy = np.sum(spectrum)
        if energy <= 0.0:
            return 0.0
        return float(np.sum(freqs * spectrum) / energy)

    def _chroma_vector(self, mono: np.ndarray, sample_rate: int) -> np.ndarray:
        if mono.size == 0:
            return np.zeros(12, dtype=np.float64)
        window = np.hanning(mono.size)
        if np.allclose(window.sum(), 0.0):
            window = np.ones_like(mono)
        spectrum = np.abs(np.fft.rfft(mono * window)) ** 2
        freqs = np.fft.rfftfreq(mono.size, d=1.0 / max(sample_rate, 1))
        chroma = np.zeros(12, dtype=np.float64)
        for freq, energy in zip(freqs, spectrum):
            if freq <= 0.0:
                continue
            if freq < 27.5:
                continue
            pitch = 69.0 + 12.0 * np.log2(freq / 440.0)
            if not np.isfinite(pitch):
                continue
            pitch_class = int(round(pitch)) % 12
            chroma[pitch_class] += float(energy)
        total = float(np.sum(chroma))
        if total > 0.0:
            chroma /= total
        return chroma

    def _pitch_class_from_key(self, key: str) -> int | None:
        if not key:
            return None
        token = key.split()[0].strip()
        token = token.replace("♯", "#").replace("♭", "b").capitalize()
        mapping = {
            "C": 0,
            "C#": 1,
            "Db": 1,
            "D": 2,
            "D#": 3,
            "Eb": 3,
            "E": 4,
            "F": 5,
            "F#": 6,
            "Gb": 6,
            "G": 7,
            "G#": 8,
            "Ab": 8,
            "A": 9,
            "A#": 10,
            "Bb": 10,
            "B": 11,
            "Cb": 11,
        }
        return mapping.get(token)

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
        renders: List[SectionRender],
    ) -> Tuple[np.ndarray, List[Dict[str, object]]]:
        if not waveforms:
            raise RuntimeError("composition produced no sections")

        combined = ensure_waveform_channels(waveforms[0])
        crossfade_records: List[Dict[str, object]] = []

        for index, waveform in enumerate(waveforms[1:], start=1):
            left = combined
            right = ensure_waveform_channels(waveform)
            left_extras = renders[index - 1].extras
            right_extras = renders[index].extras
            fade_seconds = self._crossfade_seconds(
                plan,
                left,
                right,
                sample_rate,
                left_extras,
                right_extras,
            )
            fade_samples = int(max(1, round(sample_rate * fade_seconds)))
            combined = crossfade_append(left, right, fade_samples)
            crossfade_records.append(
                {
                    "from": plan.sections[index - 1].section_id,
                    "to": plan.sections[index].section_id,
                    "seconds": fade_seconds,
                    "conditioning": {
                        "left_audio_conditioned": bool(
                            left_extras.get("audio_conditioning_applied")
                        ),
                        "right_audio_conditioned": bool(
                            right_extras.get("audio_conditioning_applied")
                        ),
                        "left_placeholder": bool(left_extras.get("placeholder")),
                        "right_placeholder": bool(right_extras.get("placeholder")),
                    },
                }
            )

        return combined.astype(np.float32), crossfade_records

    def _crossfade_seconds(
        self,
        plan: CompositionPlan,
        left: np.ndarray,
        right: np.ndarray,
        sample_rate: int,
        left_extras: Dict[str, object],
        right_extras: Dict[str, object],
    ) -> float:
        left_duration = left.shape[0] / sample_rate
        right_duration = right.shape[0] / sample_rate
        seconds_per_beat = 60.0 / max(plan.tempo_bpm, 1)
        base = min(left_duration, right_duration) * 0.35
        base = max(base, seconds_per_beat * 0.5)
        base = min(base, seconds_per_beat * 1.5)

        left_conditioned = bool(left_extras.get("audio_conditioning_applied"))
        right_conditioned = bool(right_extras.get("audio_conditioning_applied"))
        placeholder = bool(left_extras.get("placeholder")) or bool(
            right_extras.get("placeholder")
        )

        conditioning_factor = 1.0
        if left_conditioned and right_conditioned:
            conditioning_factor *= 0.7
        elif left_conditioned or right_conditioned:
            conditioning_factor *= 0.85
        else:
            conditioning_factor *= 1.1
        if placeholder:
            conditioning_factor *= 1.3

        fade = base * conditioning_factor
        return float(max(0.08, min(fade, 1.8)))
