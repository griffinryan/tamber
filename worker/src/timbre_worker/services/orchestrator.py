"""High-level composition orchestrator coordinating backends."""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Dict, Iterable, List, Optional, Tuple

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
from .types import BackendStatus, SectionPhrase, SectionRender, SectionTrack


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
        self._backend_status: Dict[str, BackendStatus] = {}

    async def warmup(
        self,
        plan: Optional[CompositionPlan] = None,
        model_hint: Optional[str] = None,
    ) -> Dict[str, BackendStatus]:
        tokens = self._collect_backend_tokens(plan=plan, model_hint=model_hint)
        if plan is None and model_hint is None:
            tokens.update({"musicgen", "riffusion"})
        statuses = await self._warmup_backends(tokens)
        return statuses

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
        phrases = self._build_phrase_plan(plan)
        await self._warmup_backends(
            self._collect_backend_tokens(plan=plan, model_hint=request.model_id)
        )

        tracks: List[SectionTrack] = []
        total_sections = len(plan.sections)
        motif_seed_render: SectionRender | None = None
        motif_seed_section: CompositionSection | None = None
        for index, section in enumerate(plan.sections):
            phrase = phrases[index]
            backend = self._select_backend_service(section.model_id or request.model_id or "")
            render_hint = phrase.duration_with_padding
            previous_render = tracks[-1].render if tracks else None
            logger.debug(
                "Rendering section %s via %s (phrase %.2fs, hint %.2fs)",
                section.section_id,
                backend.__class__.__name__,
                phrase.seconds,
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
            self._attach_phrase_metadata(render, phrase, render_hint)
            if motif_seed_render is None and self._is_motif_seed_section(section):
                motif_seed_render = render
                motif_seed_section = section
            conditioning_tail, conditioning_rate = self._conditioning_tail(render, phrase)
            track = SectionTrack(
                section_id=section.section_id,
                phrase=phrase,
                render=render,
                backend=render.extras.get("backend", backend.__class__.__name__.lower()),
                conditioning_tail=conditioning_tail,
                conditioning_rate=conditioning_rate,
            )
            tracks.append(track)
            if progress_cb is not None:
                await progress_cb(index + 1, len(plan.sections), render)

        motif_metadata = self._finalise_motif_seed(
            job_id, plan, motif_seed_section, motif_seed_render
        )

        trimmed, sample_rate, section_rms = self._prepare_section_waveforms(plan, tracks)
        waveform, crossfades = self._combine_sections(plan, trimmed, sample_rate, tracks)
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
                track.render.extras.get("placeholder", False) for track in tracks
            ),
            "sections": [track.render.extras for track in tracks],
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

    def _attach_phrase_metadata(
        self,
        render: SectionRender,
        phrase: SectionPhrase,
        render_hint: float,
    ) -> None:
        payload = render.extras.setdefault("phrase", {})
        payload.update(
            {
                "seconds": phrase.seconds,
                "beats": phrase.beats,
                "bars": phrase.bars,
                "tempo_bpm": phrase.tempo_bpm,
                "seconds_per_beat": phrase.seconds_per_beat,
                "padding_seconds": phrase.padding_seconds,
                "conditioning_tail_seconds": phrase.conditioning_tail_seconds,
                "render_hint_seconds": render_hint,
            }
        )

    def _conditioning_tail(
        self,
        render: SectionRender,
        phrase: SectionPhrase,
    ) -> Tuple[Optional[np.ndarray], Optional[int]]:
        waveform = ensure_waveform_channels(render.waveform)
        sample_rate = render.sample_rate
        if waveform.size == 0 or sample_rate <= 0:
            return None, None
        tail_seconds = max(0.0, phrase.conditioning_tail_seconds)
        tail_samples = int(round(tail_seconds * sample_rate))
        if tail_samples <= 0:
            return None, sample_rate
        tail_samples = min(tail_samples, waveform.shape[0])
        tail = waveform[-tail_samples:].astype(np.float32, copy=True)
        if tail.size == 0:
            return None, sample_rate
        fade_samples = min(512, max(1, tail.shape[0] // 32))
        if fade_samples > 0 and tail.shape[0] >= fade_samples:
            fade = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
            if tail.ndim == 1:
                tail[:fade_samples] *= fade
            else:
                tail[:fade_samples] *= fade[:, None]
        return tail, sample_rate

    def _collect_backend_tokens(
        self,
        *,
        plan: Optional[CompositionPlan],
        model_hint: Optional[str],
    ) -> set[str]:
        tokens: set[str] = set()
        if plan is not None:
            for section in plan.sections:
                model_id = section.model_id or model_hint or self._settings.default_model_id
                tokens.add(self._backend_token(model_id))
        else:
            tokens.add(self._backend_token(model_hint or self._settings.default_model_id))
        return tokens

    def _backend_token(self, model_id: str) -> str:
        token = (model_id or "").lower()
        if "musicgen" in token or "audiocraft" in token or "facebook/musicgen" in token:
            return "musicgen"
        if "riffusion" in token:
            return "riffusion"
        if token == "composer":
            return "musicgen"
        return "musicgen"

    async def _warmup_backends(self, tokens: Iterable[str]) -> Dict[str, BackendStatus]:
        token_set = set(tokens)
        task_map: list[tuple[str, asyncio.Task[BackendStatus]]] = []
        if "musicgen" in token_set:
            task_map.append(("musicgen", asyncio.create_task(self._musicgen.warmup())))
        if "riffusion" in token_set:
            task_map.append(("riffusion", asyncio.create_task(self._riffusion.warmup())))

        statuses: Dict[str, BackendStatus] = {}
        if not task_map:
            return statuses

        for name, task in task_map:
            try:
                status = await task
            except Exception as exc:  # noqa: BLE001
                logger.exception("Backend %s warmup failed", name)
                status = BackendStatus(
                    name=name,
                    ready=False,
                    device=None,
                    dtype=None,
                    error=str(exc),
                )
            self._backend_status[name] = status
            statuses[name] = status
        logger.info(
            "Backend warmup status: %s",
            {name: status.ready for name, status in statuses.items()},
        )
        return statuses

    def backend_status(self) -> Dict[str, BackendStatus]:
        return dict(self._backend_status)

    def _build_phrase_plan(self, plan: CompositionPlan) -> List[SectionPhrase]:
        tempo = max(plan.tempo_bpm, 1)
        beats_per_bar = self._beats_per_bar(plan.time_signature)
        seconds_per_beat = 60.0 / tempo
        total_sections = len(plan.sections)
        phrases: List[SectionPhrase] = []
        for index, section in enumerate(plan.sections):
            beats = float(section.bars * beats_per_bar)
            base_seconds = beats * seconds_per_beat
            target_seconds = max(base_seconds, float(section.target_seconds))
            padding = self._phrase_padding_seconds(seconds_per_beat, index, total_sections)
            conditioning_tail = min(target_seconds, seconds_per_beat * 4.0)
            phrases.append(
                SectionPhrase(
                    section_id=section.section_id,
                    tempo_bpm=tempo,
                    bars=section.bars,
                    beats=beats,
                    seconds=target_seconds,
                    seconds_per_beat=seconds_per_beat,
                    padding_seconds=padding,
                    conditioning_tail_seconds=conditioning_tail,
                )
            )
        return phrases

    def _beats_per_bar(self, time_signature: str) -> int:
        try:
            numerator = int(time_signature.split("/")[0])
            return max(1, numerator)
        except Exception:  # noqa: BLE001
            return 4

    def _phrase_padding_seconds(
        self,
        seconds_per_beat: float,
        index: int,
        total_sections: int,
    ) -> float:
        interior_padding = seconds_per_beat * 1.25
        edge_padding = seconds_per_beat * 0.75
        padding = edge_padding if index in (0, max(total_sections - 1, 0)) else interior_padding
        return float(min(1.5, max(0.35, padding)))

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

    def _select_backend_service(self, model_id: str) -> object:
        token = (model_id or "").lower()
        if "musicgen" in token or "audiocraft" in token or "facebook/musicgen" in token:
            return self._musicgen
        return self._riffusion

    def _prepare_section_waveforms(
        self,
        plan: CompositionPlan,
        tracks: List[SectionTrack],
    ) -> Tuple[List[np.ndarray], int, List[float]]:
        if not tracks:
            raise RuntimeError("composition produced no sections")
        prepared: List[np.ndarray] = []
        loudness: List[float] = []
        sample_rate = tracks[0].render.sample_rate
        for section, track in zip(plan.sections, tracks, strict=True):
            render = track.render
            if render.sample_rate != sample_rate:
                raise RuntimeError("section sample rates diverged")
            waveform = ensure_waveform_channels(render.waveform)
            target_seconds = max(track.phrase.seconds, float(section.target_seconds))
            target_samples = max(1, int(round(target_seconds * sample_rate)))
            shaped = self._shape_to_target_length(waveform, target_samples, sample_rate)
            loudness.append(rms_level(shaped))
            prepared.append(shaped)
        if len(prepared) != len(plan.sections):
            raise RuntimeError("prepared waveforms do not match plan sections")
        return prepared, sample_rate, loudness

    def _combine_sections(
        self,
        plan: CompositionPlan,
        waveforms: List[np.ndarray],
        sample_rate: int,
        tracks: List[SectionTrack],
    ) -> Tuple[np.ndarray, List[Dict[str, object]]]:
        if not waveforms:
            raise RuntimeError("composition produced no sections")

        combined = ensure_waveform_channels(waveforms[0])
        crossfade_records: List[Dict[str, object]] = []

        for index, waveform in enumerate(waveforms[1:], start=1):
            left = combined
            right = ensure_waveform_channels(waveform)
            left_extras = tracks[index - 1].render.extras
            right_extras = tracks[index].render.extras
            fade_seconds = self._crossfade_seconds(
                plan,
                left,
                right,
                sample_rate,
                left_extras,
                right_extras,
            )
            if fade_seconds <= 0.0:
                combined = self._butt_join(left, right, sample_rate)
                crossfade_records.append(
                    {
                        "from": plan.sections[index - 1].section_id,
                        "to": plan.sections[index].section_id,
                        "seconds": 0.0,
                        "mode": "butt",
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
            else:
                fade_samples = int(max(1, round(sample_rate * fade_seconds)))
                combined = crossfade_append(left, right, fade_samples)
                crossfade_records.append(
                    {
                        "from": plan.sections[index - 1].section_id,
                        "to": plan.sections[index].section_id,
                        "seconds": fade_seconds,
                        "mode": "crossfade",
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
        seconds_per_beat = 60.0 / max(plan.tempo_bpm, 1)
        left_conditioned = bool(left_extras.get("audio_conditioning_applied"))
        right_conditioned = bool(right_extras.get("audio_conditioning_applied"))
        placeholder = bool(left_extras.get("placeholder")) or bool(
            right_extras.get("placeholder")
        )
        if placeholder:
            return float(min(seconds_per_beat * 0.5, 0.35))
        if not (left_conditioned and right_conditioned):
            return float(min(seconds_per_beat * 0.3, 0.15))
        return 0.0

    def _butt_join(
        self,
        left: np.ndarray,
        right: np.ndarray,
        sample_rate: int,
    ) -> np.ndarray:
        left_data = ensure_waveform_channels(left)
        right_data = ensure_waveform_channels(right)
        if left_data.size == 0:
            return right_data.astype(np.float32)
        if right_data.size == 0:
            return left_data.astype(np.float32)
        fade_samples = min(int(round(sample_rate * 0.01)), left_data.shape[0], right_data.shape[0])
        if fade_samples <= 0:
            return np.concatenate([left_data, right_data], axis=0).astype(np.float32)
        envelope_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)[:, None]
        envelope_in = 1.0 - envelope_out
        overlap = left_data[-fade_samples:] * envelope_out + right_data[:fade_samples] * envelope_in
        joined = np.concatenate(
            [left_data[:-fade_samples], overlap, right_data[fade_samples:]],
            axis=0,
        )
        return joined.astype(np.float32)

    def _shape_to_target_length(
        self,
        waveform: np.ndarray,
        target_samples: int,
        sample_rate: int,
    ) -> np.ndarray:
        data = ensure_waveform_channels(waveform).astype(np.float32)
        if target_samples <= 0:
            return np.zeros((0, data.shape[1]), dtype=np.float32)
        if data.shape[0] > target_samples:
            trimmed = data[:target_samples].copy()
            fade_samples = min(int(round(sample_rate * 0.02)), trimmed.shape[0])
            if fade_samples > 0:
                fade = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)[:, None]
                trimmed[-fade_samples:] *= fade
            return trimmed
        if data.shape[0] < target_samples:
            fade_samples = min(int(round(sample_rate * 0.02)), data.shape[0])
            shaped = data.copy()
            if fade_samples > 0:
                fade = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)[:, None]
                shaped[-fade_samples:] *= fade
            pad = np.zeros((target_samples - shaped.shape[0], shaped.shape[1]), dtype=np.float32)
            return np.vstack((shaped, pad))
        return data
