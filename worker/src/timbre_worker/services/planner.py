"""Rule-based composition planner producing structured section plans."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable, List

from ..app.models import (
    CompositionPlan,
    CompositionSection,
    GenerationRequest,
    SectionEnergy,
    SectionRole,
)

PLAN_VERSION = "v2"
DEFAULT_TIME_SIGNATURE = "4/4"
MIN_TEMPO = 60
MAX_TEMPO = 140
MIN_SECTION_SECONDS = 5.0
MIN_TOTAL_SECONDS = 12.0

ROLE_MIN_BEATS = {
    SectionRole.INTRO: 4,
    SectionRole.MOTIF: 8,
    SectionRole.DEVELOPMENT: 8,
    SectionRole.BRIDGE: 4,
    SectionRole.RESOLUTION: 4,
    SectionRole.OUTRO: 4,
}

ADD_PRIORITY = [
    SectionRole.MOTIF,
    SectionRole.DEVELOPMENT,
    SectionRole.BRIDGE,
    SectionRole.RESOLUTION,
    SectionRole.INTRO,
    SectionRole.OUTRO,
]

REMOVE_PRIORITY = [
    SectionRole.OUTRO,
    SectionRole.INTRO,
    SectionRole.BRIDGE,
    SectionRole.RESOLUTION,
    SectionRole.DEVELOPMENT,
    SectionRole.MOTIF,
]


@dataclass(frozen=True)
class SectionTemplate:
    role: SectionRole
    label: str
    energy: SectionEnergy
    bars: int
    prompt_template: str
    transition: str | None = None


class CompositionPlanner:
    """Generates deterministic composition plans from prompts."""

    def build_plan(self, request: GenerationRequest) -> CompositionPlan:
        templates = list(_select_templates(request.duration_seconds))
        seconds_total = float(max(request.duration_seconds, MIN_TOTAL_SECONDS))
        templates = _prune_templates_by_capacity(seconds_total, templates)
        raw_bars = sum(template.bars for template in templates)
        raw_tempo = int(round(240.0 * raw_bars / seconds_total)) if seconds_total > 0 else 90
        tempo_bpm = _select_tempo(raw_tempo)
        seconds_per_beat = 60.0 / tempo_bpm if tempo_bpm > 0 else 0.5

        base_seed = (
            request.seed if request.seed is not None else _deterministic_seed(request.prompt)
        )
        musical_key = _select_key(base_seed)

        beats = _allocate_beats(templates, seconds_total, seconds_per_beat)
        total_bars = max(1, int(round(sum(beats) / 4)))
        seconds_per_section = [beat_count * seconds_per_beat for beat_count in beats]

        sections: List[CompositionSection] = []

        for index, template in enumerate(templates):
            prompt_text = template.prompt_template.format(prompt=request.prompt).strip()

            section_seconds = max(MIN_SECTION_SECONDS, seconds_per_section[index])
            sections.append(
                CompositionSection(
                    section_id=f"s{index:02d}",
                    role=template.role,
                    label=template.label,
                    prompt=prompt_text,
                    bars=template.bars,
                    target_seconds=max(2.0, section_seconds),
                    energy=template.energy,
                    model_id=None,
                    seed_offset=index,
                    transition=template.transition,
                )
            )

        total_duration = sum(section.target_seconds for section in sections)

        return CompositionPlan(
            version=PLAN_VERSION,
            tempo_bpm=tempo_bpm,
            time_signature=DEFAULT_TIME_SIGNATURE,
            key=musical_key,
            total_bars=total_bars,
            total_duration_seconds=total_duration,
            sections=sections,
        )

def _deterministic_seed(prompt: str) -> int:
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _select_key(seed: int) -> str:
    keys = [
        "C major",
        "G major",
        "D major",
        "A major",
        "E major",
        "B major",
        "F major",
        "E minor",
        "A minor",
        "D minor",
        "G minor",
        "C minor",
    ]
    return keys[seed % len(keys)]


def _select_tempo(raw_tempo: int) -> int:
    clamped = max(MIN_TEMPO, min(MAX_TEMPO, raw_tempo))
    best = clamped
    best_error = abs(raw_tempo - best)
    for delta in range(1, 16):
        for candidate in (clamped - delta, clamped + delta):
            if candidate < MIN_TEMPO or candidate > MAX_TEMPO:
                continue
            error = abs(raw_tempo - candidate)
            if error < best_error:
                best = candidate
                best_error = error
    return best


def _prune_templates_by_capacity(
    seconds_total: float,
    templates: List[SectionTemplate],
) -> List[SectionTemplate]:
    if not templates:
        return templates

    pruned = list(templates)
    while len(pruned) > 1 and seconds_total < MIN_SECTION_SECONDS * len(pruned):
        drop_index = _template_to_drop(pruned)
        pruned.pop(drop_index)
    return pruned


def _template_to_drop(templates: List[SectionTemplate]) -> int:
    role_priority = {
        SectionRole.OUTRO: 0,
        SectionRole.INTRO: 1,
        SectionRole.BRIDGE: 2,
        SectionRole.RESOLUTION: 3,
        SectionRole.DEVELOPMENT: 4,
        SectionRole.MOTIF: 5,
    }
    best_idx = 0
    best_key = (
        role_priority.get(templates[0].role, 10),
        templates[0].bars,
        0,
    )
    for index, template in enumerate(templates[1:], start=1):
        key = (
            role_priority.get(template.role, 10),
            template.bars,
            index,
        )
        if key < best_key:
            best_idx = index
            best_key = key
    return best_idx


def _role_min_beats(role: SectionRole, seconds_per_beat: float) -> int:
    role_min = ROLE_MIN_BEATS.get(role, 4)
    min_seconds_beats = math.ceil(MIN_SECTION_SECONDS / seconds_per_beat) if seconds_per_beat > 0 else 4
    return int(max(role_min, min_seconds_beats))


def _allocate_beats(
    templates: List[SectionTemplate],
    seconds_total: float,
    seconds_per_beat: float,
) -> List[int]:
    if not templates:
        return []

    seconds_per_beat = max(seconds_per_beat, 1e-3)
    total_beats_raw = sum(template.bars * 4 for template in templates)
    min_beats = [_role_min_beats(template.role, seconds_per_beat) for template in templates]
    target_beats = max(
        sum(min_beats),
        max(1, int(round(seconds_total / seconds_per_beat))),
    )

    total_beats_raw = max(total_beats_raw, 1)
    scale = target_beats / total_beats_raw

    provisional = []
    for template, min_count in zip(templates, min_beats, strict=True):
        scaled = int(round(template.bars * 4 * scale))
        provisional.append(max(min_count, max(1, scaled)))

    beats = _rebalance_beats(templates, provisional, min_beats, target_beats)
    return beats


def _rebalance_beats(
    templates: List[SectionTemplate],
    beats: List[int],
    min_beats: List[int],
    target_beats: int,
) -> List[int]:
    total = sum(beats)
    if total == target_beats:
        return beats

    role_to_indices = {}
    for index, template in enumerate(templates):
        role_to_indices.setdefault(template.role, []).append(index)

    def _expand_priority(priority: List[SectionRole]) -> List[int]:
        order: List[int] = []
        for role in priority:
            order.extend(role_to_indices.get(role, []))
        if len(order) < len(beats):
            remaining = [idx for idx in range(len(beats)) if idx not in order]
            order.extend(remaining)
        return order

    add_order = _expand_priority(ADD_PRIORITY)
    remove_order = _expand_priority(REMOVE_PRIORITY)

    safety = 0
    while total < target_beats and safety < 4096:
        modified = False
        for idx in add_order:
            beats[idx] += 1
            total += 1
            modified = True
            if total >= target_beats:
                break
        if not modified:
            break
        safety += 1

    safety = 0
    while total > target_beats and safety < 4096:
        modified = False
        for idx in remove_order:
            if beats[idx] > min_beats[idx]:
                beats[idx] -= 1
                total -= 1
                modified = True
                if total <= target_beats:
                    break
        if not modified:
            break
        safety += 1

    return beats


def _select_templates(duration_seconds: int) -> Iterable[SectionTemplate]:
    if duration_seconds >= 24:
        yield SectionTemplate(
            role=SectionRole.INTRO,
            label="Arrival",
            energy=SectionEnergy.LOW,
            bars=4,
            prompt_template="Set the stage with a hushed texture hinting at {prompt}.",
            transition="Fade in layers",
        )
        yield SectionTemplate(
            role=SectionRole.MOTIF,
            label="Statement",
            energy=SectionEnergy.MEDIUM,
            bars=8,
            prompt_template="Introduce a memorable motif expressing {prompt}.",
            transition="Build momentum",
        )
        yield SectionTemplate(
            role=SectionRole.DEVELOPMENT,
            label="Development",
            energy=SectionEnergy.HIGH,
            bars=8,
            prompt_template=(
                "Expand the motif with syncopation and evolving harmony around {prompt}."
            ),
            transition="Evolve harmonies",
        )
        yield SectionTemplate(
            role=SectionRole.BRIDGE,
            label="Bridge",
            energy=SectionEnergy.MEDIUM,
            bars=4,
            prompt_template="Introduce contrast with modal colors echoing {prompt}.",
            transition="Prepare resolution",
        )
        yield SectionTemplate(
            role=SectionRole.RESOLUTION,
            label="Resolution",
            energy=SectionEnergy.MEDIUM,
            bars=4,
            prompt_template="Resolve tension with satisfying cadences fulfilling {prompt}.",
            transition="Return home",
        )
        yield SectionTemplate(
            role=SectionRole.OUTRO,
            label="Release",
            energy=SectionEnergy.LOW,
            bars=4,
            prompt_template="Let the textures dissolve, echoing the atmosphere of {prompt}.",
            transition="Fade to silence",
        )
        return

    if duration_seconds >= 16:
        yield SectionTemplate(
            role=SectionRole.INTRO,
            label="Lead-in",
            energy=SectionEnergy.LOW,
            bars=4,
            prompt_template="Open gently and establish the mood of {prompt}.",
            transition="Invite motif",
        )
        yield SectionTemplate(
            role=SectionRole.MOTIF,
            label="Motif A",
            energy=SectionEnergy.MEDIUM,
            bars=8,
            prompt_template="Present a clear melodic phrase inspired by {prompt}.",
            transition="Increase energy",
        )
        yield SectionTemplate(
            role=SectionRole.DEVELOPMENT,
            label="Variation",
            energy=SectionEnergy.HIGH,
            bars=6,
            prompt_template=(
                "Develop the motif with rhythmic motion and harmonic color tied to {prompt}."
            ),
            transition="Soften textures",
        )
        yield SectionTemplate(
            role=SectionRole.RESOLUTION,
            label="Cadence",
            energy=SectionEnergy.MEDIUM,
            bars=4,
            prompt_template="Resolve back to calm, letting {prompt} settle.",
            transition="Release",
        )
        yield SectionTemplate(
            role=SectionRole.OUTRO,
            label="Tail",
            energy=SectionEnergy.LOW,
            bars=2,
            prompt_template="Conclude with a gentle echo of {prompt}.",
            transition="Fade",
        )
        return

    yield SectionTemplate(
        role=SectionRole.INTRO,
        label="Intro",
        energy=SectionEnergy.LOW,
        bars=2,
        prompt_template="Set a delicate entrance referencing {prompt}.",
        transition="Introduce motif",
    )
    yield SectionTemplate(
        role=SectionRole.MOTIF,
        label="Motif",
        energy=SectionEnergy.MEDIUM,
        bars=6,
        prompt_template="Deliver a simple, emotive motif capturing {prompt}.",
        transition="Lift energy",
    )
    yield SectionTemplate(
        role=SectionRole.RESOLUTION,
        label="Resolve",
        energy=SectionEnergy.MEDIUM,
        bars=4,
        prompt_template="Resolve with warm chords that fulfill {prompt}.",
        transition="Fade out",
    )
