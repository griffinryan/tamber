"""Rule-based composition planner producing structured section plans."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, List

from ..app.models import (
    CompositionPlan,
    CompositionSection,
    GenerationRequest,
    SectionEnergy,
    SectionRole,
)

PLAN_VERSION = "v1"
DEFAULT_TIME_SIGNATURE = "4/4"
MIN_TEMPO = 60
MAX_TEMPO = 140
MIN_SECTION_SECONDS = 5.0


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
        seconds_total = float(max(request.duration_seconds, 4))
        templates = _prune_templates(seconds_total, templates)
        total_bars = sum(template.bars for template in templates)
        seconds_per_bar = seconds_total / total_bars

        raw_tempo = round(240.0 * total_bars / seconds_total)
        tempo_bpm = max(MIN_TEMPO, min(MAX_TEMPO, int(raw_tempo)))

        base_seed = (
            request.seed if request.seed is not None else _deterministic_seed(request.prompt)
        )
        musical_key = _select_key(base_seed)

        sections: List[CompositionSection] = []
        accumulated = 0.0

        for index, template in enumerate(templates):
            section_seconds = template.bars * seconds_per_bar
            if index == len(templates) - 1:
                planned_total = accumulated + section_seconds
                section_seconds += seconds_total - planned_total

            prompt_text = template.prompt_template.format(prompt=request.prompt).strip()

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
            accumulated += section_seconds

        return CompositionPlan(
            version=PLAN_VERSION,
            tempo_bpm=tempo_bpm,
            time_signature=DEFAULT_TIME_SIGNATURE,
            key=musical_key,
            total_bars=total_bars,
            total_duration_seconds=accumulated,
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


def _prune_templates(
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
