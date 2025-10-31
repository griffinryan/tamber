"""Rule-based composition planner producing structured section plans."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from ..app.models import (
    ClipLayer,
    CompositionPlan,
    CompositionSection,
    GenerationMode,
    GenerationRequest,
    SectionEnergy,
    SectionOrchestration,
    SectionRole,
    ThemeDescriptor,
)

PLAN_VERSION = "v3"
CLIP_PLAN_VERSION = "v4"
DEFAULT_TIME_SIGNATURE = "4/4"
MIN_TEMPO = 68
MAX_TEMPO = 128
SHORT_MIN_TOTAL_SECONDS = 2.0
SHORT_MIN_SECTION_SECONDS = 2.0
LONG_FORM_THRESHOLD = 90.0
LONG_MIN_SECTION_SECONDS = 16.0
MOTIF_MAX_TOTAL_SECONDS = 24.0

ADD_PRIORITY = [
    SectionRole.CHORUS,
    SectionRole.MOTIF,
    SectionRole.BRIDGE,
    SectionRole.DEVELOPMENT,
    SectionRole.INTRO,
    SectionRole.OUTRO,
    SectionRole.RESOLUTION,
]

REMOVE_PRIORITY = [
    SectionRole.OUTRO,
    SectionRole.INTRO,
    SectionRole.BRIDGE,
    SectionRole.RESOLUTION,
    SectionRole.DEVELOPMENT,
    SectionRole.MOTIF,
    SectionRole.CHORUS,
]

_LEXICON_PATH = Path(__file__).resolve().parents[4] / "planner" / "lexicon.json"

_ROLE_LOOKUP = {
    "intro": SectionRole.INTRO,
    "motif": SectionRole.MOTIF,
    "chorus": SectionRole.CHORUS,
    "development": SectionRole.DEVELOPMENT,
    "bridge": SectionRole.BRIDGE,
    "resolution": SectionRole.RESOLUTION,
    "outro": SectionRole.OUTRO,
}

_ENERGY_LOOKUP = {
    "low": SectionEnergy.LOW,
    "medium": SectionEnergy.MEDIUM,
    "high": SectionEnergy.HIGH,
}


@dataclass(frozen=True)
class KeywordMapping:
    term: str
    descriptor: str
    folded: str


@dataclass(frozen=True)
class Defaults:
    instrumentation: list[str]
    texture: str


@dataclass(frozen=True)
class GenreLayers:
    rhythm: list[str]
    bass: list[str]
    harmony: list[str]
    lead: list[str]
    textures: list[str]
    vocals: list[str]


@dataclass(frozen=True)
class GenreProfileData:
    keywords: list[str]
    folded_keywords: list[str]
    instrumentation: list[str]
    rhythm: str | None
    texture: str | None
    layers: GenreLayers


@dataclass(frozen=True)
class SectionTemplateSpec:
    role: SectionRole
    label: str
    energy: SectionEnergy
    base_bars: int
    min_bars: int
    max_bars: int
    prompt_template: str
    transition: str | None = None

    def to_template(self) -> "SectionTemplate":
        return SectionTemplate(
            role=self.role,
            label=self.label,
            energy=self.energy,
            base_bars=self.base_bars,
            min_bars=self.min_bars,
            max_bars=self.max_bars,
            prompt_template=self.prompt_template,
            transition=self.transition,
        )


@dataclass(frozen=True)
class TemplateVariant:
    min_duration: float
    sections: list[SectionTemplateSpec]


@dataclass(frozen=True)
class Lexicon:
    version: int
    instrument_keywords: list[KeywordMapping]
    rhythm_keywords: list[KeywordMapping]
    texture_keywords: list[KeywordMapping]
    defaults: Defaults
    genre_profiles: list[GenreProfileData]
    long_templates: list[TemplateVariant]
    short_templates: list[TemplateVariant]

    def select_long(self, duration_seconds: float) -> list["SectionTemplate"]:
        return [
            spec.to_template()
            for spec in self._select_variant(self.long_templates, duration_seconds)
        ]

    def select_short(self, duration_seconds: float) -> list["SectionTemplate"]:
        return [
            spec.to_template()
            for spec in self._select_variant(self.short_templates, duration_seconds)
        ]

    @staticmethod
    def _select_variant(
        variants: list[TemplateVariant],
        duration_seconds: float,
    ) -> list[SectionTemplateSpec]:
        for variant in variants:
            if duration_seconds >= variant.min_duration:
                return variant.sections
        return variants[-1].sections


def _casefold(value: str) -> str:
    return value.casefold()


def _build_keyword_mappings(entries: list[dict[str, str]]) -> list[KeywordMapping]:
    return [
        KeywordMapping(
            term=entry["term"],
            descriptor=entry["descriptor"],
            folded=_casefold(entry["term"]),
        )
        for entry in entries
    ]


def _build_variant(entry: dict) -> TemplateVariant:
    sections = []
    for section in entry["sections"]:
        role_key = section["role"].lower()
        energy_key = section["energy"].lower()
        try:
            role = _ROLE_LOOKUP[role_key]
        except KeyError as exc:  # pragma: no cover - configuration error
            raise ValueError(f"unknown section role '{section['role']}' in lexicon") from exc
        try:
            energy = _ENERGY_LOOKUP[energy_key]
        except KeyError as exc:  # pragma: no cover - configuration error
            raise ValueError(f"unknown section energy '{section['energy']}' in lexicon") from exc
        sections.append(
            SectionTemplateSpec(
                role=role,
                label=section["label"],
                energy=energy,
                base_bars=int(section["base_bars"]),
                min_bars=int(section["min_bars"]),
                max_bars=int(section["max_bars"]),
                prompt_template=section["prompt_template"],
                transition=section.get("transition"),
            )
        )
    return TemplateVariant(min_duration=float(entry["min_duration"]), sections=sections)


def _load_lexicon() -> Lexicon:
    try:
        raw = json.loads(_LEXICON_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:  # pragma: no cover - configuration error
        raise RuntimeError(f"planner lexicon file missing at {_LEXICON_PATH}") from exc

    instrument_keywords = _build_keyword_mappings(raw["keywords"]["instrument"])
    rhythm_keywords = _build_keyword_mappings(raw["keywords"]["rhythm"])
    texture_keywords = _build_keyword_mappings(raw["keywords"]["texture"])

    defaults = Defaults(
        instrumentation=list(raw["defaults"]["instrumentation"]),
        texture=raw["defaults"]["texture"],
    )

    genre_profiles: list[GenreProfileData] = []
    for profile in raw["genre_profiles"]:
        keywords = list(profile["keywords"])
        layers_raw = profile["layers"]
        layers = GenreLayers(
            rhythm=list(layers_raw["rhythm"]),
            bass=list(layers_raw["bass"]),
            harmony=list(layers_raw["harmony"]),
            lead=list(layers_raw["lead"]),
            textures=list(layers_raw["textures"]),
            vocals=list(layers_raw["vocals"]),
        )
        genre_profiles.append(
            GenreProfileData(
                keywords=keywords,
                folded_keywords=[_casefold(value) for value in keywords],
                instrumentation=list(profile["instrumentation"]),
                rhythm=profile.get("rhythm"),
                texture=profile.get("texture"),
                layers=layers,
            )
        )

    long_templates = [_build_variant(entry) for entry in raw["templates"]["long"]]
    long_templates.sort(key=lambda variant: variant.min_duration, reverse=True)

    short_templates = [_build_variant(entry) for entry in raw["templates"]["short"]]
    short_templates.sort(key=lambda variant: variant.min_duration, reverse=True)

    return Lexicon(
        version=int(raw["version"]),
        instrument_keywords=instrument_keywords,
        rhythm_keywords=rhythm_keywords,
        texture_keywords=texture_keywords,
        defaults=defaults,
        genre_profiles=genre_profiles,
        long_templates=long_templates,
        short_templates=short_templates,
    )


_LEXICON = _load_lexicon()

ENERGY_DYNAMIC_MAP = {
    SectionEnergy.LOW: ("gentle entrance", "soft release"),
    SectionEnergy.MEDIUM: ("steady motion", "measured resolve"),
    SectionEnergy.HIGH: ("climactic surge", "energetic peak"),
}

CATEGORY_KEYWORDS = {
    "rhythm": (
        "drum",
        "percussion",
        "beat",
        "groove",
        "rhythm",
        "kick",
        "hip hop",
        "boom bap",
        "house",
        "techno",
        "trance",
        "breakbeat",
        "trap",
        "tabla",
        "conga",
        "bongo",
        "drum machine",
        "brush",
        "double-kick",
    ),
    "bass": ("bass", "sub", "808", "low end", "low-end"),
    "harmony": (
        "piano",
        "keys",
        "synth",
        "pad",
        "string",
        "chord",
        "organ",
        "rhodes",
        "guitar",
        "harp",
        "mandolin",
        "banjo",
        "ukulele",
        "accordion",
    ),
    "lead": (
        "lead",
        "guitar",
        "solo",
        "brass",
        "sax",
        "horn",
        "trumpet",
        "violin",
        "viola",
        "fiddle",
        "flute",
        "clarinet",
        "oboe",
    ),
    "textures": (
        "ambient",
        "texture",
        "pad",
        "choir",
        "atmosphere",
        "reverb",
        "noise",
        "wash",
        "drone",
        "tape",
        "field",
    ),
    "vocals": ("vocal", "voice", "singer", "choir", "chant", "lyric"),
}

DEFAULT_LAYER_FALLBACKS = {
    "rhythm": [
        "tight drums",
        "organic percussion",
        "punchy live kit",
        "syncopated hand percussion",
        "four-on-the-floor kick",
        "shuffling brush kit",
        "driving tom groove",
    ],
    "bass": [
        "pulsing bass",
        "sub bass swell",
        "gritty electric bass",
        "round synth bass",
        "warm upright bass",
    ],
    "harmony": [
        "lush keys",
        "stacked synth pads",
        "shimmering guitar chords",
        "wide string beds",
        "layered plucked arps",
    ],
    "lead": [
        "expressive guitar lead",
        "soulful brass line",
        "soaring synth lead",
        "lyrical woodwind melody",
        "sparkling mallet motif",
    ],
    "textures": [
        "airy ambient swells",
        "granular noise beds",
        "glassy atmosphere",
        "crowd shimmer",
        "analog tape haze",
        "rolling field recordings",
    ],
    "vocals": [
        "wordless vocal pads",
        "ethereal choirs",
        "layered vocal oohs",
    ],
}


def _match_genre_profile(prompt_fold: str) -> Optional[GenreProfileData]:
    best: Optional[GenreProfileData] = None
    best_len = -1
    for profile in _LEXICON.genre_profiles:
        for keyword in profile.folded_keywords:
            if keyword in prompt_fold and len(keyword) > best_len:
                best = profile
                best_len = len(keyword)
    return best

SECTION_LAYER_PROFILE = {
    SectionRole.INTRO: {
        "rhythm": 1,
        "bass": 1,
        "harmony": 3,
        "lead": 1,
        "textures": 2,
        "vocals": 0,
    },
    SectionRole.MOTIF: {
        "rhythm": 2,
        "bass": 1,
        "harmony": 2,
        "lead": 1,
        "textures": 1,
        "vocals": 0,
    },
    SectionRole.CHORUS: {
        "rhythm": 2,
        "bass": 1,
        "harmony": 2,
        "lead": 2,
        "textures": 1,
        "vocals": 1,
    },
    SectionRole.BRIDGE: {
        "rhythm": 1,
        "bass": 1,
        "harmony": 2,
        "lead": 1,
        "textures": 2,
        "vocals": 1,
    },
    SectionRole.DEVELOPMENT: {
        "rhythm": 2,
        "bass": 1,
        "harmony": 2,
        "lead": 2,
        "textures": 2,
        "vocals": 1,
    },
    SectionRole.RESOLUTION: {
        "rhythm": 1,
        "bass": 1,
        "harmony": 2,
        "lead": 1,
        "textures": 2,
        "vocals": 1,
    },
    SectionRole.OUTRO: {
        "rhythm": 1,
        "bass": 1,
        "harmony": 2,
        "lead": 1,
        "textures": 2,
        "vocals": 0,
    },
    "default": {
        "rhythm": 2,
        "bass": 1,
        "harmony": 2,
        "lead": 1,
        "textures": 1,
        "vocals": 0,
    },
}


@dataclass(frozen=True)
class SectionTemplate:
    role: SectionRole
    label: str
    energy: SectionEnergy
    base_bars: int
    min_bars: int
    max_bars: int
    prompt_template: str
    transition: str | None = None


def _directives_for_role(
    role: SectionRole,
) -> tuple[Optional[str], list[str], Optional[str]]:
    if role == SectionRole.INTRO:
        return (
            "foreshadow motif",
            ["texture", "register preview"],
            "establish tonic pedal",
        )
    if role == SectionRole.MOTIF:
        return ("state motif", ["motif fidelity"], "open cadence")
    if role == SectionRole.CHORUS:
        return (
            "amplify motif",
            ["dynamics", "call-and-response", "countermelody"],
            "anthemic cadence",
        )
    if role == SectionRole.DEVELOPMENT:
        return ("develop motif", ["rhythm", "harmony", "counterpoint"], None)
    if role == SectionRole.BRIDGE:
        return ("modulate motif", ["harmony", "timbre"], "pivot modulation")
    if role == SectionRole.RESOLUTION:
        return ("resolve motif", ["harmony", "dynamics"], "authentic cadence")
    if role == SectionRole.OUTRO:
        return ("dissolve motif", ["texture", "space"], "fade tonic drone")
    return (None, [], None)


CLIP_LAYER_PROMPTS: dict[ClipLayer, str] = {
    ClipLayer.RHYTHM: "Focus on tight percussion and groove-locked drums with tasteful syncopation.",
    ClipLayer.BASS: "Deliver a supportive bassline that locks to the kick and outlines the harmony.",
    ClipLayer.HARMONY: "Layer warm harmonic stabs or pads that reinforce the progression without overcrowding.",
    ClipLayer.LEAD: "Craft a memorable lead motif that plays call-and-response with the established theme.",
    ClipLayer.TEXTURES: "Add evolving textures and atmosphere to widen the stereo field.",
    ClipLayer.VOCALS: "Improvise expressive wordless vocals floating above the arrangement.",
}


def _clip_orchestration(base: SectionOrchestration, layer: ClipLayer) -> SectionOrchestration:
    def clone(values: list[str], fallback: list[str]) -> list[str]:
        return list(values) if values else list(fallback)

    if layer == ClipLayer.RHYTHM:
        return SectionOrchestration(
            rhythm=clone(base.rhythm, ["tight kit", "syncopated percussion"]),
        )
    if layer == ClipLayer.BASS:
        return SectionOrchestration(
            bass=clone(base.bass, ["electric bass", "synth low-end"]),
        )
    if layer == ClipLayer.HARMONY:
        return SectionOrchestration(
            harmony=clone(base.harmony, ["chord stabs", "lush pads"]),
            textures=clone(base.textures, []),
        )
    if layer == ClipLayer.LEAD:
        return SectionOrchestration(
            lead=clone(base.lead, ["expressive lead", "hook synth"]),
            harmony=clone(base.harmony, []),
        )
    if layer == ClipLayer.TEXTURES:
        return SectionOrchestration(
            textures=clone(base.textures, ["ambient wash", "granular shimmer"]),
        )
    if layer == ClipLayer.VOCALS:
        return SectionOrchestration(
            vocals=clone(base.vocals, ["wordless vocal", "airy choir"]),
            textures=clone(base.textures, []),
        )
    return SectionOrchestration()


def _clip_prompt_text(
    clip_prompt: str,
    descriptor: ThemeDescriptor,
    layer: ClipLayer,
) -> str:
    base = clip_prompt.strip() if clip_prompt else descriptor.motif
    layer_line = CLIP_LAYER_PROMPTS.get(layer, "Add musical interest.")
    motif = descriptor.motif or "the established motif"
    return f"{base.strip()} {layer_line} Reinforce the motif \"{motif}\" with seamless looping."


def _clip_label(layer: ClipLayer) -> str:
    return f"{layer.value.title()} clip"


def _clip_energy(layer: ClipLayer) -> SectionEnergy:
    if layer in {ClipLayer.LEAD, ClipLayer.RHYTHM}:
        return SectionEnergy.HIGH
    if layer in {ClipLayer.HARMONY, ClipLayer.BASS}:
        return SectionEnergy.MEDIUM
    return SectionEnergy.LOW


def _clip_directive(layer: ClipLayer) -> str:
    mapping = {
        ClipLayer.RHYTHM: "drive motif groove",
        ClipLayer.BASS: "anchor harmonic floor",
        ClipLayer.HARMONY: "reinforce progression",
        ClipLayer.LEAD: "embellish motif",
        ClipLayer.TEXTURES: "expand atmosphere",
        ClipLayer.VOCALS: "float vocalise",
    }
    return mapping.get(layer, "develop motif")


class CompositionPlanner:
    """Generates deterministic composition plans from prompts."""

    def build_plan(self, request: GenerationRequest) -> CompositionPlan:
        if request.mode == GenerationMode.CLIP:
            raise RuntimeError("clip plans require session context")
        if request.mode == GenerationMode.MOTIF:
            return self._build_motif_plan(request)
        if float(request.duration_seconds) >= LONG_FORM_THRESHOLD:
            return self._build_long_form_plan(request)
        return self._build_short_form_plan(request)

    def build_clip_plan(
        self,
        *,
        seed_plan: CompositionPlan,
        session_theme: ThemeDescriptor | None,
        clip_prompt: str,
        layer: ClipLayer,
        bars: int,
    ) -> CompositionPlan:
        if bars <= 0:
            raise ValueError("clip bars must be positive")

        tempo_bpm = seed_plan.tempo_bpm
        time_signature = seed_plan.time_signature or DEFAULT_TIME_SIGNATURE
        beats_per_bar = _beats_per_bar(time_signature)
        seconds_per_bar = (60.0 / float(max(tempo_bpm, 1))) * float(max(beats_per_bar, 1))
        target_seconds = max(SHORT_MIN_SECTION_SECONDS, float(bars) * seconds_per_bar)

        descriptor = session_theme
        if descriptor is None:
            descriptor = ThemeDescriptor(
                motif=clip_prompt[:64] if clip_prompt else "session motif clip",
                instrumentation=[],
                rhythm="steady pulse",
                dynamic_curve=[],
                texture=None,
            )

        base_orchestration = (
            seed_plan.sections[0].orchestration
            if seed_plan.sections
            else SectionOrchestration()
        )
        orchestration = _clip_orchestration(base_orchestration, layer)

        prompt_text = _clip_prompt_text(clip_prompt, descriptor, layer)
        label = _clip_label(layer)
        energy = _clip_energy(layer)
        motif_directive = _clip_directive(layer)

        section = CompositionSection(
            section_id="c00",
            role=SectionRole.DEVELOPMENT,
            label=label,
            prompt=prompt_text,
            bars=bars,
            target_seconds=target_seconds,
            energy=energy,
            model_id=None,
            seed_offset=0,
            transition=None,
            motif_directive=motif_directive,
            variation_axes=["layer", layer.value],
            cadence_hint=None,
            orchestration=orchestration,
        )

        return CompositionPlan(
            version=CLIP_PLAN_VERSION,
            tempo_bpm=tempo_bpm,
            time_signature=time_signature,
            key=seed_plan.key,
            total_bars=bars,
            total_duration_seconds=target_seconds,
            theme=descriptor,
            sections=[section],
        )

    def _build_motif_plan(self, request: GenerationRequest) -> CompositionPlan:
        seconds_total = float(request.duration_seconds)
        seconds_total = max(SHORT_MIN_TOTAL_SECONDS, min(seconds_total, MOTIF_MAX_TOTAL_SECONDS))
        template = _select_motif_template()
        templates = [template]
        beats_per_bar = _beats_per_bar(DEFAULT_TIME_SIGNATURE)
        total_weight = max(template.base_bars, 1)
        if seconds_total > 0.0:
            raw_tempo = int(round(240.0 * float(total_weight) / seconds_total))
        else:
            raw_tempo = 90
        tempo_bpm = _select_tempo(raw_tempo)
        seconds_per_bar = (60.0 / float(max(tempo_bpm, 1))) * float(beats_per_bar)

        base_seed = request.seed if request.seed is not None else _deterministic_seed(request.prompt)
        prompt_fold = request.prompt.casefold()
        profile = _match_genre_profile(prompt_fold)

        descriptor = _build_theme_descriptor(
            request.prompt,
            prompt_fold,
            templates,
            profile,
            base_seed,
        )
        palette = _categorise_instrumentation(
            descriptor,
            prompt_fold,
            profile,
            base_seed,
        )
        palette_offsets = _palette_offsets(palette, base_seed)
        orchestrations = _plan_orchestrations(templates, palette, palette_offsets)
        musical_key = _select_key(base_seed)

        arrangement_text = _describe_orchestration(orchestrations[0])
        prompt_text = _render_prompt(
            template.prompt_template,
            prompt=request.prompt,
            descriptor=descriptor,
            section_index=0,
            arrangement=arrangement_text,
        ).strip()

        motif_directive, variation_axes, cadence_hint = _directives_for_role(template.role)

        if seconds_per_bar > 0.0:
            bar_count = int(round(seconds_total / seconds_per_bar))
        else:
            bar_count = template.base_bars
        bar_count = max(template.min_bars, min(template.max_bars, max(bar_count, 1)))
        target_seconds = max(SHORT_MIN_SECTION_SECONDS, bar_count * seconds_per_bar)

        section = CompositionSection(
            section_id="s00",
            role=template.role,
            label=template.label,
            prompt=prompt_text,
            bars=bar_count,
            target_seconds=target_seconds,
            energy=template.energy,
            model_id=None,
            seed_offset=0,
            transition=template.transition,
            motif_directive=motif_directive,
            variation_axes=variation_axes,
            cadence_hint=cadence_hint,
            orchestration=orchestrations[0],
        )

        return CompositionPlan(
            version=PLAN_VERSION,
            tempo_bpm=tempo_bpm,
            time_signature=DEFAULT_TIME_SIGNATURE,
            key=musical_key,
            total_bars=bar_count,
            total_duration_seconds=target_seconds,
            theme=descriptor,
            sections=[section],
        )

    def _build_long_form_plan(self, request: GenerationRequest) -> CompositionPlan:
        seconds_total = float(max(request.duration_seconds, LONG_FORM_THRESHOLD))
        templates = _select_long_templates(seconds_total)
        beats_per_bar = _beats_per_bar(DEFAULT_TIME_SIGNATURE)
        tempo_hint = _tempo_hint(seconds_total, templates, beats_per_bar)
        tempo_bpm = _select_tempo(tempo_hint)
        effective_tempo = max(tempo_bpm, MIN_TEMPO)
        seconds_per_bar = (60.0 / float(effective_tempo)) * float(beats_per_bar)

        bars = _allocate_bars(templates, seconds_total, seconds_per_bar)

        base_seed = (
            request.seed if request.seed is not None else _deterministic_seed(request.prompt)
        )
        prompt_fold = request.prompt.casefold()
        profile = _match_genre_profile(prompt_fold)

        descriptor = _build_theme_descriptor(
            request.prompt,
            prompt_fold,
            templates,
            profile,
            base_seed,
        )
        palette = _categorise_instrumentation(
            descriptor,
            prompt_fold,
            profile,
            base_seed,
        )
        palette_offsets = _palette_offsets(palette, base_seed)
        orchestrations = _plan_orchestrations(templates, palette, palette_offsets)
        musical_key = _select_key(base_seed)

        sections: List[CompositionSection] = []
        total_bars = 0

        for index, template in enumerate(templates):
            arrangement_text = _describe_orchestration(orchestrations[index])
            prompt_text = _render_prompt(
                template.prompt_template,
                prompt=request.prompt,
                descriptor=descriptor,
                section_index=index,
                arrangement=arrangement_text,
            ).strip()

            motif_directive, variation_axes, cadence_hint = _directives_for_role(
                template.role
            )
            bar_count = max(1, bars[index])
            total_bars += bar_count
            section_seconds = max(LONG_MIN_SECTION_SECONDS, bar_count * seconds_per_bar)
            sections.append(
                CompositionSection(
                    section_id=f"s{index:02d}",
                    role=template.role,
                    label=template.label,
                    prompt=prompt_text,
                    bars=bar_count,
                    target_seconds=section_seconds,
                    energy=template.energy,
                    model_id=None,
                    seed_offset=index,
                    transition=template.transition,
                    motif_directive=motif_directive,
                    variation_axes=variation_axes,
                    cadence_hint=cadence_hint,
                    orchestration=orchestrations[index],
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
            theme=descriptor,
            sections=sections,
        )

    def _build_short_form_plan(self, request: GenerationRequest) -> CompositionPlan:
        seconds_total = float(max(request.duration_seconds, SHORT_MIN_TOTAL_SECONDS))
        templates = _select_short_templates(seconds_total)
        beats_per_bar = _beats_per_bar(DEFAULT_TIME_SIGNATURE)
        total_weight = float(sum(template.base_bars for template in templates) or 1)
        raw_tempo = int(round(240.0 * total_weight / seconds_total)) if seconds_total > 0 else 90
        tempo_bpm = _select_tempo(raw_tempo)
        effective_tempo = max(tempo_bpm, MIN_TEMPO)
        seconds_per_bar = (60.0 / float(effective_tempo)) * float(beats_per_bar)

        base_seed = (
            request.seed if request.seed is not None else _deterministic_seed(request.prompt)
        )
        prompt_fold = request.prompt.casefold()
        profile = _match_genre_profile(prompt_fold)

        descriptor = _build_theme_descriptor(
            request.prompt,
            prompt_fold,
            templates,
            profile,
            base_seed,
        )
        palette = _categorise_instrumentation(
            descriptor,
            prompt_fold,
            profile,
            base_seed,
        )
        palette_offsets = _palette_offsets(palette, base_seed)
        orchestrations = _plan_orchestrations(templates, palette, palette_offsets)
        musical_key = _select_key(base_seed)

        sections: List[CompositionSection] = []
        total_bars = 0
        total_duration = 0.0

        for index, template in enumerate(templates):
            arrangement_text = _describe_orchestration(orchestrations[index])
            prompt_text = _render_prompt(
                template.prompt_template,
                prompt=request.prompt,
                descriptor=descriptor,
                section_index=index,
                arrangement=arrangement_text,
            ).strip()

            motif_directive, variation_axes, cadence_hint = _directives_for_role(
                template.role
            )

            ratio = float(template.base_bars) / total_weight
            section_seconds = max(
                SHORT_MIN_SECTION_SECONDS,
                seconds_total * ratio,
            )
            bar_count = (
                int(round(section_seconds / seconds_per_bar))
                if seconds_per_bar > 0
                else template.base_bars
            )
            bar_count = max(template.min_bars, min(template.max_bars, max(1, bar_count)))
            section_seconds = max(SHORT_MIN_SECTION_SECONDS, bar_count * seconds_per_bar)

            total_bars += bar_count
            total_duration += section_seconds

            sections.append(
                CompositionSection(
                    section_id=f"s{index:02d}",
                    role=template.role,
                    label=template.label,
                    prompt=prompt_text,
                    bars=bar_count,
                    target_seconds=section_seconds,
                    energy=template.energy,
                    model_id=None,
                    seed_offset=index,
                    transition=template.transition,
                    motif_directive=motif_directive,
                    variation_axes=variation_axes,
                    cadence_hint=cadence_hint,
                    orchestration=orchestrations[index],
                )
            )

        return CompositionPlan(
            version=PLAN_VERSION,
            tempo_bpm=tempo_bpm,
            time_signature=DEFAULT_TIME_SIGNATURE,
            key=musical_key,
            total_bars=total_bars,
            total_duration_seconds=total_duration,
            theme=descriptor,
            sections=sections,
        )


def _beats_per_bar(time_signature: str) -> int:
    try:
        numerator = int(time_signature.split("/")[0])
        return max(1, numerator)
    except Exception:  # noqa: BLE001
        return 4

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


def _tempo_hint(
    total_seconds: float,
    templates: List[SectionTemplate],
    beats_per_bar: int,
) -> int:
    if total_seconds <= 0:
        return 96
    base_bars = sum(template.base_bars for template in templates)
    if base_bars <= 0:
        base_bars = max(1, len(templates)) * 8
    beats = base_bars * max(beats_per_bar, 1)
    raw = int(round((60.0 * beats) / total_seconds))
    return max(MIN_TEMPO, raw)


def _allocate_bars(
    templates: List[SectionTemplate],
    seconds_total: float,
    seconds_per_bar: float,
) -> List[int]:
    if not templates:
        return []
    seconds_per_bar = max(seconds_per_bar, 1.0)
    target_bars_float = seconds_total / seconds_per_bar
    min_total_bars = sum(template.min_bars for template in templates)
    target_bars = max(min_total_bars, int(round(target_bars_float)))
    base_total = sum(template.base_bars for template in templates)
    scale = target_bars / max(base_total, 1)
    bars = []
    for template in templates:
        scaled = int(round(template.base_bars * scale))
        bounded = max(template.min_bars, min(template.max_bars, max(1, scaled)))
        bars.append(bounded)
    return _rebalance_bars(templates, bars, target_bars)


def _rebalance_bars(
    templates: List[SectionTemplate],
    bars: List[int],
    target_bars: int,
) -> List[int]:
    if not bars:
        return []
    min_bars = [template.min_bars for template in templates]
    max_bars = [template.max_bars for template in templates]
    total = sum(bars)
    add_order = _expand_priority(templates, ADD_PRIORITY)
    remove_order = _expand_priority(templates, REMOVE_PRIORITY)

    safety = 0
    while total < target_bars and safety < 4096:
        adjusted = False
        for idx in add_order:
            if bars[idx] < max_bars[idx]:
                bars[idx] += 1
                total += 1
                adjusted = True
                if total >= target_bars:
                    break
        if not adjusted:
            break
        safety += 1

    safety = 0
    while total > target_bars and safety < 4096:
        adjusted = False
        for idx in remove_order:
            if bars[idx] > min_bars[idx]:
                bars[idx] -= 1
                total -= 1
                adjusted = True
                if total <= target_bars:
                    break
        if not adjusted:
            break
        safety += 1

    return bars


def _expand_priority(
    templates: List[SectionTemplate],
    priority: List[SectionRole],
) -> List[int]:
    order: List[int] = []
    for role in priority:
        for index, template in enumerate(templates):
            if template.role == role and index not in order:
                order.append(index)
    for index in range(len(templates)):
        if index not in order:
            order.append(index)
    return order


def _build_theme_descriptor(
    prompt: str,
    prompt_fold: str,
    templates: List[SectionTemplate],
    profile: Optional[GenreProfileData],
    base_seed: int,
) -> ThemeDescriptor:
    instrumentation = _extract_keywords(prompt_fold, _LEXICON.instrument_keywords)
    if profile is not None:
        instrumentation.extend(
            item for item in profile.instrumentation if item not in instrumentation
        )
    if not instrumentation:
        instrumentation = _fallback_instrumentation(base_seed, count=3)
    instrumentation = _dedupe(instrumentation)

    rhythm = _derive_rhythm(prompt_fold, templates, profile)
    motif = _derive_motif(prompt)
    texture = _derive_texture(prompt_fold, instrumentation, profile)
    dynamic_curve = [
        _dynamic_label(template.role, template.energy, index, len(templates))
        for index, template in enumerate(templates)
    ]

    return ThemeDescriptor(
        motif=motif,
        instrumentation=instrumentation,
        rhythm=rhythm,
        dynamic_curve=dynamic_curve,
        texture=texture,
    )


def _extract_keywords(prompt_fold: str, mapping: list[KeywordMapping]) -> list[str]:
    results: list[str] = []
    for entry in mapping:
        if entry.folded in prompt_fold and entry.descriptor not in results:
            results.append(entry.descriptor)
    return results


def _derive_rhythm(
    prompt_fold: str,
    templates: List[SectionTemplate],
    profile: Optional[GenreProfileData],
) -> str:
    for entry in _LEXICON.rhythm_keywords:
        if entry.folded in prompt_fold:
            return entry.descriptor
    if profile is not None and profile.rhythm:
        return profile.rhythm
    energies = {template.energy for template in templates}
    if SectionEnergy.HIGH in energies:
        return "driving pulse"
    if SectionEnergy.MEDIUM in energies:
        return "steady groove"
    return "gentle pulse"


def _derive_motif(prompt: str) -> str:
    words = [
        token.strip(".,;:!?")
        for token in prompt.split()
        if token.strip(".,;:!?").isalpha()
    ]
    if len(words) >= 3:
        return " ".join(words[:3])
    if words:
        return " ".join(words)
    return "primary motif"


def _derive_texture(
    prompt_fold: str,
    instrumentation: list[str],
    profile: Optional[GenreProfileData],
) -> str:
    for entry in _LEXICON.texture_keywords:
        if entry.folded in prompt_fold:
            return entry.descriptor
    if profile is not None and profile.texture:
        return profile.texture
    if instrumentation:
        return f"focused blend of {', '.join(instrumentation[:2])}"
    return _LEXICON.defaults.texture


def _dynamic_label(
    role: SectionRole,
    energy: SectionEnergy,
    index: int,
    total: int,
) -> str:
    primary, release = ENERGY_DYNAMIC_MAP.get(energy, ("flowing motion", "gentle close"))
    if index == 0:
        if role == SectionRole.INTRO:
            return primary
        return "emerging momentum"
    if index == total - 1:
        if role == SectionRole.OUTRO:
            return release
        return "resolved cadence"
    if role == SectionRole.CHORUS:
        return "anthemic peak"
    if role == SectionRole.BRIDGE:
        return "suspended tension"
    if energy == SectionEnergy.HIGH:
        return "heightened intensity"
    if energy == SectionEnergy.MEDIUM:
        return "building drive"
    return "textural breath"


def _render_prompt(
    template: str,
    *,
    prompt: str,
    descriptor: ThemeDescriptor,
    section_index: int,
    arrangement: str,
) -> str:
    instrumentation_text = (
        ", ".join(descriptor.instrumentation)
        or ", ".join(_LEXICON.defaults.instrumentation)
    )
    if descriptor.dynamic_curve:
        if section_index < len(descriptor.dynamic_curve):
            dynamic = descriptor.dynamic_curve[section_index]
        else:
            dynamic = descriptor.dynamic_curve[-1]
    else:
        dynamic = "flowing dynamic"
    texture = descriptor.texture or _LEXICON.defaults.texture
    arrangement_text = arrangement or instrumentation_text
    return template.format(
        prompt=prompt,
        motif=descriptor.motif,
        instrumentation=instrumentation_text,
        rhythm=descriptor.rhythm,
        texture=texture,
        dynamic=dynamic,
        arrangement=arrangement_text,
    )


def _categorise_instrumentation(
    descriptor: ThemeDescriptor,
    prompt_fold: str,
    profile: Optional[GenreProfileData],
    _base_seed: int,
) -> dict[str, list[str]]:
    palette = {category: [] for category in CATEGORY_KEYWORDS}
    for label in descriptor.instrumentation:
        lowered = label.lower()
        matched = False
        for category, keywords in CATEGORY_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                palette[category].append(label)
                matched = True
        if not matched:
            palette.setdefault("textures", []).append(label)

    if any(keyword in prompt_fold for keyword in CATEGORY_KEYWORDS["vocals"]):
        palette.setdefault("vocals", []).append("expressive vocals")

    if profile is not None:
        palette.setdefault("rhythm", []).extend(profile.layers.rhythm)
        palette.setdefault("bass", []).extend(profile.layers.bass)
        palette.setdefault("harmony", []).extend(profile.layers.harmony)
        palette.setdefault("lead", []).extend(profile.layers.lead)
        palette.setdefault("textures", []).extend(profile.layers.textures)
        palette.setdefault("vocals", []).extend(profile.layers.vocals)

    for category, defaults in DEFAULT_LAYER_FALLBACKS.items():
        existing = palette.get(category, [])
        if category == "vocals" and not existing:
            palette[category] = []
        else:
            palette[category] = _dedupe(existing) if existing else list(defaults)
    return palette


def _plan_orchestrations(
    templates: List[SectionTemplate],
    palette: dict[str, list[str]],
    offsets: dict[str, int],
) -> List[SectionOrchestration]:
    orchestrations: List[SectionOrchestration] = []
    for index, template in enumerate(templates):
        counts = SECTION_LAYER_PROFILE.get(template.role, SECTION_LAYER_PROFILE.get("default", {}))
        orchestrations.append(
            _build_orchestration(counts, palette, offsets=offsets, section_index=index)
        )
    return orchestrations


def _palette_offsets(palette: dict[str, list[str]], seed: int) -> dict[str, int]:
    categories = set(DEFAULT_LAYER_FALLBACKS.keys()) | set(palette.keys())
    offsets: dict[str, int] = {}
    for category in categories:
        offsets[category] = _stable_offset(f"palette:{category}", seed)
    return offsets


def _build_orchestration(
    counts: dict[str, int],
    palette: dict[str, list[str]],
    *,
    offsets: dict[str, int],
    section_index: int,
) -> SectionOrchestration:
    orchestration = SectionOrchestration()
    for category in ("rhythm", "bass", "harmony", "lead", "textures", "vocals"):
        count = counts.get(category, 0)
        if count <= 0:
            items: list[str] = []
        else:
            source = list(palette.get(category, []))
            if not source:
                if category == "vocals":
                    items = []
                else:
                    source = list(DEFAULT_LAYER_FALLBACKS.get(category, []))
            if source:
                base_offset = offsets.get(category, 0)
                items = _cycled_slice(source, count, base_offset + section_index)
            else:
                items = []
        setattr(orchestration, category, items)
    return orchestration


def _cycled_slice(source: list[str], count: int, offset: int) -> list[str]:
    if not source or count <= 0:
        return []
    length = len(source)
    result: list[str] = []
    for index in range(count):
        result.append(source[(offset + index) % length])
    return result


def _fallback_instrumentation(seed: int, *, count: int) -> list[str]:
    pool = list(_LEXICON.defaults.instrumentation) or ["blended instrumentation"]
    count = max(1, min(count, len(pool)))
    offset = _stable_offset("instrumentation", seed)
    return _cycled_slice(pool, count, offset)


def _stable_offset(label: str, seed: int) -> int:
    token = f"{label}:{seed}".encode("utf-8")
    hash_value = 0x811C9DC5
    for byte in token:
        hash_value ^= byte
        hash_value = (hash_value * 0x01000193) % (1 << 32)
    return hash_value


def _describe_orchestration(orchestration: SectionOrchestration) -> str:
    parts: list[str] = []
    for category in ("rhythm", "bass", "harmony", "lead", "textures", "vocals"):
        instruments = getattr(orchestration, category)
        if instruments:
            parts.append(", ".join(instruments))
    return ", ".join(parts)


def _dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _select_long_templates(duration_seconds: float) -> Iterable[SectionTemplate]:
    return _LEXICON.select_long(duration_seconds)


def _select_short_templates(duration_seconds: float) -> List[SectionTemplate]:
    return _LEXICON.select_short(duration_seconds)


def _select_motif_template() -> SectionTemplate:
    for template in _LEXICON.select_short(0.0):
        if template.role == SectionRole.MOTIF:
            return template
    for template in _LEXICON.select_short(SHORT_MIN_TOTAL_SECONDS):
        if template.role == SectionRole.MOTIF:
            return template
    return SectionTemplate(
        role=SectionRole.MOTIF,
        label="Motif",
        energy=SectionEnergy.MEDIUM,
        base_bars=6,
        min_bars=4,
        max_bars=8,
        prompt_template=(
            "Present the {motif} motif clearly, keeping {instrumentation} tight around the {rhythm} while {dynamic} grows."
        ),
        transition=None,
    )
