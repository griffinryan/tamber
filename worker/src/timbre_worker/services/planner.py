"""Rule-based composition planner producing structured section plans."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from ..app.models import (
    CompositionPlan,
    CompositionSection,
    GenerationRequest,
    SectionEnergy,
    SectionOrchestration,
    SectionRole,
    ThemeDescriptor,
)

PLAN_VERSION = "v3"
DEFAULT_TIME_SIGNATURE = "4/4"
MIN_TEMPO = 68
MAX_TEMPO = 128
SHORT_MIN_TOTAL_SECONDS = 2.0
SHORT_MIN_SECTION_SECONDS = 2.0
LONG_FORM_THRESHOLD = 90.0
LONG_MIN_SECTION_SECONDS = 16.0

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

INSTRUMENT_KEYWORDS = [
    ("piano", "warm piano"),
    ("upright piano", "intimate upright piano"),
    ("keys", "soft keys"),
    ("synthwave", "retro synth layers"),
    ("synth", "lush synth pads"),
    ("analog", "analog synths"),
    ("modular", "modular synth textures"),
    ("arp", "arpeggiated synths"),
    ("guitar", "ambient guitar"),
    ("guitars", "layered guitars"),
    ("acoustic guitar", "acoustic guitar"),
    ("electric guitar", "electric guitars"),
    ("banjo", "twangy banjo"),
    ("mandolin", "sparkling mandolin"),
    ("violin", "expressive strings"),
    ("viola", "velvety viola"),
    ("cello", "warm cello"),
    ("strings", "layered strings"),
    ("string", "layered strings"),
    ("orchestra", "orchestral strings"),
    ("bass", "deep bass"),
    ("upright bass", "upright bass"),
    ("808", "808 bass"),
    ("drum", "tight drums"),
    ("drums", "tight drums"),
    ("drum machine", "drum machine groove"),
    ("percussion", "organic percussion"),
    ("tabla", "tabla rhythms"),
    ("conga", "conga grooves"),
    ("bongo", "hand drum patterns"),
    ("choir", "airy choir voices"),
    ("choral", "lush choir"),
    ("vocal", "ethereal vocals"),
    ("vocals", "expressive vocals"),
    ("singer", "soulful vocalist"),
    ("vocalist", "soaring vocals"),
    ("rap", "rhythmic rap vocals"),
    ("brass", "smooth brass"),
    ("horn", "bold brass horns"),
    ("sax", "saxophone lead"),
    ("saxophone", "saxophone lead"),
    ("trumpet", "radiant trumpet lead"),
    ("trombone", "velvet trombone swells"),
    ("flute", "breathy flute"),
    ("clarinet", "warm clarinet"),
    ("oboe", "lyrical oboe"),
    ("bassoon", "warm bassoon"),
    ("harp", "glittering harp arpeggios"),
    ("organ", "vintage organ"),
    ("rhodes", "rhodes electric piano"),
    ("vibraphone", "glassy vibraphone"),
    ("marimba", "woody marimba"),
    ("kalimba", "sparkling kalimba"),
    ("sitar", "resonant sitar"),
    ("accordion", "lively accordion"),
    ("harmonica", "bluesy harmonica"),
    ("synthpop", "glassy synth pop layers"),
    ("ambient", "atmospheric textures"),
    ("lofi", "dusty keys"),
]

RHYTHM_KEYWORDS = [
    ("waltz", "gentle 3/4 sway"),
    ("swing", "swinging groove"),
    ("jazz", "swinging groove"),
    ("hip hop", "laid-back hip hop beat"),
    ("boom bap", "boom-bap pulse"),
    ("house", "four-on-the-floor pulse"),
    ("techno", "driving techno rhythm"),
    ("trance", "rolling trance rhythm"),
    ("trap", "stuttered trap beat"),
    ("downtempo", "downtempo pulse"),
    ("breakbeat", "syncopated breakbeat"),
    ("drum and bass", "rapid drum and bass break"),
    ("dnb", "rapid drum and bass break"),
    ("bossa", "bossa nova sway"),
    ("reggae", "off-beat reggae groove"),
    ("rock", "driving rock beat"),
    ("metal", "pummelling double-kick groove"),
    ("edm", "steady four-on-the-floor pulse"),
    ("dance", "club-ready four-on-the-floor beat"),
    ("electronic", "mechanical electronic pulse"),
    ("folk", "organic strummed pulse"),
    ("latin", "syncopated latin groove"),
    ("ambient", "floating pulse"),
]

TEXTURE_KEYWORDS = [
    ("dream", "dreamy haze"),
    ("noir", "late-night noir mood"),
    ("dark", "shadowy atmosphere"),
    ("bright", "glowing shimmer"),
    ("glitch", "glitchy texture"),
    ("cinematic", "cinematic expanse"),
    ("epic", "soaring atmosphere"),
    ("mystic", "mystical aura"),
    ("lofi", "dusty vignette"),
    ("rain", "rain-soaked ambience"),
    ("storm", "electrified storm ambience"),
    ("forest", "woodland ambience"),
    ("ocean", "rolling ocean spray"),
    ("space", "cosmic expanse"),
    ("desert", "sun-baked desert shimmer"),
    ("vintage", "vintage tape patina"),
    ("psychedelic", "psychedelic kaleidoscope"),
]

DEFAULT_INSTRUMENTATION = [
    "blended instrumentation",
    "layered synth textures",
    "hybrid acoustic-electric palette",
    "ensemble interplay",
]
DEFAULT_TEXTURE = "immersive atmosphere"

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


@dataclass(frozen=True)
class GenreProfile:
    name: str
    keywords: tuple[str, ...]
    instrumentation: tuple[str, ...]
    rhythm: Optional[str] = None
    texture: Optional[str] = None
    layer_overrides: dict[str, tuple[str, ...]] = field(default_factory=dict)


GENRE_PROFILES: tuple[GenreProfile, ...] = (
    GenreProfile(
        name="lofi",
        keywords=("lofi", "lo-fi", "chillhop", "chill hop", "study beats"),
        instrumentation=(
            "dusty electric piano",
            "tape-saturated drum loop",
            "warm bass guitar",
            "soft nylon guitar",
        ),
        rhythm="laid-back hip hop beat",
        texture="hazy lo-fi ambience",
        layer_overrides={
            "rhythm": ("dusty drum loop", "lazy snare with brushes"),
            "bass": ("mellow sub bass", "upright bass with tape wobble"),
            "harmony": ("noisy electric piano chords", "soft guitar voicings"),
            "lead": ("wistful synth lead", "gentle vibraphone motifs"),
            "textures": ("vinyl crackle", "nocturnal rain ambience"),
            "vocals": ("hummed vocal chops",),
        },
    ),
    GenreProfile(
        name="jazz",
        keywords=("jazz", "bebop", "swing", "hard bop", "cool jazz"),
        instrumentation=(
            "upright bass",
            "brush drum kit",
            "extended jazz piano",
            "saxophone lead",
        ),
        rhythm="swinging groove",
        texture="smoky lounge atmosphere",
        layer_overrides={
            "rhythm": ("brush drum kit", "ride cymbal patterns"),
            "bass": ("walking upright bass",),
            "harmony": ("extended jazz piano voicings", "comping guitar chords"),
            "lead": ("saxophone lead", "muted trumpet phrases"),
            "textures": ("club room ambience", "late-night crowd murmur"),
            "vocals": ("scat vocal ad-libs",),
        },
    ),
    GenreProfile(
        name="rock",
        keywords=("rock", "alt rock", "indie rock", "garage rock", "post-punk"),
        instrumentation=(
            "distorted electric guitars",
            "live drum kit",
            "electric bass",
            "anthemic vocals",
        ),
        rhythm="driving rock beat",
        texture="amplified stage energy",
        layer_overrides={
            "rhythm": ("punchy live drums", "crashing cymbals"),
            "bass": ("gritty electric bass",),
            "harmony": ("crunchy rhythm guitars", "stadium power chords"),
            "lead": ("soaring guitar lead", "anthemic vocal hooks"),
            "textures": ("amp feedback swells", "crowd reverb wash"),
            "vocals": ("anthemic rock vocals",),
        },
    ),
    GenreProfile(
        name="metal",
        keywords=("metal", "thrash", "doom", "heavy metal", "prog metal"),
        instrumentation=(
            "double kick drums",
            "down-tuned rhythm guitars",
            "growling bass",
            "aggressive vocals",
        ),
        rhythm="relentless double-kick drive",
        texture="dense wall of distortion",
        layer_overrides={
            "rhythm": ("blast beat drums", "double-kick assaults"),
            "bass": ("growling electric bass",),
            "harmony": ("down-tuned rhythm guitars", "chugging power chords"),
            "lead": ("shredded guitar leads", "screaming harmonised solos"),
            "textures": ("amp roar", "crowd roar tails"),
            "vocals": ("harsh metal vocals",),
        },
    ),
    GenreProfile(
        name="orchestral",
        keywords=("orchestral", "symphonic", "film score", "cinematic", "soundtrack"),
        instrumentation=(
            "string ensemble",
            "brass section",
            "woodwind choir",
            "concert percussion",
        ),
        rhythm="expansive cinematic pulse",
        texture="lush orchestral hall",
        layer_overrides={
            "rhythm": ("gran cassa hits", "rolling orchestral percussion"),
            "bass": ("double basses", "low brass swells"),
            "harmony": ("legato string ensemble", "warm horn chords"),
            "lead": ("solo violin", "trumpet fanfare"),
            "textures": ("concert hall reverberation", "choir pads"),
            "vocals": ("wordless choir",),
        },
    ),
    GenreProfile(
        name="folk",
        keywords=("folk", "acoustic", "bluegrass", "roots", "americana", "country"),
        instrumentation=(
            "acoustic guitar",
            "mandolin",
            "upright bass",
            "fiddle melodies",
        ),
        rhythm="organic strummed pulse",
        texture="warm rustic atmosphere",
        layer_overrides={
            "rhythm": ("brushy folk kit", "hand clap patterns"),
            "bass": ("plucked upright bass",),
            "harmony": ("fingerpicked acoustic guitar", "sparkling mandolin arpeggios"),
            "lead": ("fiddle leads", "slide guitar phrases"),
            "textures": ("barn room reverb", "nature ambience"),
            "vocals": ("stacked folk harmonies",),
        },
    ),
    GenreProfile(
        name="ambient",
        keywords=("ambient", "soundscape", "drone", "meditation", "ethereal", "atmospheric"),
        instrumentation=(
            "evolving synth pads",
            "granular textures",
            "soft mallet swells",
            "choral atmospheres",
        ),
        rhythm="floating pulse",
        texture="expansive ambient haze",
        layer_overrides={
            "rhythm": ("subtle percussive pulses", "soft percussion swells"),
            "bass": ("deep sine bass swells",),
            "harmony": ("evolving synth pads", "glacial piano chords"),
            "lead": ("breathy vocal motifs", "glass harmonica lines"),
            "textures": ("wind soundscape", "distant shimmer"),
            "vocals": ("celestial vocal layers",),
        },
    ),
    GenreProfile(
        name="electronic",
        keywords=("electronic", "edm", "house", "techno", "trance", "club", "electro", "synthwave", "synth-pop"),
        instrumentation=(
            "punchy electronic kick",
            "rolling bass synth",
            "arpeggiated synths",
            "bright lead plucks",
        ),
        rhythm="four-on-the-floor pulse",
        texture="neon club atmosphere",
        layer_overrides={
            "rhythm": ("four-on-the-floor kick", "syncopated hi-hats"),
            "bass": ("rolling bass synth", "acid bassline"),
            "harmony": ("wide pad chords", "lush supersaw stacks"),
            "lead": ("glittering synth lead", "vocoder hooks"),
            "textures": ("club crowd ambience", "side-chained pads"),
            "vocals": ("processed vocal chops",),
        },
    ),
    GenreProfile(
        name="hiphop",
        keywords=("hip hop", "trap", "boom bap", "rap", "grime"),
        instrumentation=(
            "knocking drum machine",
            "808 bass",
            "sampled keys",
            "vocal chops",
        ),
        rhythm="head-nod hip hop pulse",
        texture="urban night energy",
        layer_overrides={
            "rhythm": ("punchy drum machine groove", "crisp claps"),
            "bass": ("808 bass", "subby bass drops"),
            "harmony": ("minor key piano loops", "sampled soul chops"),
            "lead": ("pitched vocal chops", "synth brass stabs"),
            "textures": ("street ambience", "vinyl noise"),
            "vocals": ("rap ad-libs",),
        },
    ),
)


def _match_genre_profile(prompt_lower: str) -> Optional[GenreProfile]:
    best: Optional[GenreProfile] = None
    best_len = -1
    for profile in GENRE_PROFILES:
        for keyword in profile.keywords:
            if keyword in prompt_lower and len(keyword) > best_len:
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


class CompositionPlanner:
    """Generates deterministic composition plans from prompts."""

    def build_plan(self, request: GenerationRequest) -> CompositionPlan:
        if float(request.duration_seconds) >= LONG_FORM_THRESHOLD:
            return self._build_long_form_plan(request)
        return self._build_short_form_plan(request)

    def _build_long_form_plan(self, request: GenerationRequest) -> CompositionPlan:
        seconds_total = float(max(request.duration_seconds, LONG_FORM_THRESHOLD))
        templates = list(_select_long_templates(seconds_total))
        beats_per_bar = _beats_per_bar(DEFAULT_TIME_SIGNATURE)
        tempo_hint = _tempo_hint(seconds_total, templates, beats_per_bar)
        tempo_bpm = _select_tempo(tempo_hint)
        effective_tempo = max(tempo_bpm, MIN_TEMPO)
        seconds_per_bar = (60.0 / float(effective_tempo)) * float(beats_per_bar)

        bars = _allocate_bars(templates, seconds_total, seconds_per_bar)

        base_seed = (
            request.seed if request.seed is not None else _deterministic_seed(request.prompt)
        )
        prompt_lower = request.prompt.lower()
        profile = _match_genre_profile(prompt_lower)

        descriptor = _build_theme_descriptor(
            request.prompt,
            templates,
            profile,
            base_seed,
        )
        palette = _categorise_instrumentation(
            descriptor,
            request.prompt,
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
        templates = list(_select_short_templates(seconds_total))
        beats_per_bar = _beats_per_bar(DEFAULT_TIME_SIGNATURE)
        total_weight = float(sum(template.base_bars for template in templates) or 1)
        raw_tempo = int(round(240.0 * total_weight / seconds_total)) if seconds_total > 0 else 90
        tempo_bpm = _select_tempo(raw_tempo)
        effective_tempo = max(tempo_bpm, MIN_TEMPO)
        seconds_per_bar = (60.0 / float(effective_tempo)) * float(beats_per_bar)

        base_seed = (
            request.seed if request.seed is not None else _deterministic_seed(request.prompt)
        )
        prompt_lower = request.prompt.lower()
        profile = _match_genre_profile(prompt_lower)

        descriptor = _build_theme_descriptor(
            request.prompt,
            templates,
            profile,
            base_seed,
        )
        palette = _categorise_instrumentation(
            descriptor,
            request.prompt,
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
    templates: List[SectionTemplate],
    profile: Optional[GenreProfile],
    base_seed: int,
) -> ThemeDescriptor:
    prompt_lower = prompt.lower()
    instrumentation = _extract_keywords(prompt_lower, INSTRUMENT_KEYWORDS)
    if profile is not None:
        instrumentation.extend(item for item in profile.instrumentation if item not in instrumentation)
    if not instrumentation:
        instrumentation = _fallback_instrumentation(base_seed, count=3)
    instrumentation = _dedupe(instrumentation)

    rhythm = _derive_rhythm(prompt_lower, templates, profile)
    motif = _derive_motif(prompt)
    texture = _derive_texture(prompt_lower, instrumentation, profile)
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


def _extract_keywords(prompt_lower: str, mapping: list[tuple[str, str]]) -> list[str]:
    results: list[str] = []
    for keyword, label in mapping:
        if keyword in prompt_lower and label not in results:
            results.append(label)
    return results


def _derive_rhythm(
    prompt_lower: str,
    templates: List[SectionTemplate],
    profile: Optional[GenreProfile],
) -> str:
    for keyword, label in RHYTHM_KEYWORDS:
        if keyword in prompt_lower:
            return label
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
    prompt_lower: str,
    instrumentation: list[str],
    profile: Optional[GenreProfile],
) -> str:
    for keyword, label in TEXTURE_KEYWORDS:
        if keyword in prompt_lower:
            return label
    if profile is not None and profile.texture:
        return profile.texture
    if instrumentation:
        return f"focused blend of {', '.join(instrumentation[:2])}"
    return DEFAULT_TEXTURE


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
        or ", ".join(DEFAULT_INSTRUMENTATION)
    )
    if descriptor.dynamic_curve:
        if section_index < len(descriptor.dynamic_curve):
            dynamic = descriptor.dynamic_curve[section_index]
        else:
            dynamic = descriptor.dynamic_curve[-1]
    else:
        dynamic = "flowing dynamic"
    texture = descriptor.texture or DEFAULT_TEXTURE
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
    prompt: str,
    profile: Optional[GenreProfile],
    base_seed: int,
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

    prompt_lower = prompt.lower()
    if any(keyword in prompt_lower for keyword in CATEGORY_KEYWORDS["vocals"]):
        palette.setdefault("vocals", []).append("expressive vocals")

    if profile is not None:
        for category, values in profile.layer_overrides.items():
            if not values:
                continue
            palette.setdefault(category, []).extend(values)

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
    pool = list(DEFAULT_INSTRUMENTATION) or ["blended instrumentation"]
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
    if duration_seconds >= 150.0:
        yield SectionTemplate(
            role=SectionRole.INTRO,
            label="Intro",
            energy=SectionEnergy.LOW,
            base_bars=16,
            min_bars=10,
            max_bars=18,
            prompt_template=(
                "Set the stage with {arrangement}, foreshadowing the {motif} motif over a {rhythm} pulse and {texture} atmosphere."
            ),
            transition="Invite motif",
        )
        yield SectionTemplate(
            role=SectionRole.MOTIF,
            label="Motif",
            energy=SectionEnergy.MEDIUM,
            base_bars=20,
            min_bars=16,
            max_bars=24,
            prompt_template=(
                "State the {motif} motif in full, letting {arrangement} lock into the {rhythm} while {dynamic} blooms."
            ),
            transition="Build anticipation",
        )
        yield SectionTemplate(
            role=SectionRole.BRIDGE,
            label="Bridge",
            energy=SectionEnergy.MEDIUM,
            base_bars=12,
            min_bars=8,
            max_bars=16,
            prompt_template=(
                "Recast the {motif} motif by thinning the layers so {arrangement} can explore contrasting colours before the chorus returns."
            ),
            transition="Spark chorus",
        )
        yield SectionTemplate(
            role=SectionRole.CHORUS,
            label="Chorus",
            energy=SectionEnergy.HIGH,
            base_bars=24,
            min_bars=20,
            max_bars=28,
            prompt_template=(
                "Lift the {motif} motif into an anthemic chorus where {arrangement} drives the groove and {dynamic} peaks."
            ),
            transition="Glide to outro",
        )
        yield SectionTemplate(
            role=SectionRole.OUTRO,
            label="Outro",
            energy=SectionEnergy.MEDIUM,
            base_bars=14,
            min_bars=8,
            max_bars=18,
            prompt_template=(
                "Close by reshaping the {motif} motif, letting {arrangement} ease the {rhythm} into a reflective {texture} fade."
            ),
            transition="Fade to silence",
        )
        return

    yield SectionTemplate(
        role=SectionRole.INTRO,
        label="Intro",
        energy=SectionEnergy.LOW,
        base_bars=12,
        min_bars=8,
        max_bars=16,
        prompt_template=(
            "Establish the world with {arrangement}, hinting at the {motif} motif over a {rhythm} pulse and {texture} backdrop."
        ),
        transition="Reveal motif",
    )
    yield SectionTemplate(
        role=SectionRole.MOTIF,
        label="Motif",
        energy=SectionEnergy.MEDIUM,
        base_bars=18,
        min_bars=14,
        max_bars=22,
        prompt_template=(
            "Present the {motif} motif clearly, allowing {arrangement} to weave through the {rhythm} as {dynamic} intensifies."
        ),
        transition="Ignite chorus",
    )
    yield SectionTemplate(
        role=SectionRole.CHORUS,
        label="Chorus",
        energy=SectionEnergy.HIGH,
        base_bars=24,
        min_bars=18,
        max_bars=26,
        prompt_template=(
            "Amplify the {motif} motif into its fiercest form, with {arrangement} pushing the {rhythm} to a triumphant crest."
        ),
        transition="Settle to outro",
    )
    yield SectionTemplate(
        role=SectionRole.OUTRO,
        label="Outro",
        energy=SectionEnergy.MEDIUM,
        base_bars=12,
        min_bars=8,
        max_bars=16,
        prompt_template=(
            "Offer a final reflection on the {motif} motif, as {arrangement} dissolves the {rhythm} into {texture}."
        ),
        transition="Fade to silence",
    )


def _select_short_templates(duration_seconds: float) -> List[SectionTemplate]:
    if duration_seconds >= 24.0:
        return [
            SectionTemplate(
                role=SectionRole.INTRO,
                label="Arrival",
                energy=SectionEnergy.LOW,
                base_bars=4,
                min_bars=3,
                max_bars=6,
                prompt_template=(
                    "Set a {texture} scene for {prompt} by introducing the {motif} motif with {instrumentation} over a {rhythm} pulse."
                ),
                transition="Fade in layers",
            ),
            SectionTemplate(
                role=SectionRole.MOTIF,
                label="Statement",
                energy=SectionEnergy.MEDIUM,
                base_bars=8,
                min_bars=6,
                max_bars=10,
                prompt_template=(
                    "Deliver the core {motif} motif through {instrumentation}, keeping the {rhythm} driving as {dynamic} begins."
                ),
                transition="Build momentum",
            ),
            SectionTemplate(
                role=SectionRole.DEVELOPMENT,
                label="Development",
                energy=SectionEnergy.HIGH,
                base_bars=8,
                min_bars=6,
                max_bars=10,
                prompt_template=(
                    "Evolve the {motif} motif with adventurous variations, letting {instrumentation} weave syncopations over the {rhythm} while {dynamic} unfolds."
                ),
                transition="Evolve harmonies",
            ),
            SectionTemplate(
                role=SectionRole.RESOLUTION,
                label="Resolution",
                energy=SectionEnergy.MEDIUM,
                base_bars=4,
                min_bars=3,
                max_bars=6,
                prompt_template=(
                    "Guide the {motif} motif toward resolution, using {instrumentation} to ease the {rhythm} while highlighting {dynamic}."
                ),
                transition="Return home",
            ),
            SectionTemplate(
                role=SectionRole.OUTRO,
                label="Release",
                energy=SectionEnergy.LOW,
                base_bars=4,
                min_bars=2,
                max_bars=6,
                prompt_template=(
                    "Let the {motif} motif dissolve into ambience as {instrumentation} softens atop the {rhythm}, allowing {dynamic} to close the journey."
                ),
                transition="Fade to silence",
            ),
        ]
    if duration_seconds >= 16.0:
        return [
            SectionTemplate(
                role=SectionRole.INTRO,
                label="Lead-in",
                energy=SectionEnergy.LOW,
                base_bars=4,
                min_bars=2,
                max_bars=6,
                prompt_template=(
                    "Open gently with {instrumentation}, introducing the {motif} motif against a {rhythm} pulse that hints at {prompt}."
                ),
                transition="Invite motif",
            ),
            SectionTemplate(
                role=SectionRole.MOTIF,
                label="Motif A",
                energy=SectionEnergy.MEDIUM,
                base_bars=8,
                min_bars=6,
                max_bars=10,
                prompt_template=(
                    "Present the {motif} motif clearly, keeping {instrumentation} tight around the {rhythm} while {dynamic} grows."
                ),
                transition="Increase energy",
            ),
            SectionTemplate(
                role=SectionRole.DEVELOPMENT,
                label="Variation",
                energy=SectionEnergy.HIGH,
                base_bars=6,
                min_bars=4,
                max_bars=8,
                prompt_template=(
                    "Develop the {motif} motif with rhythmic twists, letting {instrumentation} ride the {rhythm} as {dynamic} intensifies."
                ),
                transition="Soften textures",
            ),
            SectionTemplate(
                role=SectionRole.RESOLUTION,
                label="Cadence",
                energy=SectionEnergy.MEDIUM,
                base_bars=4,
                min_bars=2,
                max_bars=6,
                prompt_template=(
                    "Ease the energy back, guiding {instrumentation} to resolve the {motif} motif and settle the {rhythm} with {dynamic}."
                ),
                transition="Release",
            ),
            SectionTemplate(
                role=SectionRole.OUTRO,
                label="Tail",
                energy=SectionEnergy.LOW,
                base_bars=2,
                min_bars=1,
                max_bars=4,
                prompt_template=(
                    "Conclude with a gentle echo of the {motif} motif, letting {instrumentation} and the {rhythm} fade as {dynamic} sighs out."
                ),
                transition="Fade",
            ),
        ]
    return [
        SectionTemplate(
            role=SectionRole.INTRO,
            label="Intro",
            energy=SectionEnergy.LOW,
            base_bars=2,
            min_bars=1,
            max_bars=4,
            prompt_template=(
                "Set a delicate entrance, introducing the {motif} motif with {instrumentation} over a {rhythm} that nods to {prompt}."
            ),
            transition="Introduce motif",
        ),
        SectionTemplate(
            role=SectionRole.MOTIF,
            label="Motif",
            energy=SectionEnergy.MEDIUM,
            base_bars=6,
            min_bars=4,
            max_bars=8,
            prompt_template=(
                "Deliver the {motif} motif in full, keeping {instrumentation} aligned with the {rhythm} while {dynamic} expands."
            ),
            transition="Lift energy",
        ),
    ]
