"""Rule-based composition planner producing structured section plans."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
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
from .genre_classifier import GenreClassifier, GenreProfile, VocalCharacteristics

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
    # Keyboards & Synths
    ("piano", "warm piano"),
    ("keys", "soft keys"),
    ("grand piano", "concert grand piano"),
    ("upright piano", "intimate upright piano"),
    ("electric piano", "electric piano"),
    ("rhodes", "vintage rhodes"),
    ("wurlitzer", "warm wurlitzer"),
    ("synth", "lush synth pads"),
    ("synthesizer", "layered synthesizers"),
    ("synthwave", "retro synth layers"),
    ("analog synth", "analog synth warmth"),
    ("modular", "modular synth textures"),
    ("moog", "fat moog bass"),
    ("arp", "vintage arp"),
    ("dx7", "digital dx7 tines"),
    ("organ", "swelling organ"),
    ("hammond", "hammond organ"),
    ("pipe organ", "cathedral pipe organ"),
    ("harpsichord", "baroque harpsichord"),
    ("clavinet", "funky clavinet"),
    ("mellotron", "haunting mellotron"),
    ("accordion", "expressive accordion"),

    # Guitars
    ("guitar", "ambient guitar"),
    ("guitars", "layered guitars"),
    ("electric guitar", "electric guitar"),
    ("acoustic guitar", "acoustic guitar"),
    ("distorted guitar", "distorted guitar"),
    ("clean guitar", "clean guitar tones"),
    ("fingerstyle", "fingerstyle guitar"),
    ("12-string", "shimmering 12-string"),
    ("slide guitar", "slide guitar"),
    ("banjo", "bright banjo"),
    ("ukulele", "cheerful ukulele"),
    ("mandolin", "bright mandolin"),
    ("sitar", "resonant sitar"),
    ("koto", "delicate koto"),
    ("oud", "middle eastern oud"),

    # Bass
    ("bass", "deep bass"),
    ("808", "808 sub bass"),
    ("sub bass", "rumbling sub bass"),
    ("bass guitar", "electric bass guitar"),
    ("upright bass", "walking upright bass"),
    ("double bass", "orchestral double bass"),
    ("synth bass", "pulsing synth bass"),
    ("wobble bass", "wobble bass"),
    ("slap bass", "funky slap bass"),

    # Drums & Percussion
    ("drum", "tight drums"),
    ("drums", "driving drums"),
    ("percussion", "organic percussion"),
    ("808 drums", "808 drum machine"),
    ("tr-808", "roland tr-808"),
    ("tr-909", "roland tr-909"),
    ("drum machine", "vintage drum machine"),
    ("kick", "powerful kick drum"),
    ("snare", "crisp snare"),
    ("hi-hat", "shimmering hi-hats"),
    ("cymbals", "crashing cymbals"),
    ("toms", "thundering toms"),
    ("congas", "latin congas"),
    ("bongos", "rhythmic bongos"),
    ("djembe", "african djembe"),
    ("tabla", "indian tabla"),
    ("taiko", "japanese taiko drums"),
    ("timpani", "orchestral timpani"),
    ("marimba", "wooden marimba"),
    ("vibraphone", "shimmering vibraphone"),
    ("xylophone", "bright xylophone"),
    ("glockenspiel", "delicate glockenspiel"),
    ("tambourine", "shaking tambourine"),
    ("shaker", "rhythmic shaker"),
    ("cowbell", "bright cowbell"),
    ("claps", "hand claps"),

    # Strings
    ("string", "layered strings"),
    ("strings", "orchestral strings"),
    ("violin", "expressive violin"),
    ("viola", "warm viola"),
    ("cello", "warm cello"),
    ("contrabass", "deep contrabass"),
    ("string quartet", "string quartet"),
    ("pizzicato", "pizzicato strings"),
    ("arco", "arco strings"),
    ("tremolo strings", "tremolo strings"),
    ("erhu", "chinese erhu"),
    ("fiddle", "folk fiddle"),

    # Brass
    ("brass", "smooth brass"),
    ("horn", "bold brass horns"),
    ("french horn", "french horn"),
    ("trumpet", "radiant trumpet"),
    ("trombone", "velvet trombone"),
    ("tuba", "deep tuba"),
    ("flugelhorn", "mellow flugelhorn"),
    ("cornet", "bright cornet"),
    ("euphonium", "rich euphonium"),
    ("brass section", "brass section"),

    # Woodwinds
    ("sax", "saxophone lead"),
    ("saxophone", "smooth saxophone"),
    ("alto sax", "alto saxophone"),
    ("tenor sax", "tenor saxophone"),
    ("soprano sax", "soprano saxophone"),
    ("baritone sax", "baritone saxophone"),
    ("flute", "breathy flute"),
    ("piccolo", "piercing piccolo"),
    ("clarinet", "warm clarinet"),
    ("bass clarinet", "bass clarinet"),
    ("oboe", "lyrical oboe"),
    ("bassoon", "deep bassoon"),
    ("english horn", "melancholic english horn"),
    ("recorder", "baroque recorder"),
    ("pan flute", "ethereal pan flute"),
    ("shakuhachi", "japanese shakuhachi"),
    ("didgeridoo", "australian didgeridoo"),
    ("harmonica", "bluesy harmonica"),

    # Vocals
    ("vocal", "ethereal vocals"),
    ("vocals", "expressive vocals"),
    ("singer", "soulful vocalist"),
    ("vocalist", "soaring vocals"),
    ("choir", "airy choir voices"),
    ("chorus", "vocal chorus"),
    ("falsetto", "falsetto vocals"),
    ("harmonies", "vocal harmonies"),
    ("chant", "meditative chant"),
    ("rap", "rap vocals"),
    ("spoken word", "spoken word"),
    ("whisper", "whispered vocals"),
    ("scream", "screaming vocals"),
    ("growl", "growling vocals"),

    # Electronic & Effects
    ("ambient", "atmospheric textures"),
    ("pad", "lush pad"),
    ("lofi", "dusty lo-fi keys"),
    ("vaporwave", "vaporwave aesthetics"),
    ("glitch", "glitchy textures"),
    ("noise", "textured noise"),
    ("field recording", "field recordings"),
    ("vinyl crackle", "vinyl crackle"),
    ("tape hiss", "analog tape hiss"),
    ("vocoder", "vocoded vocals"),
    ("autotune", "autotuned vocals"),
    ("talkbox", "talkbox effect"),
    ("reverb", "cavernous reverb"),
    ("delay", "spacious delay"),
    ("distortion", "heavy distortion"),
    ("phaser", "swooshing phaser"),
    ("chorus effect", "lush chorus"),

    # Orchestral & World
    ("orchestra", "full orchestra"),
    ("chamber", "chamber ensemble"),
    ("woodwind ensemble", "woodwind ensemble"),
    ("brass ensemble", "brass ensemble"),
    ("harp", "glittering harp"),
    ("celesta", "magical celesta"),
    ("dulcimer", "hammered dulcimer"),
    ("balalaika", "russian balalaika"),
    ("zither", "alpine zither"),
    ("gamelan", "indonesian gamelan"),
]

RHYTHM_KEYWORDS = [
    # Classical & Traditional
    ("waltz", "gentle 3/4 sway"),
    ("march", "steady march rhythm"),
    ("polka", "lively polka bounce"),

    # Jazz & Swing
    ("swing", "swinging groove"),
    ("bebop", "fast bebop rhythm"),
    ("shuffle", "shuffle groove"),
    ("syncopated", "syncopated rhythm"),

    # Hip-Hop & R&B
    ("hip hop", "laid-back hip hop beat"),
    ("boom bap", "boom-bap pulse"),
    ("trap", "stuttered trap beat"),
    ("half-time", "half-time hip hop"),
    ("double-time", "double-time rap flow"),

    # Electronic & Dance
    ("house", "four-on-the-floor pulse"),
    ("techno", "driving techno rhythm"),
    ("trance", "rolling trance rhythm"),
    ("dubstep", "dubstep half-time"),
    ("drum and bass", "breakneck drum and bass"),
    ("dnb", "rapid dnb rhythm"),
    ("breakbeat", "syncopated breakbeat"),
    ("downtempo", "downtempo pulse"),
    ("jungle", "jungle breakbeat"),
    ("garage", "UK garage swing"),
    ("footwork", "frenetic footwork"),
    ("breakcore", "chaotic breakcore"),
    ("grime", "grime rhythm"),

    # Latin & World
    ("bossa", "bossa nova sway"),
    ("samba", "driving samba rhythm"),
    ("rumba", "rumba groove"),
    ("salsa", "salsa rhythm"),
    ("merengue", "merengue beat"),
    ("cumbia", "cumbia rhythm"),
    ("afrobeat", "afrobeat polyrhythm"),
    ("flamenco", "flamenco compás"),
    ("reggae", "off-beat reggae groove"),
    ("dancehall", "dancehall riddim"),
    ("dub", "dub reggae beat"),

    # Rock & Funk
    ("funk", "funky groove"),
    ("disco", "disco four-on-the-floor"),
    ("rock", "driving rock beat"),
    ("backbeat", "strong backbeat"),

    # Experimental
    ("polyrhythmic", "polyrhythmic layers"),
    ("odd meter", "odd meter groove"),
    ("free time", "free-flowing rhythm"),
]

TEXTURE_KEYWORDS = [
    # Mood & Atmosphere
    ("dream", "dreamy haze"),
    ("dreamy", "ethereal dreamscape"),
    ("noir", "late-night noir mood"),
    ("dark", "shadowy atmosphere"),
    ("bright", "glowing shimmer"),
    ("warm", "warm analog feel"),
    ("cold", "cold digital air"),
    ("intimate", "intimate closeness"),
    ("spacious", "vast spacious atmosphere"),
    ("epic", "soaring atmosphere"),
    ("mystic", "mystical aura"),
    ("ethereal", "ethereal floating quality"),
    ("haunting", "haunting atmosphere"),
    ("melancholic", "melancholic mood"),
    ("euphoric", "euphoric atmosphere"),
    ("aggressive", "aggressive energy"),
    ("mellow", "mellow softness"),
    ("gentle", "gentle caress"),

    # Production Quality
    ("lofi", "dusty vignette"),
    ("hi-fi", "pristine clarity"),
    ("gritty", "gritty texture"),
    ("smooth", "silky smooth texture"),
    ("crisp", "crisp and clean"),
    ("vintage", "vintage warmth"),
    ("retro", "retro aesthetic"),
    ("analog", "analog warmth"),
    ("digital", "digital precision"),
    ("pristine", "pristine production"),
    ("degraded", "degraded lo-fi"),

    # Spatial & Textural
    ("cinematic", "cinematic expanse"),
    ("glitch", "glitchy texture"),
    ("metallic", "metallic sheen"),
    ("organic", "organic textures"),
    ("synthetic", "synthetic textures"),
    ("crystalline", "crystalline clarity"),
    ("muddy", "thick muddy texture"),
    ("sparse", "sparse minimal texture"),
    ("dense", "dense layered texture"),
    ("lush", "lush rich texture"),
    ("airy", "light airy quality"),

    # Environmental
    ("rain", "rain-soaked ambience"),
    ("underwater", "submerged underwater feel"),
    ("cathedral", "cathedral reverb"),
    ("chamber", "intimate chamber space"),
    ("futuristic", "futuristic sci-fi"),
]

ARTIST_KEYWORDS = [
    # Classical & Film Score Composers
    ("john williams", ["cinematic", "orchestral"]),
    ("hans zimmer", ["cinematic", "epic"]),
    ("ennio morricone", ["cinematic", "western"]),
    ("mozart", ["classical", "elegant"]),
    ("beethoven", ["classical", "dramatic"]),
    ("bach", ["baroque", "intricate"]),
    ("debussy", ["impressionist", "delicate"]),
    ("tchaikovsky", ["romantic", "ballet"]),
    ("danny elfman", ["quirky", "cinematic"]),
    ("howard shore", ["epic", "fantasy"]),

    # Jazz
    ("miles davis", ["jazz", "modal"]),
    ("john coltrane", ["jazz", "spiritual"]),
    ("charlie parker", ["bebop", "fast"]),
    ("duke ellington", ["big band", "swing"]),
    ("thelonious monk", ["jazz", "angular"]),
    ("bill evans", ["piano jazz", "intimate"]),
    ("herbie hancock", ["fusion", "funk"]),
    ("weather report", ["fusion", "groovy"]),

    # Rock & Alternative
    ("led zeppelin", ["hard rock", "powerful"]),
    ("pink floyd", ["psychedelic", "atmospheric"]),
    ("radiohead", ["alternative", "experimental"]),
    ("nirvana", ["grunge", "raw"]),
    ("the beatles", ["pop rock", "melodic"]),
    ("david bowie", ["art rock", "theatrical"]),
    ("queen", ["rock", "operatic"]),

    # Electronic & Dance
    ("daft punk", ["electronic", "funk"]),
    ("aphex twin", ["idm", "experimental"]),
    ("deadmau5", ["progressive house"]),
    ("skrillex", ["dubstep", "aggressive"]),
    ("four tet", ["idm", "textural"]),
    ("boards of canada", ["idm", "nostalgic"]),
    ("kraftwerk", ["electronic", "robotic"]),
    ("jean-michel jarre", ["electronic", "spacious"]),

    # Hip-Hop & R&B
    ("kendrick lamar", ["hip hop", "lyrical"]),
    ("drake", ["hip hop", "melodic"]),
    ("travis scott", ["trap", "psychedelic"]),
    ("kanye west", ["hip hop", "experimental"]),
    ("the weeknd", ["r&b", "moody"]),
    ("sza", ["r&b", "soulful"]),
    ("anderson .paak", ["r&b", "funk"]),
    ("j dilla", ["boom bap", "soulful"]),

    # Pop
    ("michael jackson", ["pop", "funk"]),
    ("madonna", ["pop", "dance"]),
    ("prince", ["funk", "pop"]),
    ("ariana grande", ["pop", "vocal"]),
    ("taylor swift", ["pop", "storytelling"]),
    ("wham", ["80s pop", "upbeat"]),

    # Ambient & Experimental
    ("brian eno", ["ambient", "atmospheric"]),
    ("stars of the lid", ["ambient", "drone"]),
    ("william basinski", ["ambient", "decay"]),
    ("fennesz", ["experimental", "textural"]),
]

TIME_SIGNATURE_KEYWORDS = [
    ("waltz", "3/4"),
    ("3/4", "3/4"),
    ("6/8", "6/8"),
    ("5/4", "5/4"),
    ("7/8", "7/8"),
    ("9/8", "9/8"),
    ("11/8", "11/8"),
    ("odd meter", "7/8"),
    ("odd time", "7/8"),
]

MODE_KEYWORDS = [
    ("dorian", "Dorian"),
    ("phrygian", "Phrygian"),
    ("lydian", "Lydian"),
    ("mixolydian", "Mixolydian"),
    ("aeolian", "Aeolian"),
    ("locrian", "Locrian"),
    ("minor", "minor"),
    ("major", "major"),
]

DEFAULT_INSTRUMENTATION = ["blended instrumentation"]
DEFAULT_TEXTURE = "immersive atmosphere"

# Era-specific production descriptors for instrumentation flavoring
ERA_PRODUCTION_DESCRIPTORS = {
    "60s": {
        "flavor": ["analog warmth", "tape saturation", "vintage compression", "mono feel"],
        "instruments": {
            "guitar": "60s clean guitar",
            "synth": "early analog synth",
            "drum": "vintage jazz drums",
            "bass": "motown bass",
            "organ": "60s hammond organ",
        }
    },
    "70s": {
        "flavor": ["warm analog", "lush production", "funk grooves", "disco shimmer"],
        "instruments": {
            "guitar": "70s funk guitar",
            "synth": "70s analog synth",
            "drum": "70s disco drums",
            "bass": "funk bass",
            "string": "70s string section",
        }
    },
    "80s": {
        "flavor": ["gated reverb", "digital synths", "big drums", "chorus effects"],
        "instruments": {
            "guitar": "80s power guitar",
            "synth": "80s digital synth",
            "drum": "gated 80s drums",
            "bass": "80s synth bass",
            "vocal": "80s vocals",
        }
    },
    "90s": {
        "flavor": ["grunge aesthetics", "lo-fi production", "sampled breaks", "digital clarity"],
        "instruments": {
            "guitar": "90s alternative guitar",
            "synth": "90s synth pads",
            "drum": "90s breakbeat drums",
            "bass": "grunge bass",
            "vocal": "90s vocal production",
        }
    },
    "2000s": {
        "flavor": ["compressed modern", "digital precision", "sidechained pumping"],
        "instruments": {
            "guitar": "modern electric guitar",
            "synth": "2000s synth",
            "drum": "2000s programmed drums",
            "bass": "modern bass",
            "vocal": "auto-tuned vocals",
        }
    },
    "2010s": {
        "flavor": ["trap hi-hats", "EDM build-drops", "dubstep wobbles", "maximized loudness"],
        "instruments": {
            "guitar": "modern processed guitar",
            "synth": "modern synth",
            "drum": "trap drums",
            "bass": "sub-bass",
            "vocal": "modern vocal production",
        }
    },
    "modern": {
        "flavor": ["spatial audio", "pristine clarity", "surgical EQ", "contemporary production"],
        "instruments": {
            "guitar": "contemporary guitar",
            "synth": "contemporary synth",
            "drum": "modern drums",
            "bass": "contemporary bass",
            "vocal": "contemporary vocals",
        }
    },
}

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
    ),
    "vocals": ("vocal", "voice", "singer", "choir", "chant", "lyric"),
}

DEFAULT_LAYER_FALLBACKS = {
    "rhythm": [
        "tight drums", "organic percussion", "808 drum machine", "vintage drum machine",
        "driving drums", "crisp snare", "powerful kick", "shimmering hi-hats",
        "hand claps", "shaker", "tambourine", "congas", "bongos", "tabla",
        "taiko drums", "orchestral timpani", "wooden marimba", "syncopated breakbeat",
        "four-on-the-floor kick", "swinging drums with brushes"
    ],
    "bass": [
        "pulsing bass", "sub bass swell", "deep bass", "808 sub bass",
        "walking upright bass", "electric bass groove", "synth bass", "wobble bass",
        "funk bass", "slap bass", "rumbling sub bass", "plucky bass",
        "distorted bass", "smooth bass line", "jazz walking bass",
        "grounding low end", "heavy sub frequencies"
    ],
    "harmony": [
        "lush keys", "stacked synth pads", "warm piano", "vintage rhodes",
        "layered strings", "acoustic guitar", "electric guitar rhythm", "power chords",
        "hammond organ", "mellotron", "vibraphone", "string section",
        "brass ensemble", "woodwind harmony", "synth chords", "arpeggiated synths",
        "piano comping", "guitar chords", "lush pads", "orchestral strings"
    ],
    "lead": [
        "expressive guitar lead", "soulful brass line", "saxophone", "trumpet",
        "violin solo", "synth lead", "flute", "electric guitar solo",
        "vocal melodies", "clarinet", "oboe", "trombone solo",
        "soaring lead synth", "distorted guitar", "french horn",
        "harmonica", "lead vocals", "melodic lines"
    ],
    "textures": [
        "airy ambient swells", "granular noise beds", "atmospheric pads",
        "reverberant textures", "field recordings", "white noise sweeps",
        "vinyl crackle", "tape hiss", "ambient guitar", "drone",
        "choir pads", "string tremolo", "cymbal swells", "feedback",
        "glitchy textures", "reverb tails", "sustained tones", "spectral wash"
    ],
    "vocals": [
        "wordless vocal pads", "ethereal vocals", "vocal harmonies",
        "choir voices", "soulful vocals", "falsetto", "rap vocals",
        "sung melodies", "vocal chops", "vocoded vocals"
    ],
}

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

    def __init__(self):
        self.genre_classifier = GenreClassifier()

    def build_plan(self, request: GenerationRequest) -> CompositionPlan:
        if float(request.duration_seconds) >= LONG_FORM_THRESHOLD:
            return self._build_long_form_plan(request)
        return self._build_short_form_plan(request)

    def _build_long_form_plan(self, request: GenerationRequest) -> CompositionPlan:
        seconds_total = float(max(request.duration_seconds, LONG_FORM_THRESHOLD))

        # Classify genre from prompt
        genre_profile = self.genre_classifier.classify(request.prompt)
        vocals = self.genre_classifier.detect_vocals(request.prompt)

        # Determine time signature (genre-aware or default)
        time_signature = _detect_time_signature(request.prompt, genre_profile)

        templates = list(_select_long_templates(seconds_total))
        beats_per_bar = _beats_per_bar(time_signature)
        tempo_hint = _tempo_hint(seconds_total, templates, beats_per_bar)

        # Use genre tempo constraints
        tempo_bpm = _select_tempo_for_genre(tempo_hint, genre_profile)
        effective_tempo = max(tempo_bpm, genre_profile.tempo_range[0])
        seconds_per_bar = (60.0 / float(effective_tempo)) * float(beats_per_bar)

        bars = _allocate_bars(templates, seconds_total, seconds_per_bar)

        descriptor = _build_theme_descriptor(request.prompt, templates, genre_profile, vocals)
        palette = _categorise_instrumentation(descriptor, request.prompt, genre_profile)
        orchestrations = _plan_orchestrations(templates, palette, genre_profile, vocals)

        base_seed = (
            request.seed if request.seed is not None else _deterministic_seed(request.prompt)
        )
        musical_key, mode = _select_key_and_mode(base_seed, genre_profile)

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
            time_signature=time_signature,
            key=musical_key,
            mode=mode,
            total_bars=total_bars,
            total_duration_seconds=total_duration,
            theme=descriptor,
            sections=sections,
        )

    def _build_short_form_plan(self, request: GenerationRequest) -> CompositionPlan:
        seconds_total = float(max(request.duration_seconds, SHORT_MIN_TOTAL_SECONDS))

        # Classify genre from prompt
        genre_profile = self.genre_classifier.classify(request.prompt)
        vocals = self.genre_classifier.detect_vocals(request.prompt)

        # Determine time signature (genre-aware or default)
        time_signature = _detect_time_signature(request.prompt, genre_profile)

        templates = list(_select_short_templates(seconds_total))
        beats_per_bar = _beats_per_bar(time_signature)
        total_weight = float(sum(template.base_bars for template in templates) or 1)
        raw_tempo = int(round(240.0 * total_weight / seconds_total)) if seconds_total > 0 else 90

        # Use genre tempo constraints
        tempo_bpm = _select_tempo_for_genre(raw_tempo, genre_profile)
        effective_tempo = max(tempo_bpm, genre_profile.tempo_range[0])
        seconds_per_bar = (60.0 / float(effective_tempo)) * float(beats_per_bar)

        descriptor = _build_theme_descriptor(request.prompt, templates, genre_profile, vocals)
        palette = _categorise_instrumentation(descriptor, request.prompt, genre_profile)
        orchestrations = _plan_orchestrations(templates, palette, genre_profile, vocals)

        base_seed = (
            request.seed if request.seed is not None else _deterministic_seed(request.prompt)
        )
        musical_key, mode = _select_key_and_mode(base_seed, genre_profile)

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
            bar_count = int(round(section_seconds / seconds_per_bar)) if seconds_per_bar > 0 else template.base_bars
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
            time_signature=time_signature,
            key=musical_key,
            mode=mode,
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


def _select_key_and_mode(seed: int, genre_profile: GenreProfile) -> tuple[str, Optional[str]]:
    """
    Select musical key and mode based on genre profile and seed.

    Returns:
        tuple of (key, mode) where mode is None for standard major/minor or modal name
    """
    # Use genre-specific typical keys if available and high confidence
    if genre_profile.confidence >= 0.7 and genre_profile.typical_keys:
        keys = genre_profile.typical_keys
        key = keys[seed % len(keys)]
    else:
        # Fallback to default key selection
        key = _select_key(seed)

    # Detect mode from key string or genre defaults
    mode = None
    if any(modal in key.lower() for modal in ["dorian", "phrygian", "lydian", "mixolydian", "aeolian", "locrian"]):
        # Extract mode from key string
        for modal_name in ["Dorian", "Phrygian", "Lydian", "Mixolydian", "Aeolian", "Locrian"]:
            if modal_name in key:
                mode = modal_name
                break
    elif "modal" in key.lower():
        # Generic modal indication - choose mode based on genre
        if genre_profile.genre == "jazz":
            mode = "Dorian"
        elif genre_profile.genre == "world":
            mode = "Phrygian"
        elif genre_profile.genre == "ambient":
            mode = "Lydian"
        else:
            mode = "Dorian"  # Default modal flavor
    elif "minor" in key.lower():
        mode = "minor"
    elif "major" in key.lower():
        mode = "major"

    return key, mode


def _detect_time_signature(prompt: str, genre_profile: GenreProfile) -> str:
    """
    Detect time signature from prompt keywords or genre profile.
    """
    prompt_lower = prompt.lower()

    # Check for explicit time signature keywords in prompt
    for keyword, time_sig in TIME_SIGNATURE_KEYWORDS:
        if keyword in prompt_lower:
            return time_sig

    # Use genre's preferred meters if high confidence
    if genre_profile.confidence >= 0.6 and genre_profile.meters:
        # Prefer first meter from genre (usually the most common)
        return genre_profile.meters[0]

    # Default fallback
    return DEFAULT_TIME_SIGNATURE


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


def _select_tempo_for_genre(raw_tempo: int, genre_profile: GenreProfile) -> int:
    """
    Select tempo constrained by genre's typical tempo range.

    Uses genre's typical_tempos if high confidence, otherwise uses tempo_range.
    Falls back to standard _select_tempo if low confidence.
    """
    if genre_profile.confidence < 0.6:
        # Low confidence, use standard tempo selection
        return _select_tempo(raw_tempo)

    min_tempo, max_tempo = genre_profile.tempo_range

    # Clamp to genre range
    clamped = max(min_tempo, min(max_tempo, raw_tempo))

    # Prefer genre's typical tempos if available
    if genre_profile.typical_tempos:
        # Find closest typical tempo
        closest = min(genre_profile.typical_tempos, key=lambda t: abs(t - clamped))
        # Use typical tempo if within reasonable range
        if abs(closest - clamped) <= 20:
            return closest

    # Otherwise find best tempo within genre range
    best = clamped
    best_error = abs(raw_tempo - best)
    for delta in range(1, 20):
        for candidate in (clamped - delta, clamped + delta):
            if candidate < min_tempo or candidate > max_tempo:
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
    genre_profile: GenreProfile,
    vocals: VocalCharacteristics,
) -> ThemeDescriptor:
    prompt_lower = prompt.lower()
    instrumentation = _extract_keywords(prompt_lower, INSTRUMENT_KEYWORDS)

    # CRITICAL FIX: Populate from genre profile instead of generic fallback
    if not instrumentation:
        if genre_profile.confidence >= 0.4:  # Lower threshold for better variety
            instrumentation = _extract_genre_instruments(genre_profile)
        else:
            instrumentation = list(DEFAULT_INSTRUMENTATION)

    rhythm = _derive_rhythm(prompt_lower, templates, genre_profile)
    motif = _derive_motif(prompt)
    texture = _derive_texture(prompt_lower, instrumentation)
    dynamic_curve = [
        _dynamic_label(template.role, template.energy, index, len(templates))
        for index, template in enumerate(templates)
    ]

    # Derive harmonic complexity from genre
    harmonic_complexity = _derive_harmonic_complexity(genre_profile, prompt_lower)

    return ThemeDescriptor(
        motif=motif,
        instrumentation=instrumentation,
        rhythm=rhythm,
        dynamic_curve=dynamic_curve,
        texture=texture,
        genre=genre_profile.genre,
        genre_confidence=genre_profile.confidence,
        era=genre_profile.era,
        subgenre=genre_profile.subgenre,
        harmonic_complexity=harmonic_complexity,
    )


def _extract_keywords(prompt_lower: str, mapping: list[tuple[str, str]]) -> list[str]:
    results: list[str] = []
    for keyword, label in mapping:
        if keyword in prompt_lower and label not in results:
            results.append(label)
    return results


def _extract_genre_instruments(genre_profile: GenreProfile) -> list[str]:
    """
    Extract representative instruments from genre profile when prompt has no explicit instruments.

    Takes first 2 instruments from each category to create diverse palette.
    """
    result: list[str] = []

    # Priority order: harmony, lead, rhythm, bass, textures (most musical variety)
    for category in ["harmony", "lead", "rhythm", "bass", "textures"]:
        instruments = genre_profile.instruments.get(category, [])
        if instruments:
            # Take first 2 from each category for variety
            result.extend(instruments[:2])

    # Add vocals if genre emphasizes them
    if genre_profile.vocal_prevalence > 0.5:
        vocal_instruments = genre_profile.instruments.get("vocals", [])
        if vocal_instruments:
            result.append(vocal_instruments[0])

    return result if result else ["blended instrumentation"]


def _derive_rhythm(prompt_lower: str, templates: List[SectionTemplate], genre_profile: GenreProfile) -> str:
    # Check prompt keywords first
    for keyword, label in RHYTHM_KEYWORDS:
        if keyword in prompt_lower:
            return label

    # Use genre-specific rhythm descriptors
    if genre_profile.genre == "jazz":
        return "swinging groove"
    elif genre_profile.genre == "hip_hop":
        return "laid-back hip hop beat"
    elif genre_profile.genre == "electronic":
        if genre_profile.subgenre == "house":
            return "four-on-the-floor pulse"
        elif genre_profile.subgenre == "techno":
            return "driving techno rhythm"
        return "electronic pulse"
    elif genre_profile.genre == "rock":
        return "driving rock beat"
    elif genre_profile.genre == "ambient":
        return "subtle rhythmic texture"
    elif genre_profile.genre == "classical":
        return "orchestral rhythmic foundation"

    # Fallback to energy-based rhythm
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


def _derive_texture(prompt_lower: str, instrumentation: list[str]) -> str:
    for keyword, label in TEXTURE_KEYWORDS:
        if keyword in prompt_lower:
            return label
    if instrumentation:
        return f"focused blend of {', '.join(instrumentation[:2])}"
    return DEFAULT_TEXTURE


def _derive_harmonic_complexity(genre_profile: GenreProfile, prompt_lower: str) -> Optional[str]:
    """
    Derive harmonic complexity descriptor based on genre and prompt.

    Returns genre-specific harmonic vocabulary or None for simple harmony.
    """
    # Check for explicit harmonic keywords in prompt
    if "extended" in prompt_lower or "complex harmony" in prompt_lower or "jazz chords" in prompt_lower:
        return "extended voicings and altered dominants"
    if "simple" in prompt_lower or "basic chords" in prompt_lower:
        return "simple triadic harmony"

    # Genre-specific harmonic vocabulary
    if genre_profile.genre == "jazz":
        return "extended voicings, modal interchange, and altered dominants"
    elif genre_profile.genre == "classical":
        return "functional harmonic progression with cadential motion"
    elif genre_profile.genre == "film_score":
        return "cinematic harmonic swells and orchestral voicings"
    elif genre_profile.genre == "r_and_b":
        return "soulful chord extensions and smooth voice leading"
    elif genre_profile.genre == "ambient":
        return "sustained harmonic fields and drone-like textures"
    elif genre_profile.genre == "world":
        if "modal" in prompt_lower:
            return "modal harmonic structures"
        return "ethnic scale patterns and drone foundations"
    elif genre_profile.genre == "experimental":
        return "atonal clusters and spectral harmonies"
    elif genre_profile.genre in ("rock", "electronic", "pop"):
        # These genres typically use simpler harmony
        return None

    return None


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


def _apply_era_modifier(instrument: str, era: Optional[str]) -> str:
    """
    Apply era-specific modifiers to instrument labels.

    Args:
        instrument: Base instrument description
        era: Detected era (60s, 70s, 80s, 90s, 2000s, 2010s, modern)

    Returns:
        Modified instrument string with era flavor
    """
    if not era or era not in ERA_PRODUCTION_DESCRIPTORS:
        return instrument

    era_data = ERA_PRODUCTION_DESCRIPTORS[era]
    instrument_modifiers = era_data.get("instruments", {})

    instrument_lower = instrument.lower()

    # Check for instrument type matches and apply era modifier
    for inst_type, era_label in instrument_modifiers.items():
        if inst_type in instrument_lower:
            # Replace generic description with era-specific version
            return era_label

    # If no specific match, prepend era to instrument
    if era in ["60s", "70s", "80s", "90s"]:
        return f"{era} {instrument}"

    return instrument


def _categorise_instrumentation(
    descriptor: ThemeDescriptor,
    prompt: str,
    genre_profile: GenreProfile,
) -> dict[str, list[str]]:
    palette = {category: [] for category in CATEGORY_KEYWORDS}

    # First, add prompt-extracted instruments with era flavoring
    era = descriptor.era
    for label in descriptor.instrumentation:
        # Apply era modifier if era is detected
        if era:
            label = _apply_era_modifier(label, era)

        lowered = label.lower()
        matched = False
        for category, keywords in CATEGORY_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                palette[category].append(label)
                matched = True
        if not matched:
            palette.setdefault("textures", []).append(label)

    # Merge with genre-specific instruments (lowered threshold for better variety)
    if genre_profile.confidence >= 0.5:  # Lowered from 0.6
        for category, genre_instruments in genre_profile.instruments.items():
            # Add genre instruments that aren't already present
            existing_lower = [inst.lower() for inst in palette.get(category, [])]
            for genre_inst in genre_instruments:
                if genre_inst.lower() not in existing_lower:
                    palette[category].append(genre_inst)

    prompt_lower = prompt.lower()
    if any(keyword in prompt_lower for keyword in CATEGORY_KEYWORDS["vocals"]):
        palette.setdefault("vocals", []).append("expressive vocals")

    # Use genre-appropriate fallbacks when palette is empty
    for category, defaults in DEFAULT_LAYER_FALLBACKS.items():
        existing = palette.get(category, [])
        if category == "vocals" and not existing:
            palette[category] = []
        else:
            # Prefer genre instruments as fallback if available
            genre_fallback = genre_profile.instruments.get(category, [])
            fallback = genre_fallback if genre_fallback else defaults
            palette[category] = _dedupe(existing) if existing else list(fallback)

    return palette


def _plan_orchestrations(
    templates: List[SectionTemplate],
    palette: dict[str, list[str]],
    genre_profile: GenreProfile,
    vocals: VocalCharacteristics,
) -> List[SectionOrchestration]:
    orchestrations: List[SectionOrchestration] = []

    # Use genre layer counts if high confidence, otherwise use role-based defaults
    use_genre_layers = genre_profile.confidence >= 0.7

    for index, template in enumerate(templates):
        if use_genre_layers:
            # Start with genre's layer profile
            counts = dict(genre_profile.layers)

            # Adjust vocal count based on vocal prevalence and detection
            if vocals.has_vocals or genre_profile.vocal_prevalence > 0.7:
                # Increase vocals for chorus/development sections
                if template.role in (SectionRole.CHORUS, SectionRole.DEVELOPMENT):
                    counts["vocals"] = max(counts.get("vocals", 0), 2)
                elif template.role not in (SectionRole.INTRO, SectionRole.OUTRO):
                    counts["vocals"] = max(counts.get("vocals", 0), 1)
            else:
                # Reduce or eliminate vocals if not detected and genre doesn't emphasize them
                if genre_profile.vocal_prevalence < 0.3:
                    counts["vocals"] = 0

        else:
            # Use traditional role-based profile
            counts = SECTION_LAYER_PROFILE.get(template.role, SECTION_LAYER_PROFILE.get("default", {}))

        orchestrations.append(_build_orchestration(counts, palette, offset=index))

    return orchestrations


def _build_orchestration(
    counts: dict[str, int],
    palette: dict[str, list[str]],
    *,
    offset: int,
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
                    items = _cycled_slice(source, count, offset)
            else:
                items = _cycled_slice(source, count, offset)
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
