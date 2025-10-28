use crate::types::{
    CompositionPlan, CompositionSection, SectionEnergy, SectionOrchestration, SectionRole,
    ThemeDescriptor,
};
use std::collections::HashSet;

const PLAN_VERSION: &str = "v3";
const MIN_TEMPO: u16 = 68;
const MAX_TEMPO: u16 = 128;
const DEFAULT_TIME_SIGNATURE: &str = "4/4";
const SHORT_MIN_TOTAL_SECONDS: f32 = 2.0;
const SHORT_MIN_SECTION_SECONDS: f32 = 2.0;
const LONG_FORM_THRESHOLD: f32 = 90.0;
const LONG_MIN_SECTION_SECONDS: f32 = 16.0;

const ADD_PRIORITY: &[SectionRole] = &[
    SectionRole::Chorus,
    SectionRole::Motif,
    SectionRole::Bridge,
    SectionRole::Development,
    SectionRole::Intro,
    SectionRole::Outro,
    SectionRole::Resolution,
];

const REMOVE_PRIORITY: &[SectionRole] = &[
    SectionRole::Outro,
    SectionRole::Intro,
    SectionRole::Bridge,
    SectionRole::Resolution,
    SectionRole::Development,
    SectionRole::Motif,
    SectionRole::Chorus,
];

const INSTRUMENT_KEYWORDS: &[(&str, &str)] = &[
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
];

const RHYTHM_KEYWORDS: &[(&str, &str)] = &[
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
];

const TEXTURE_KEYWORDS: &[(&str, &str)] = &[
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
];

const DEFAULT_INSTRUMENTATION: &[&str] = &[
    "blended instrumentation",
    "layered synth textures",
    "hybrid acoustic-electric palette",
    "ensemble interplay",
];
const DEFAULT_TEXTURE: &str = "immersive atmosphere";

struct LayerCounts {
    rhythm: u8,
    bass: u8,
    harmony: u8,
    lead: u8,
    textures: u8,
    vocals: u8,
}

impl LayerCounts {
    const fn new(rhythm: u8, bass: u8, harmony: u8, lead: u8, textures: u8, vocals: u8) -> Self {
        Self { rhythm, bass, harmony, lead, textures, vocals }
    }
}

#[derive(Default, Clone)]
struct InstrumentPalette {
    rhythm: Vec<String>,
    bass: Vec<String>,
    harmony: Vec<String>,
    lead: Vec<String>,
    textures: Vec<String>,
    vocals: Vec<String>,
}

#[derive(Default)]
struct PaletteOffsets {
    rhythm: usize,
    bass: usize,
    harmony: usize,
    lead: usize,
    textures: usize,
    vocals: usize,
}

struct GenreLayers {
    rhythm: &'static [&'static str],
    bass: &'static [&'static str],
    harmony: &'static [&'static str],
    lead: &'static [&'static str],
    textures: &'static [&'static str],
    vocals: &'static [&'static str],
}

struct GenreProfile {
    keywords: &'static [&'static str],
    instrumentation: &'static [&'static str],
    rhythm: Option<&'static str>,
    texture: Option<&'static str>,
    layers: GenreLayers,
}

const GENRE_PROFILES: &[GenreProfile] = &[
    GenreProfile {
        keywords: &["lofi", "lo-fi", "chillhop", "chill hop", "study beats"],
        instrumentation: &[
            "dusty electric piano",
            "tape-saturated drum loop",
            "warm bass guitar",
            "soft nylon guitar",
        ],
        rhythm: Some("laid-back hip hop beat"),
        texture: Some("hazy lo-fi ambience"),
        layers: GenreLayers {
            rhythm: &["dusty drum loop", "lazy snare with brushes"],
            bass: &["mellow sub bass", "upright bass with tape wobble"],
            harmony: &["noisy electric piano chords", "soft guitar voicings"],
            lead: &["wistful synth lead", "gentle vibraphone motifs"],
            textures: &["vinyl crackle", "nocturnal rain ambience"],
            vocals: &["hummed vocal chops"],
        },
    },
    GenreProfile {
        keywords: &["jazz", "bebop", "swing", "hard bop", "cool jazz"],
        instrumentation: &[
            "upright bass",
            "brush drum kit",
            "extended jazz piano",
            "saxophone lead",
        ],
        rhythm: Some("swinging groove"),
        texture: Some("smoky lounge atmosphere"),
        layers: GenreLayers {
            rhythm: &["brush drum kit", "ride cymbal patterns"],
            bass: &["walking upright bass"],
            harmony: &["extended jazz piano voicings", "comping guitar chords"],
            lead: &["saxophone lead", "muted trumpet phrases"],
            textures: &["club room ambience", "late-night crowd murmur"],
            vocals: &["scat vocal ad-libs"],
        },
    },
    GenreProfile {
        keywords: &["rock", "alt rock", "indie rock", "garage rock", "post-punk"],
        instrumentation: &[
            "distorted electric guitars",
            "live drum kit",
            "electric bass",
            "anthemic vocals",
        ],
        rhythm: Some("driving rock beat"),
        texture: Some("amplified stage energy"),
        layers: GenreLayers {
            rhythm: &["punchy live drums", "crashing cymbals"],
            bass: &["gritty electric bass"],
            harmony: &["crunchy rhythm guitars", "stadium power chords"],
            lead: &["soaring guitar lead", "anthemic vocal hooks"],
            textures: &["amp feedback swells", "crowd reverb wash"],
            vocals: &["anthemic rock vocals"],
        },
    },
    GenreProfile {
        keywords: &["metal", "thrash", "doom", "heavy metal", "prog metal"],
        instrumentation: &[
            "double kick drums",
            "down-tuned rhythm guitars",
            "growling bass",
            "aggressive vocals",
        ],
        rhythm: Some("relentless double-kick drive"),
        texture: Some("dense wall of distortion"),
        layers: GenreLayers {
            rhythm: &["blast beat drums", "double-kick assaults"],
            bass: &["growling electric bass"],
            harmony: &["down-tuned rhythm guitars", "chugging power chords"],
            lead: &["shredded guitar leads", "screaming harmonised solos"],
            textures: &["amp roar", "crowd roar tails"],
            vocals: &["harsh metal vocals"],
        },
    },
    GenreProfile {
        keywords: &["orchestral", "symphonic", "film score", "cinematic", "soundtrack"],
        instrumentation: &[
            "string ensemble",
            "brass section",
            "woodwind choir",
            "concert percussion",
        ],
        rhythm: Some("expansive cinematic pulse"),
        texture: Some("lush orchestral hall"),
        layers: GenreLayers {
            rhythm: &["gran cassa hits", "rolling orchestral percussion"],
            bass: &["double basses", "low brass swells"],
            harmony: &["legato string ensemble", "warm horn chords"],
            lead: &["solo violin", "trumpet fanfare"],
            textures: &["concert hall reverberation", "choir pads"],
            vocals: &["wordless choir"],
        },
    },
    GenreProfile {
        keywords: &["folk", "acoustic", "bluegrass", "roots", "americana", "country"],
        instrumentation: &[
            "acoustic guitar",
            "mandolin",
            "upright bass",
            "fiddle melodies",
        ],
        rhythm: Some("organic strummed pulse"),
        texture: Some("warm rustic atmosphere"),
        layers: GenreLayers {
            rhythm: &["brushy folk kit", "hand clap patterns"],
            bass: &["plucked upright bass"],
            harmony: &["fingerpicked acoustic guitar", "sparkling mandolin arpeggios"],
            lead: &["fiddle leads", "slide guitar phrases"],
            textures: &["barn room reverb", "nature ambience"],
            vocals: &["stacked folk harmonies"],
        },
    },
    GenreProfile {
        keywords: &["ambient", "soundscape", "drone", "meditation", "ethereal", "atmospheric"],
        instrumentation: &[
            "evolving synth pads",
            "granular textures",
            "soft mallet swells",
            "choral atmospheres",
        ],
        rhythm: Some("floating pulse"),
        texture: Some("expansive ambient haze"),
        layers: GenreLayers {
            rhythm: &["subtle percussive pulses", "soft percussion swells"],
            bass: &["deep sine bass swells"],
            harmony: &["evolving synth pads", "glacial piano chords"],
            lead: &["breathy vocal motifs", "glass harmonica lines"],
            textures: &["wind soundscape", "distant shimmer"],
            vocals: &["celestial vocal layers"],
        },
    },
    GenreProfile {
        keywords: &[
            "electronic",
            "edm",
            "house",
            "techno",
            "trance",
            "club",
            "electro",
            "synthwave",
            "synth-pop",
        ],
        instrumentation: &[
            "punchy electronic kick",
            "rolling bass synth",
            "arpeggiated synths",
            "bright lead plucks",
        ],
        rhythm: Some("four-on-the-floor pulse"),
        texture: Some("neon club atmosphere"),
        layers: GenreLayers {
            rhythm: &["four-on-the-floor kick", "syncopated hi-hats"],
            bass: &["rolling bass synth", "acid bassline"],
            harmony: &["wide pad chords", "lush supersaw stacks"],
            lead: &["glittering synth lead", "vocoder hooks"],
            textures: &["club crowd ambience", "side-chained pads"],
            vocals: &["processed vocal chops"],
        },
    },
    GenreProfile {
        keywords: &["hip hop", "trap", "boom bap", "rap", "grime"],
        instrumentation: &[
            "knocking drum machine",
            "808 bass",
            "sampled keys",
            "vocal chops",
        ],
        rhythm: Some("head-nod hip hop pulse"),
        texture: Some("urban night energy"),
        layers: GenreLayers {
            rhythm: &["punchy drum machine groove", "crisp claps"],
            bass: &["808 bass", "subby bass drops"],
            harmony: &["minor key piano loops", "sampled soul chops"],
            lead: &["pitched vocal chops", "synth brass stabs"],
            textures: &["street ambience", "vinyl noise"],
            vocals: &["rap ad-libs"],
        },
    },
];

fn match_genre_profile(prompt_lower: &str) -> Option<&'static GenreProfile> {
    let mut best: Option<&GenreProfile> = None;
    let mut best_len = 0usize;
    for profile in GENRE_PROFILES {
        for keyword in profile.keywords {
            if prompt_lower.contains(keyword) && keyword.len() > best_len {
                best = Some(profile);
                best_len = keyword.len();
            }
        }
    }
    best
}

fn directives_for_role(role: &SectionRole) -> (Option<String>, Vec<String>, Option<String>) {
    match role {
        SectionRole::Intro => (
            Some("foreshadow motif".to_string()),
            vec!["texture".to_string(), "register preview".to_string()],
            Some("establish tonic pedal".to_string()),
        ),
        SectionRole::Motif => (
            Some("state motif".to_string()),
            vec!["motif fidelity".to_string()],
            Some("open cadence".to_string()),
        ),
        SectionRole::Chorus => (
            Some("amplify motif".to_string()),
            vec![
                "dynamics".to_string(),
                "call-and-response".to_string(),
                "countermelody".to_string(),
            ],
            Some("anthemic cadence".to_string()),
        ),
        SectionRole::Development => (
            Some("develop motif".to_string()),
            vec!["rhythm".to_string(), "harmony".to_string(), "counterpoint".to_string()],
            None,
        ),
        SectionRole::Bridge => (
            Some("modulate motif".to_string()),
            vec!["harmony".to_string(), "timbre".to_string()],
            Some("pivot modulation".to_string()),
        ),
        SectionRole::Resolution => (
            Some("resolve motif".to_string()),
            vec!["harmony".to_string(), "dynamics".to_string()],
            Some("authentic cadence".to_string()),
        ),
        SectionRole::Outro => (
            Some("dissolve motif".to_string()),
            vec!["texture".to_string(), "space".to_string()],
            Some("fade tonic drone".to_string()),
        ),
    }
}

struct SectionTemplate {
    role: SectionRole,
    label: &'static str,
    energy: SectionEnergy,
    base_bars: u16,
    min_bars: u16,
    max_bars: u16,
    prompt_template: &'static str,
    transition: Option<&'static str>,
}

#[derive(Debug, Default)]
pub struct CompositionPlanner;

impl CompositionPlanner {
    pub fn new() -> Self {
        Self
    }

    pub fn build_plan(
        &self,
        prompt: &str,
        duration_seconds: u8,
        seed: Option<u64>,
    ) -> CompositionPlan {
        if (duration_seconds as f32) >= LONG_FORM_THRESHOLD {
            return self.build_long_form_plan(prompt, duration_seconds, seed);
        }
        self.build_short_form_plan(prompt, duration_seconds, seed)
    }

    fn build_long_form_plan(
        &self,
        prompt: &str,
        duration_seconds: u8,
        seed: Option<u64>,
    ) -> CompositionPlan {
        let seconds_total = f32::max(duration_seconds as f32, LONG_FORM_THRESHOLD);
        let templates = select_long_templates(seconds_total);
        let beats_per_bar = beats_per_bar(DEFAULT_TIME_SIGNATURE);
        let tempo_hint = tempo_hint(seconds_total, &templates, beats_per_bar);
        let tempo_bpm = select_tempo(tempo_hint);
        let seconds_per_bar = (60.0 / tempo_bpm as f32) * beats_per_bar as f32;

        let bars = allocate_bars(&templates, seconds_total, seconds_per_bar);

        let base_seed = seed.unwrap_or_else(|| deterministic_seed(prompt));
        let prompt_lower = prompt.to_lowercase();
        let profile = match_genre_profile(&prompt_lower);

        let descriptor = build_theme_descriptor(prompt, &templates, profile, base_seed);
        let palette = categorise_instrumentation(&descriptor, prompt, profile, base_seed);
        let offsets = palette_offsets(&palette, base_seed);
        let orchestrations = plan_orchestrations(&templates, &palette, &offsets);
        let key = select_key(base_seed);

        let mut sections: Vec<CompositionSection> = Vec::with_capacity(templates.len());
        let mut total_bars: u16 = 0;

        for (index, template) in templates.iter().enumerate() {
            let arrangement = describe_orchestration(&orchestrations[index]);
            let section_prompt =
                render_prompt(template.prompt_template, prompt, &descriptor, index, &arrangement);
            let (motif_directive, variation_axes, cadence_hint) =
                directives_for_role(&template.role);
            let bar_count = bars[index].max(1);
            total_bars += bar_count;
            let mut target_seconds = bar_count as f32 * seconds_per_bar;
            if target_seconds < LONG_MIN_SECTION_SECONDS {
                target_seconds = LONG_MIN_SECTION_SECONDS;
            }

            sections.push(CompositionSection {
                section_id: format!("s{:02}", index),
                role: template.role.clone(),
                label: template.label.to_string(),
                prompt: section_prompt,
                bars: bar_count as u8,
                target_seconds,
                energy: template.energy.clone(),
                model_id: None,
                seed_offset: Some(index as i32),
                transition: template.transition.map(|text| text.to_string()),
                motif_directive,
                variation_axes,
                cadence_hint,
                orchestration: orchestrations[index].clone(),
            });
        }

        CompositionPlan {
            version: PLAN_VERSION.to_string(),
            tempo_bpm,
            time_signature: DEFAULT_TIME_SIGNATURE.to_string(),
            key,
            total_bars,
            total_duration_seconds: sections.iter().map(|s| s.target_seconds).sum(),
            theme: Some(descriptor.clone()),
            sections,
        }
    }

    fn build_short_form_plan(
        &self,
        prompt: &str,
        duration_seconds: u8,
        seed: Option<u64>,
    ) -> CompositionPlan {
        let seconds_total = f32::max(duration_seconds as f32, SHORT_MIN_TOTAL_SECONDS);
        let templates = select_short_templates(seconds_total);
        let beats_per_bar = beats_per_bar(DEFAULT_TIME_SIGNATURE);
        let total_weight: u32 = templates.iter().map(|tpl| tpl.base_bars as u32).sum();
        let weight =
            if total_weight == 0 { (templates.len().max(1) as u32) * 4 } else { total_weight };
        let raw_tempo = if seconds_total > 0.0 {
            (240.0 * weight as f32 / seconds_total).round() as u16
        } else {
            90
        };
        let tempo_bpm = select_tempo(raw_tempo);
        let seconds_per_bar = (60.0 / tempo_bpm as f32) * beats_per_bar as f32;

        let base_seed = seed.unwrap_or_else(|| deterministic_seed(prompt));
        let prompt_lower = prompt.to_lowercase();
        let profile = match_genre_profile(&prompt_lower);

        let descriptor = build_theme_descriptor(prompt, &templates, profile, base_seed);
        let palette = categorise_instrumentation(&descriptor, prompt, profile, base_seed);
        let offsets = palette_offsets(&palette, base_seed);
        let orchestrations = plan_orchestrations(&templates, &palette, &offsets);
        let key = select_key(base_seed);

        let mut sections: Vec<CompositionSection> = Vec::with_capacity(templates.len());
        let mut total_bars: u16 = 0;
        let mut total_duration = 0.0f32;

        for (index, template) in templates.iter().enumerate() {
            let arrangement = describe_orchestration(&orchestrations[index]);
            let section_prompt =
                render_prompt(template.prompt_template, prompt, &descriptor, index, &arrangement);
            let (motif_directive, variation_axes, cadence_hint) =
                directives_for_role(&template.role);

            let ratio = template.base_bars as f32 / (weight.max(1) as f32);
            let mut target_seconds = (seconds_total * ratio).max(SHORT_MIN_SECTION_SECONDS);
            let mut bar_count = if seconds_per_bar > 0.0 {
                (target_seconds / seconds_per_bar).round() as i32
            } else {
                template.base_bars as i32
            };
            if bar_count < template.min_bars as i32 {
                bar_count = template.min_bars as i32;
            }
            if bar_count > template.max_bars as i32 {
                bar_count = template.max_bars as i32;
            }
            if bar_count < 1 {
                bar_count = 1;
            }

            target_seconds = (bar_count as f32 * seconds_per_bar).max(SHORT_MIN_SECTION_SECONDS);
            total_bars += bar_count as u16;
            total_duration += target_seconds;

            sections.push(CompositionSection {
                section_id: format!("s{:02}", index),
                role: template.role.clone(),
                label: template.label.to_string(),
                prompt: section_prompt,
                bars: bar_count as u8,
                target_seconds,
                energy: template.energy.clone(),
                model_id: None,
                seed_offset: Some(index as i32),
                transition: template.transition.map(|text| text.to_string()),
                motif_directive,
                variation_axes,
                cadence_hint,
                orchestration: orchestrations[index].clone(),
            });
        }

        CompositionPlan {
            version: PLAN_VERSION.to_string(),
            tempo_bpm,
            time_signature: DEFAULT_TIME_SIGNATURE.to_string(),
            key,
            total_bars,
            total_duration_seconds: total_duration,
            theme: Some(descriptor.clone()),
            sections,
        }
    }
}

fn select_key(seed: u64) -> String {
    const KEYS: [&str; 12] = [
        "C major", "G major", "D major", "A major", "E major", "B major", "F major", "E minor",
        "A minor", "D minor", "G minor", "C minor",
    ];
    let idx = (seed as usize) % KEYS.len();
    KEYS[idx].to_string()
}

fn deterministic_seed(prompt: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in prompt.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn select_tempo(raw_tempo: u16) -> u16 {
    let clamped = raw_tempo.clamp(MIN_TEMPO, MAX_TEMPO);
    let mut best = clamped;
    let mut best_error = raw_tempo.abs_diff(clamped);
    for delta in 1..16 {
        for candidate in [clamped.saturating_sub(delta), clamped.saturating_add(delta)] {
            if candidate < MIN_TEMPO || candidate > MAX_TEMPO {
                continue;
            }
            let error = raw_tempo.abs_diff(candidate);
            if error < best_error {
                best = candidate;
                best_error = error;
            }
        }
    }
    best
}

fn tempo_hint(seconds_total: f32, templates: &[SectionTemplate], beats_per_bar: u16) -> u16 {
    if seconds_total <= 0.0 {
        return 96;
    }
    let base_bars: u32 = templates.iter().map(|tpl| tpl.base_bars as u32).sum();
    let base_bars = if base_bars == 0 { (templates.len().max(1) as u32) * 8 } else { base_bars };
    let beats = base_bars * beats_per_bar as u32;
    let raw = (60.0 * beats as f32 / seconds_total).round() as u16;
    raw.max(MIN_TEMPO)
}

fn beats_per_bar(signature: &str) -> u16 {
    signature.split('/').next().and_then(|value| value.parse::<u16>().ok()).unwrap_or(4).max(1)
}

fn allocate_bars(
    templates: &[SectionTemplate],
    seconds_total: f32,
    seconds_per_bar: f32,
) -> Vec<u16> {
    if templates.is_empty() {
        return Vec::new();
    }
    let seconds_per_bar = seconds_per_bar.max(1.0);
    let target_float = seconds_total / seconds_per_bar;
    let min_total: u16 = templates.iter().map(|tpl| tpl.min_bars).sum();
    let target = target_float.round() as i32;
    let target = std::cmp::max(target, min_total as i32) as u16;
    let base_total: u16 = templates.iter().map(|tpl| tpl.base_bars).sum();
    let scale = if base_total == 0 { 1.0 } else { target as f32 / base_total as f32 };

    let mut bars: Vec<u16> = templates
        .iter()
        .map(|tpl| {
            let scaled = (tpl.base_bars as f32 * scale).round() as i32;
            let bounded = scaled.clamp(tpl.min_bars as i32, tpl.max_bars as i32).max(1);
            bounded as u16
        })
        .collect();

    rebalance_bars(templates, &mut bars, target);
    bars
}

fn rebalance_bars(templates: &[SectionTemplate], bars: &mut [u16], target_bars: u16) {
    let mut total: i32 = bars.iter().map(|value| *value as i32).sum();
    let target = target_bars as i32;
    let add_order = expand_priority(templates, ADD_PRIORITY);
    let remove_order = expand_priority(templates, REMOVE_PRIORITY);

    let mut safety = 0;
    while total < target && safety < 4096 {
        let mut adjusted = false;
        for idx in &add_order {
            let template = &templates[*idx];
            if bars[*idx] < template.max_bars {
                bars[*idx] += 1;
                total += 1;
                adjusted = true;
                if total >= target {
                    break;
                }
            }
        }
        if !adjusted {
            break;
        }
        safety += 1;
    }

    safety = 0;
    while total > target && safety < 4096 {
        let mut adjusted = false;
        for idx in &remove_order {
            let template = &templates[*idx];
            if bars[*idx] > template.min_bars {
                bars[*idx] -= 1;
                total -= 1;
                adjusted = true;
                if total <= target {
                    break;
                }
            }
        }
        if !adjusted {
            break;
        }
        safety += 1;
    }
}

fn expand_priority(templates: &[SectionTemplate], priority: &[SectionRole]) -> Vec<usize> {
    let mut order = Vec::new();
    for role in priority {
        for (index, template) in templates.iter().enumerate() {
            if &template.role == role && !order.contains(&index) {
                order.push(index);
            }
        }
    }
    for index in 0..templates.len() {
        if !order.contains(&index) {
            order.push(index);
        }
    }
    order
}

fn layer_counts_for_role(role: &SectionRole) -> LayerCounts {
    match role {
        SectionRole::Intro => LayerCounts::new(1, 1, 3, 1, 2, 0),
        SectionRole::Motif => LayerCounts::new(2, 1, 2, 1, 1, 0),
        SectionRole::Chorus => LayerCounts::new(2, 1, 2, 2, 1, 1),
        SectionRole::Bridge => LayerCounts::new(1, 1, 2, 1, 2, 1),
        SectionRole::Development => LayerCounts::new(2, 1, 2, 2, 2, 1),
        SectionRole::Resolution => LayerCounts::new(1, 1, 2, 1, 2, 1),
        SectionRole::Outro => LayerCounts::new(1, 1, 2, 1, 2, 0),
    }
}

fn categorise_instrumentation(
    descriptor: &ThemeDescriptor,
    prompt: &str,
    profile: Option<&GenreProfile>,
    _seed: u64,
) -> InstrumentPalette {
    let mut palette = InstrumentPalette::default();
    for label in &descriptor.instrumentation {
        let lower = label.to_lowercase();
        let mut matched = false;
        for category in ["rhythm", "bass", "harmony", "lead", "textures", "vocals"] {
            if category_keywords(category).iter().any(|keyword| lower.contains(keyword)) {
                push_category(&mut palette, category, label.clone());
                matched = true;
            }
        }
        if !matched {
            palette.textures.push(label.clone());
        }
    }

    let prompt_lower = prompt.to_lowercase();
    if category_keywords("vocals").iter().any(|keyword| prompt_lower.contains(keyword)) {
        palette.vocals.push("expressive vocals".to_string());
    }

    if let Some(profile) = profile {
        apply_genre_layers(&mut palette, profile);
    }

    palette.rhythm =
        if palette.rhythm.is_empty() { fallback_layer("rhythm") } else { dedupe(palette.rhythm) };
    palette.bass =
        if palette.bass.is_empty() { fallback_layer("bass") } else { dedupe(palette.bass) };
    palette.harmony = if palette.harmony.is_empty() {
        fallback_layer("harmony")
    } else {
        dedupe(palette.harmony)
    };
    palette.lead =
        if palette.lead.is_empty() { fallback_layer("lead") } else { dedupe(palette.lead) };
    palette.textures = if palette.textures.is_empty() {
        fallback_layer("textures")
    } else {
        dedupe(palette.textures)
    };
    palette.vocals = dedupe(palette.vocals);

    palette
}

fn apply_genre_layers(palette: &mut InstrumentPalette, profile: &GenreProfile) {
    palette
        .rhythm
        .extend(profile.layers.rhythm.iter().map(|value| (*value).to_string()));
    palette
        .bass
        .extend(profile.layers.bass.iter().map(|value| (*value).to_string()));
    palette
        .harmony
        .extend(profile.layers.harmony.iter().map(|value| (*value).to_string()));
    palette
        .lead
        .extend(profile.layers.lead.iter().map(|value| (*value).to_string()));
    palette
        .textures
        .extend(profile.layers.textures.iter().map(|value| (*value).to_string()));
    palette
        .vocals
        .extend(profile.layers.vocals.iter().map(|value| (*value).to_string()));
}

fn palette_offsets(_palette: &InstrumentPalette, seed: u64) -> PaletteOffsets {
    PaletteOffsets {
        rhythm: stable_offset("palette:rhythm", seed),
        bass: stable_offset("palette:bass", seed),
        harmony: stable_offset("palette:harmony", seed),
        lead: stable_offset("palette:lead", seed),
        textures: stable_offset("palette:textures", seed),
        vocals: stable_offset("palette:vocals", seed),
    }
}

fn push_category(palette: &mut InstrumentPalette, category: &str, value: String) {
    match category {
        "rhythm" => palette.rhythm.push(value),
        "bass" => palette.bass.push(value),
        "harmony" => palette.harmony.push(value),
        "lead" => palette.lead.push(value),
        "textures" => palette.textures.push(value),
        "vocals" => palette.vocals.push(value),
        _ => {}
    }
}

fn plan_orchestrations(
    templates: &[SectionTemplate],
    palette: &InstrumentPalette,
    offsets: &PaletteOffsets,
) -> Vec<SectionOrchestration> {
    templates
        .iter()
        .enumerate()
        .map(|(index, template)| {
            let counts = layer_counts_for_role(&template.role);
            build_orchestration(counts, palette, offsets, index)
        })
        .collect()
}

fn build_orchestration(
    counts: LayerCounts,
    palette: &InstrumentPalette,
    offsets: &PaletteOffsets,
    section_index: usize,
) -> SectionOrchestration {
    let mut orchestration = SectionOrchestration::default();
    orchestration.rhythm = select_layer(
        "rhythm",
        &palette.rhythm,
        counts.rhythm,
        offsets.rhythm,
        section_index,
    );
    orchestration.bass = select_layer(
        "bass",
        &palette.bass,
        counts.bass,
        offsets.bass,
        section_index,
    );
    orchestration.harmony = select_layer(
        "harmony",
        &palette.harmony,
        counts.harmony,
        offsets.harmony,
        section_index,
    );
    orchestration.lead = select_layer(
        "lead",
        &palette.lead,
        counts.lead,
        offsets.lead,
        section_index,
    );
    orchestration.textures = select_layer(
        "textures",
        &palette.textures,
        counts.textures,
        offsets.textures,
        section_index,
    );
    orchestration.vocals = if palette.vocals.is_empty() {
        Vec::new()
    } else {
        select_layer(
            "vocals",
            &palette.vocals,
            counts.vocals,
            offsets.vocals,
            section_index,
        )
    };
    orchestration
}

fn select_layer(
    category: &str,
    source: &[String],
    count: u8,
    base_offset: usize,
    section_index: usize,
) -> Vec<String> {
    if count == 0 {
        return Vec::new();
    }
    let offset = base_offset + section_index;
    if !source.is_empty() {
        return cycle_slice(source, count as usize, offset);
    }
    if category == "vocals" {
        return Vec::new();
    }
    let fallback = fallback_layer(category);
    if fallback.is_empty() {
        Vec::new()
    } else {
        cycle_slice(&fallback, count as usize, offset)
    }
}

fn fallback_layer(category: &str) -> Vec<String> {
    default_layer_fallback(category).iter().map(|value| (*value).to_string()).collect()
}

fn cycle_slice(source: &[String], count: usize, offset: usize) -> Vec<String> {
    if source.is_empty() || count == 0 {
        return Vec::new();
    }
    (0..count).map(|index| source[(offset + index) % source.len()].clone()).collect()
}

fn cycle_slice_str(source: &[&str], count: usize, offset: usize) -> Vec<String> {
    if source.is_empty() || count == 0 {
        return Vec::new();
    }
    (0..count)
        .map(|index| source[(offset + index) % source.len()].to_string())
        .collect()
}

fn fallback_instrumentation(seed: u64, count: usize) -> Vec<String> {
    if DEFAULT_INSTRUMENTATION.is_empty() {
        return vec!["blended instrumentation".to_string()];
    }
    let count = count.max(1).min(DEFAULT_INSTRUMENTATION.len());
    let offset = stable_offset("instrumentation", seed);
    cycle_slice_str(DEFAULT_INSTRUMENTATION, count, offset)
}

fn stable_offset(label: &str, seed: u64) -> usize {
    let token = format!("{}:{}", label, seed);
    let mut hash: u32 = 0x811C9DC5;
    for byte in token.bytes() {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash as usize
}

fn describe_orchestration(orchestration: &SectionOrchestration) -> String {
    let mut parts = Vec::new();
    if !orchestration.rhythm.is_empty() {
        parts.push(orchestration.rhythm.join(", "));
    }
    if !orchestration.bass.is_empty() {
        parts.push(orchestration.bass.join(", "));
    }
    if !orchestration.harmony.is_empty() {
        parts.push(orchestration.harmony.join(", "));
    }
    if !orchestration.lead.is_empty() {
        parts.push(orchestration.lead.join(", "));
    }
    if !orchestration.textures.is_empty() {
        parts.push(orchestration.textures.join(", "));
    }
    if !orchestration.vocals.is_empty() {
        parts.push(orchestration.vocals.join(", "));
    }
    parts.join(", ")
}

fn category_keywords(category: &str) -> &'static [&'static str] {
    match category {
        "rhythm" => &[
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
        ],
        "bass" => &["bass", "sub", "808", "low end", "low-end"],
        "harmony" => &[
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
        ],
        "lead" => &[
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
        ],
        "textures" => &[
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
        ],
        "vocals" => &["vocal", "voice", "singer", "choir", "chant", "lyric"],
        _ => &[],
    }
}

fn default_layer_fallback(category: &str) -> &'static [&'static str] {
    match category {
        "rhythm" => &[
            "tight drums",
            "organic percussion",
            "punchy live kit",
            "syncopated hand percussion",
            "four-on-the-floor kick",
            "shuffling brush kit",
            "driving tom groove",
        ],
        "bass" => &[
            "pulsing bass",
            "sub bass swell",
            "gritty electric bass",
            "round synth bass",
            "warm upright bass",
        ],
        "harmony" => &[
            "lush keys",
            "stacked synth pads",
            "shimmering guitar chords",
            "wide string beds",
            "layered plucked arps",
        ],
        "lead" => &[
            "expressive guitar lead",
            "soulful brass line",
            "soaring synth lead",
            "lyrical woodwind melody",
            "sparkling mallet motif",
        ],
        "textures" => &[
            "airy ambient swells",
            "granular noise beds",
            "glassy atmosphere",
            "crowd shimmer",
            "analog tape haze",
            "rolling field recordings",
        ],
        "vocals" => &[
            "wordless vocal pads",
            "ethereal choirs",
            "layered vocal oohs",
        ],
        _ => &[],
    }
}

fn dedupe(items: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut result = Vec::new();
    for item in items {
        if seen.insert(item.clone()) {
            result.push(item);
        }
    }
    result
}

fn build_theme_descriptor(
    prompt: &str,
    templates: &[SectionTemplate],
    profile: Option<&GenreProfile>,
    base_seed: u64,
) -> ThemeDescriptor {
    let prompt_lower = prompt.to_lowercase();
    let mut instrumentation = extract_keywords(&prompt_lower, INSTRUMENT_KEYWORDS);
    if let Some(profile) = profile {
        for item in profile.instrumentation {
            if !instrumentation.iter().any(|existing| existing == item) {
                instrumentation.push((*item).to_string());
            }
        }
    }
    if instrumentation.is_empty() {
        instrumentation = fallback_instrumentation(base_seed, 3);
    }

    let rhythm = derive_rhythm(&prompt_lower, templates, profile);
    let motif = derive_motif(prompt);
    let texture = derive_texture(&prompt_lower, &instrumentation, profile);
    let dynamic_curve = templates
        .iter()
        .enumerate()
        .map(|(index, template)| {
            dynamic_label(&template.role, &template.energy, index, templates.len())
        })
        .collect();

    ThemeDescriptor { motif, instrumentation, rhythm, dynamic_curve, texture: Some(texture) }
}

fn extract_keywords(prompt_lower: &str, mapping: &[(&str, &str)]) -> Vec<String> {
    let mut results = Vec::new();
    for (keyword, label) in mapping {
        if prompt_lower.contains(keyword) && !results.iter().any(|existing| existing == label) {
            results.push((*label).to_string());
        }
    }
    results
}

fn derive_rhythm(
    prompt_lower: &str,
    templates: &[SectionTemplate],
    profile: Option<&GenreProfile>,
) -> String {
    for (keyword, label) in RHYTHM_KEYWORDS {
        if prompt_lower.contains(keyword) {
            return (*label).to_string();
        }
    }
    if let Some(profile) = profile {
        if let Some(rhythm) = profile.rhythm {
            return rhythm.to_string();
        }
    }
    if templates.iter().any(|template| matches!(template.energy, SectionEnergy::High)) {
        return "driving pulse".to_string();
    }
    if templates.iter().any(|template| matches!(template.energy, SectionEnergy::Medium)) {
        return "steady groove".to_string();
    }
    "gentle pulse".to_string()
}

fn derive_motif(prompt: &str) -> String {
    let mut words: Vec<String> = prompt
        .split_whitespace()
        .map(|token| token.trim_matches(|c: char| ",.;:!?\"'".contains(c)))
        .filter(|token| !token.is_empty())
        .map(|token| token.to_string())
        .collect();
    words.retain(|word| word.chars().all(|c| c.is_alphabetic()));
    if words.len() >= 3 {
        return words[..3].join(" ");
    }
    if !words.is_empty() {
        return words.join(" ");
    }
    "primary motif".to_string()
}

fn derive_texture(
    prompt_lower: &str,
    instrumentation: &[String],
    profile: Option<&GenreProfile>,
) -> String {
    for (keyword, label) in TEXTURE_KEYWORDS {
        if prompt_lower.contains(keyword) {
            return (*label).to_string();
        }
    }
    if let Some(profile) = profile {
        if let Some(texture) = profile.texture {
            return texture.to_string();
        }
    }
    if !instrumentation.is_empty() {
        let joined = instrumentation.iter().take(2).cloned().collect::<Vec<_>>().join(", ");
        return format!("focused blend of {}", joined);
    }
    DEFAULT_TEXTURE.to_string()
}

fn dynamic_label(role: &SectionRole, energy: &SectionEnergy, index: usize, total: usize) -> String {
    let (primary, release) = match energy {
        SectionEnergy::Low => ("gentle entrance", "soft release"),
        SectionEnergy::Medium => ("steady motion", "measured resolve"),
        SectionEnergy::High => ("climactic surge", "energetic peak"),
    };
    if index == 0 {
        if matches!(role, SectionRole::Intro) {
            return primary.to_string();
        }
        return "emerging momentum".to_string();
    }
    if index + 1 == total {
        if matches!(role, SectionRole::Outro) {
            return release.to_string();
        }
        return "resolved cadence".to_string();
    }
    if matches!(role, SectionRole::Chorus) {
        return "anthemic peak".to_string();
    }
    if matches!(role, SectionRole::Bridge) {
        return "suspended tension".to_string();
    }
    match energy {
        SectionEnergy::High => "heightened intensity".to_string(),
        SectionEnergy::Medium => "building drive".to_string(),
        SectionEnergy::Low => "textural breath".to_string(),
    }
}

fn render_prompt(
    template: &str,
    prompt: &str,
    descriptor: &ThemeDescriptor,
    index: usize,
    arrangement: &str,
) -> String {
    let instrumentation_text = if descriptor.instrumentation.is_empty() {
        DEFAULT_INSTRUMENTATION.join(", ")
    } else {
        descriptor.instrumentation.join(", ")
    };
    let dynamic = descriptor
        .dynamic_curve
        .get(index)
        .cloned()
        .unwrap_or_else(|| "flowing dynamic".to_string());
    let texture = descriptor.texture.clone().unwrap_or_else(|| DEFAULT_TEXTURE.to_string());
    let arrangement_text =
        if arrangement.is_empty() { instrumentation_text.clone() } else { arrangement.to_string() };

    let mut rendered = template.to_string();
    rendered = rendered.replace("{prompt}", prompt);
    rendered = rendered.replace("{motif}", descriptor.motif.as_str());
    rendered = rendered.replace("{instrumentation}", instrumentation_text.as_str());
    rendered = rendered.replace("{rhythm}", descriptor.rhythm.as_str());
    rendered = rendered.replace("{texture}", texture.as_str());
    rendered = rendered.replace("{dynamic}", dynamic.as_str());
    rendered = rendered.replace("{arrangement}", arrangement_text.as_str());
    rendered.trim().to_string()
}

fn select_long_templates(duration_seconds: f32) -> Vec<SectionTemplate> {
    if duration_seconds >= 150.0 {
        return vec![
            SectionTemplate {
                role: SectionRole::Intro,
                label: "Intro",
                energy: SectionEnergy::Low,
                base_bars: 16,
                min_bars: 10,
                max_bars: 18,
                prompt_template: "Set the stage with {arrangement}, foreshadowing the {motif} motif over a {rhythm} pulse and {texture} atmosphere.",
                transition: Some("Invite motif"),
            },
            SectionTemplate {
                role: SectionRole::Motif,
                label: "Motif",
                energy: SectionEnergy::Medium,
                base_bars: 20,
                min_bars: 16,
                max_bars: 24,
                prompt_template: "State the {motif} motif in full, letting {arrangement} lock into the {rhythm} while {dynamic} blooms.",
                transition: Some("Build anticipation"),
            },
            SectionTemplate {
                role: SectionRole::Bridge,
                label: "Bridge",
                energy: SectionEnergy::Medium,
                base_bars: 12,
                min_bars: 8,
                max_bars: 16,
                prompt_template: "Recast the {motif} motif by thinning the layers so {arrangement} can explore contrasting colours before the chorus returns.",
                transition: Some("Spark chorus"),
            },
            SectionTemplate {
                role: SectionRole::Chorus,
                label: "Chorus",
                energy: SectionEnergy::High,
                base_bars: 24,
                min_bars: 20,
                max_bars: 28,
                prompt_template: "Lift the {motif} motif into an anthemic chorus where {arrangement} drives the groove and {dynamic} peaks.",
                transition: Some("Glide to outro"),
            },
            SectionTemplate {
                role: SectionRole::Outro,
                label: "Outro",
                energy: SectionEnergy::Medium,
                base_bars: 14,
                min_bars: 8,
                max_bars: 18,
                prompt_template: "Close by reshaping the {motif} motif, letting {arrangement} ease the {rhythm} into a reflective {texture} fade.",
                transition: Some("Fade to silence"),
            },
        ];
    }

    vec![
        SectionTemplate {
            role: SectionRole::Intro,
            label: "Intro",
            energy: SectionEnergy::Low,
            base_bars: 12,
            min_bars: 8,
            max_bars: 16,
            prompt_template: "Establish the world with {arrangement}, hinting at the {motif} motif over a {rhythm} pulse and {texture} backdrop.",
            transition: Some("Reveal motif"),
        },
        SectionTemplate {
            role: SectionRole::Motif,
            label: "Motif",
            energy: SectionEnergy::Medium,
            base_bars: 18,
            min_bars: 14,
            max_bars: 22,
            prompt_template: "Present the {motif} motif clearly, allowing {arrangement} to weave through the {rhythm} as {dynamic} intensifies.",
            transition: Some("Ignite chorus"),
        },
        SectionTemplate {
            role: SectionRole::Chorus,
            label: "Chorus",
            energy: SectionEnergy::High,
            base_bars: 24,
            min_bars: 18,
            max_bars: 26,
            prompt_template: "Amplify the {motif} motif into its fiercest form, with {arrangement} pushing the {rhythm} to a triumphant crest.",
            transition: Some("Settle to outro"),
        },
        SectionTemplate {
            role: SectionRole::Outro,
            label: "Outro",
            energy: SectionEnergy::Medium,
            base_bars: 12,
            min_bars: 8,
            max_bars: 16,
            prompt_template: "Offer a final reflection on the {motif} motif, as {arrangement} dissolves the {rhythm} into {texture}.",
            transition: Some("Fade to silence"),
        },
    ]
}

fn select_short_templates(duration_seconds: f32) -> Vec<SectionTemplate> {
    if duration_seconds >= 24.0 {
        return vec![
            SectionTemplate {
                role: SectionRole::Intro,
                label: "Arrival",
                energy: SectionEnergy::Low,
                base_bars: 4,
                min_bars: 3,
                max_bars: 6,
                prompt_template: "Set a {texture} scene for {prompt} by introducing the {motif} motif with {instrumentation} over a {rhythm} pulse.",
                transition: Some("Fade in layers"),
            },
            SectionTemplate {
                role: SectionRole::Motif,
                label: "Statement",
                energy: SectionEnergy::Medium,
                base_bars: 8,
                min_bars: 6,
                max_bars: 10,
                prompt_template: "Deliver the core {motif} motif through {instrumentation}, keeping the {rhythm} driving as {dynamic} begins.",
                transition: Some("Build momentum"),
            },
            SectionTemplate {
                role: SectionRole::Development,
                label: "Development",
                energy: SectionEnergy::High,
                base_bars: 8,
                min_bars: 6,
                max_bars: 10,
                prompt_template: "Evolve the {motif} motif with adventurous variations, letting {instrumentation} weave syncopations over the {rhythm} while {dynamic} unfolds.",
                transition: Some("Evolve harmonies"),
            },
            SectionTemplate {
                role: SectionRole::Resolution,
                label: "Resolution",
                energy: SectionEnergy::Medium,
                base_bars: 4,
                min_bars: 3,
                max_bars: 6,
                prompt_template: "Guide the {motif} motif toward resolution, using {instrumentation} to ease the {rhythm} while highlighting {dynamic}.",
                transition: Some("Return home"),
            },
            SectionTemplate {
                role: SectionRole::Outro,
                label: "Release",
                energy: SectionEnergy::Low,
                base_bars: 4,
                min_bars: 2,
                max_bars: 6,
                prompt_template: "Let the {motif} motif dissolve into ambience as {instrumentation} softens atop the {rhythm}, allowing {dynamic} to close the journey.",
                transition: Some("Fade to silence"),
            },
        ];
    }

    if duration_seconds >= 16.0 {
        return vec![
            SectionTemplate {
                role: SectionRole::Intro,
                label: "Lead-in",
                energy: SectionEnergy::Low,
                base_bars: 4,
                min_bars: 2,
                max_bars: 6,
                prompt_template: "Open gently with {instrumentation}, introducing the {motif} motif against a {rhythm} pulse that hints at {prompt}.",
                transition: Some("Invite motif"),
            },
            SectionTemplate {
                role: SectionRole::Motif,
                label: "Motif A",
                energy: SectionEnergy::Medium,
                base_bars: 8,
                min_bars: 6,
                max_bars: 10,
                prompt_template: "Present the {motif} motif clearly, keeping {instrumentation} tight around the {rhythm} while {dynamic} grows.",
                transition: Some("Increase energy"),
            },
            SectionTemplate {
                role: SectionRole::Development,
                label: "Variation",
                energy: SectionEnergy::High,
                base_bars: 6,
                min_bars: 4,
                max_bars: 8,
                prompt_template: "Develop the {motif} motif with rhythmic twists, letting {instrumentation} ride the {rhythm} as {dynamic} intensifies.",
                transition: Some("Soften textures"),
            },
            SectionTemplate {
                role: SectionRole::Resolution,
                label: "Cadence",
                energy: SectionEnergy::Medium,
                base_bars: 4,
                min_bars: 2,
                max_bars: 6,
                prompt_template: "Ease the energy back, guiding {instrumentation} to resolve the {motif} motif and settle the {rhythm} with {dynamic}.",
                transition: Some("Release"),
            },
            SectionTemplate {
                role: SectionRole::Outro,
                label: "Tail",
                energy: SectionEnergy::Low,
                base_bars: 2,
                min_bars: 1,
                max_bars: 4,
                prompt_template: "Conclude with a gentle echo of the {motif} motif, letting {instrumentation} and the {rhythm} fade as {dynamic} sighs out.",
                transition: Some("Fade"),
            },
        ];
    }

    vec![
        SectionTemplate {
            role: SectionRole::Intro,
            label: "Intro",
            energy: SectionEnergy::Low,
            base_bars: 2,
            min_bars: 1,
            max_bars: 4,
            prompt_template: "Set a delicate entrance, introducing the {motif} motif with {instrumentation} over a {rhythm} that nods to {prompt}.",
            transition: Some("Introduce motif"),
        },
        SectionTemplate {
            role: SectionRole::Motif,
            label: "Motif",
            energy: SectionEnergy::Medium,
            base_bars: 6,
            min_bars: 4,
            max_bars: 8,
            prompt_template: "Deliver the {motif} motif in full, keeping {instrumentation} aligned with the {rhythm} while {dynamic} expands.",
            transition: Some("Lift energy"),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_plan_with_at_least_three_sections() {
        let planner = CompositionPlanner::new();
        let requested = 16;
        let plan = planner.build_plan("dreamy piano", requested, Some(42));
        assert!(plan.sections.len() >= 3);
        assert_eq!(plan.version, PLAN_VERSION);
        assert!(plan.total_duration_seconds > 0.0);
        assert!(plan.total_duration_seconds + 1.0 >= requested as f32);
        assert!(plan.theme.is_some());
        assert!(plan.sections.iter().all(|section| section.motif_directive.is_some()));
        assert!(plan
            .sections
            .iter()
            .any(|section| section.motif_directive.as_deref() == Some("state motif")));
    }

    #[test]
    fn collapses_short_duration_into_minimum_sections() {
        let planner = CompositionPlanner::new();
        let plan = planner.build_plan("short clip", 8, None);
        assert!(plan.sections.len() <= 2);
        assert!(plan.sections.iter().any(|section| matches!(section.role, SectionRole::Motif)));
        assert!(plan.sections.iter().all(|section| section.target_seconds >= 2.0));
        assert!(plan.theme.is_some());
        assert!(plan.sections.iter().all(|section| section.motif_directive.is_some()));
    }
}
