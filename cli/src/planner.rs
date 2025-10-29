use crate::types::{
    CompositionPlan, CompositionSection, GenerationMode, SectionEnergy, SectionOrchestration,
    SectionRole, ThemeDescriptor,
};
use once_cell::sync::Lazy;
use serde::Deserialize;
use std::collections::HashSet;
use unicode_normalization::UnicodeNormalization;

const PLAN_VERSION: &str = "v3";
const MIN_TEMPO: u16 = 68;
const MAX_TEMPO: u16 = 128;
const DEFAULT_TIME_SIGNATURE: &str = "4/4";
const SHORT_MIN_TOTAL_SECONDS: f32 = 2.0;
const SHORT_MIN_SECTION_SECONDS: f32 = 2.0;
const LONG_FORM_THRESHOLD: f32 = 90.0;
const LONG_MIN_SECTION_SECONDS: f32 = 16.0;
const MOTIF_MAX_TOTAL_SECONDS: f32 = 24.0;

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

static LEXICON: Lazy<Lexicon> = Lazy::new(|| {
    let raw: RawLexicon =
        serde_json::from_str(include_str!("../../planner/lexicon.json")).expect("invalid lexicon");
    Lexicon::from_raw(raw)
});

fn lexicon() -> &'static Lexicon {
    &LEXICON
}

fn leak_str(value: String) -> &'static str {
    Box::leak(value.into_boxed_str())
}

fn casefold(value: &str) -> String {
    value.nfkc().flat_map(|c| c.to_lowercase()).collect()
}

#[derive(Debug, Deserialize)]
struct RawLexicon {
    version: u32,
    keywords: RawKeywordGroups,
    defaults: RawDefaults,
    #[serde(rename = "genre_profiles")]
    genre_profiles: Vec<RawGenreProfile>,
    templates: RawTemplateLibrary,
}

#[derive(Debug, Deserialize)]
struct RawKeywordGroups {
    #[serde(rename = "instrument")]
    instrument: Vec<RawKeyword>,
    #[serde(rename = "rhythm")]
    rhythm: Vec<RawKeyword>,
    #[serde(rename = "texture")]
    texture: Vec<RawKeyword>,
}

#[derive(Debug, Deserialize)]
struct RawKeyword {
    term: String,
    descriptor: String,
}

#[derive(Debug, Deserialize)]
struct RawDefaults {
    instrumentation: Vec<String>,
    texture: String,
}

#[derive(Debug, Deserialize)]
struct RawGenreProfile {
    keywords: Vec<String>,
    instrumentation: Vec<String>,
    rhythm: Option<String>,
    texture: Option<String>,
    layers: RawGenreLayers,
}

#[derive(Debug, Deserialize)]
struct RawGenreLayers {
    rhythm: Vec<String>,
    bass: Vec<String>,
    harmony: Vec<String>,
    lead: Vec<String>,
    textures: Vec<String>,
    vocals: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RawTemplateLibrary {
    long: Vec<RawTemplateVariant>,
    short: Vec<RawTemplateVariant>,
}

#[derive(Debug, Deserialize)]
struct RawTemplateVariant {
    min_duration: f32,
    sections: Vec<RawSectionTemplate>,
}

#[derive(Debug, Deserialize)]
struct RawSectionTemplate {
    role: RawSectionRole,
    label: String,
    energy: RawSectionEnergy,
    base_bars: u16,
    min_bars: u16,
    max_bars: u16,
    prompt_template: String,
    transition: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum RawSectionRole {
    Intro,
    Motif,
    Chorus,
    Development,
    Bridge,
    Resolution,
    Outro,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum RawSectionEnergy {
    Low,
    Medium,
    High,
}

#[derive(Debug)]
struct Lexicon {
    _version: u32,
    instrument_keywords: Vec<KeywordMapping>,
    rhythm_keywords: Vec<KeywordMapping>,
    texture_keywords: Vec<KeywordMapping>,
    defaults: Defaults,
    genre_profiles: Vec<GenreProfile>,
    templates: TemplateLibrary,
}

#[derive(Debug)]
struct KeywordMapping {
    _term: &'static str,
    descriptor: &'static str,
    folded_term: &'static str,
}

#[derive(Debug)]
struct Defaults {
    instrumentation: Vec<&'static str>,
    texture: &'static str,
}

#[derive(Debug)]
struct GenreProfile {
    _keywords: Vec<&'static str>,
    folded_keywords: Vec<&'static str>,
    instrumentation: Vec<&'static str>,
    rhythm: Option<&'static str>,
    texture: Option<&'static str>,
    layers: GenreLayers,
}

#[derive(Debug)]
struct GenreLayers {
    rhythm: Vec<&'static str>,
    bass: Vec<&'static str>,
    harmony: Vec<&'static str>,
    lead: Vec<&'static str>,
    textures: Vec<&'static str>,
    vocals: Vec<&'static str>,
}

#[derive(Debug)]
struct TemplateLibrary {
    long: Vec<TemplateVariant>,
    short: Vec<TemplateVariant>,
}

#[derive(Debug)]
struct TemplateVariant {
    min_duration: f32,
    sections: Vec<SectionTemplateBlueprint>,
}

#[derive(Debug)]
struct SectionTemplateBlueprint {
    role: SectionRole,
    label: &'static str,
    energy: SectionEnergy,
    base_bars: u16,
    min_bars: u16,
    max_bars: u16,
    prompt_template: &'static str,
    transition: Option<&'static str>,
}

impl Lexicon {
    fn from_raw(raw: RawLexicon) -> Self {
        let instrument_keywords =
            raw.keywords.instrument.into_iter().map(KeywordMapping::from_raw).collect();
        let rhythm_keywords =
            raw.keywords.rhythm.into_iter().map(KeywordMapping::from_raw).collect();
        let texture_keywords =
            raw.keywords.texture.into_iter().map(KeywordMapping::from_raw).collect();
        let defaults = Defaults::from_raw(raw.defaults);
        let genre_profiles = raw.genre_profiles.into_iter().map(GenreProfile::from_raw).collect();
        let templates = TemplateLibrary::from_raw(raw.templates);
        Self {
            _version: raw.version,
            instrument_keywords,
            rhythm_keywords,
            texture_keywords,
            defaults,
            genre_profiles,
            templates,
        }
    }
}

impl KeywordMapping {
    fn from_raw(raw: RawKeyword) -> Self {
        let term = leak_str(raw.term);
        let descriptor = leak_str(raw.descriptor);
        let folded_term = leak_str(casefold(term));
        Self { _term: term, descriptor, folded_term }
    }
}

impl Defaults {
    fn from_raw(raw: RawDefaults) -> Self {
        let instrumentation = raw.instrumentation.into_iter().map(leak_str).collect();
        let texture = leak_str(raw.texture);
        Self { instrumentation, texture }
    }
}

impl GenreProfile {
    fn from_raw(raw: RawGenreProfile) -> Self {
        let keywords: Vec<&'static str> = raw.keywords.into_iter().map(leak_str).collect();
        let folded_keywords =
            keywords.iter().map(|value| leak_str(casefold(value))).collect::<Vec<_>>();
        let instrumentation = raw.instrumentation.into_iter().map(leak_str).collect();
        let rhythm = raw.rhythm.map(leak_str);
        let texture = raw.texture.map(leak_str);
        let layers = GenreLayers::from_raw(raw.layers);
        Self { _keywords: keywords, folded_keywords, instrumentation, rhythm, texture, layers }
    }
}

impl GenreLayers {
    fn from_raw(raw: RawGenreLayers) -> Self {
        Self {
            rhythm: raw.rhythm.into_iter().map(leak_str).collect(),
            bass: raw.bass.into_iter().map(leak_str).collect(),
            harmony: raw.harmony.into_iter().map(leak_str).collect(),
            lead: raw.lead.into_iter().map(leak_str).collect(),
            textures: raw.textures.into_iter().map(leak_str).collect(),
            vocals: raw.vocals.into_iter().map(leak_str).collect(),
        }
    }
}

impl TemplateLibrary {
    fn from_raw(raw: RawTemplateLibrary) -> Self {
        let mut long = raw.long.into_iter().map(TemplateVariant::from_raw).collect::<Vec<_>>();
        long.sort_by(|a, b| {
            b.min_duration.partial_cmp(&a.min_duration).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut short = raw.short.into_iter().map(TemplateVariant::from_raw).collect::<Vec<_>>();
        short.sort_by(|a, b| {
            b.min_duration.partial_cmp(&a.min_duration).unwrap_or(std::cmp::Ordering::Equal)
        });
        Self { long, short }
    }

    fn select_long(&self, duration_seconds: f32) -> Vec<SectionTemplate> {
        self.select_variant(&self.long, duration_seconds)
    }

    fn select_short(&self, duration_seconds: f32) -> Vec<SectionTemplate> {
        self.select_variant(&self.short, duration_seconds)
    }

    fn select_variant(
        &self,
        variants: &[TemplateVariant],
        duration_seconds: f32,
    ) -> Vec<SectionTemplate> {
        let selected = variants
            .iter()
            .find(|variant| duration_seconds >= variant.min_duration)
            .or_else(|| variants.last())
            .expect("planner templates cannot be empty");
        selected.to_templates()
    }
}

impl TemplateVariant {
    fn from_raw(raw: RawTemplateVariant) -> Self {
        let sections = raw.sections.into_iter().map(SectionTemplateBlueprint::from_raw).collect();
        Self { min_duration: raw.min_duration, sections }
    }

    fn to_templates(&self) -> Vec<SectionTemplate> {
        self.sections.iter().map(SectionTemplateBlueprint::to_template).collect()
    }
}

impl SectionTemplateBlueprint {
    fn from_raw(raw: RawSectionTemplate) -> Self {
        Self {
            role: raw.role.into(),
            label: leak_str(raw.label),
            energy: raw.energy.into(),
            base_bars: raw.base_bars,
            min_bars: raw.min_bars,
            max_bars: raw.max_bars,
            prompt_template: leak_str(raw.prompt_template),
            transition: raw.transition.map(leak_str),
        }
    }

    fn to_template(&self) -> SectionTemplate {
        SectionTemplate {
            role: self.role,
            label: self.label,
            energy: self.energy,
            base_bars: self.base_bars,
            min_bars: self.min_bars,
            max_bars: self.max_bars,
            prompt_template: self.prompt_template,
            transition: self.transition,
        }
    }
}

impl From<RawSectionRole> for SectionRole {
    fn from(value: RawSectionRole) -> Self {
        match value {
            RawSectionRole::Intro => SectionRole::Intro,
            RawSectionRole::Motif => SectionRole::Motif,
            RawSectionRole::Chorus => SectionRole::Chorus,
            RawSectionRole::Development => SectionRole::Development,
            RawSectionRole::Bridge => SectionRole::Bridge,
            RawSectionRole::Resolution => SectionRole::Resolution,
            RawSectionRole::Outro => SectionRole::Outro,
        }
    }
}

impl From<RawSectionEnergy> for SectionEnergy {
    fn from(value: RawSectionEnergy) -> Self {
        match value {
            RawSectionEnergy::Low => SectionEnergy::Low,
            RawSectionEnergy::Medium => SectionEnergy::Medium,
            RawSectionEnergy::High => SectionEnergy::High,
        }
    }
}

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

fn match_genre_profile<'a>(prompt_fold: &str, data: &'a Lexicon) -> Option<&'a GenreProfile> {
    let mut best: Option<&GenreProfile> = None;
    let mut best_len = 0usize;
    for profile in &data.genre_profiles {
        for keyword in &profile.folded_keywords {
            if prompt_fold.contains(keyword) && keyword.len() > best_len {
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

#[derive(Clone)]
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
        mode: GenerationMode,
    ) -> CompositionPlan {
        match mode {
            GenerationMode::Motif => self.build_motif_plan(prompt, duration_seconds, seed),
            GenerationMode::FullTrack => {
                if (duration_seconds as f32) >= LONG_FORM_THRESHOLD {
                    return self.build_long_form_plan(prompt, duration_seconds, seed);
                }
                self.build_short_form_plan(prompt, duration_seconds, seed)
            }
        }
    }

    fn build_motif_plan(
        &self,
        prompt: &str,
        duration_seconds: u8,
        seed: Option<u64>,
    ) -> CompositionPlan {
        let data = lexicon();
        let motif_template = select_motif_template(data);
        let templates = vec![motif_template.clone()];
        let seconds_total =
            (duration_seconds as f32).clamp(SHORT_MIN_TOTAL_SECONDS, MOTIF_MAX_TOTAL_SECONDS);
        let beats_per_bar = beats_per_bar(DEFAULT_TIME_SIGNATURE);
        let total_weight = motif_template.base_bars.max(1) as u32;
        let raw_tempo = if seconds_total > 0.0 {
            (240.0 * total_weight as f32 / seconds_total).round().max(MIN_TEMPO as f32) as u16
        } else {
            90
        };
        let tempo_bpm = select_tempo(raw_tempo);
        let seconds_per_bar = (60.0 / tempo_bpm as f32) * beats_per_bar as f32;

        let base_seed = seed.unwrap_or_else(|| deterministic_seed(prompt));
        let prompt_fold = casefold(prompt.as_ref());
        let profile = match_genre_profile(&prompt_fold, data);

        let descriptor =
            build_theme_descriptor(prompt, &prompt_fold, &templates, profile, base_seed, data);
        let palette =
            categorise_instrumentation(&descriptor, &prompt_fold, profile, base_seed, data);
        let offsets = palette_offsets(&palette, base_seed);
        let orchestrations = plan_orchestrations(&templates, &palette, &offsets);
        let key = select_key(base_seed);

        let arrangement = describe_orchestration(&orchestrations[0]);
        let section_prompt = render_prompt(
            motif_template.prompt_template,
            prompt,
            &descriptor,
            0,
            &arrangement,
            &data.defaults,
        );
        let (motif_directive, variation_axes, cadence_hint) =
            directives_for_role(&motif_template.role);

        let mut bar_count = if seconds_per_bar > 0.0 {
            (seconds_total / seconds_per_bar).round() as i32
        } else {
            motif_template.base_bars as i32
        };
        if bar_count < motif_template.min_bars as i32 {
            bar_count = motif_template.min_bars as i32;
        }
        if bar_count > motif_template.max_bars as i32 {
            bar_count = motif_template.max_bars as i32;
        }
        if bar_count < 1 {
            bar_count = 1;
        }

        let target_seconds = (bar_count as f32 * seconds_per_bar).max(SHORT_MIN_SECTION_SECONDS);

        let section = CompositionSection {
            section_id: "s00".to_string(),
            role: motif_template.role.clone(),
            label: motif_template.label.to_string(),
            prompt: section_prompt,
            bars: bar_count as u8,
            target_seconds,
            energy: motif_template.energy.clone(),
            model_id: None,
            seed_offset: Some(0),
            transition: motif_template.transition.map(|text| text.to_string()),
            motif_directive,
            variation_axes,
            cadence_hint,
            orchestration: orchestrations[0].clone(),
        };

        CompositionPlan {
            version: PLAN_VERSION.to_string(),
            tempo_bpm,
            time_signature: DEFAULT_TIME_SIGNATURE.to_string(),
            key,
            total_bars: bar_count as u16,
            total_duration_seconds: target_seconds,
            theme: Some(descriptor),
            sections: vec![section],
        }
    }

    fn build_long_form_plan(
        &self,
        prompt: &str,
        duration_seconds: u8,
        seed: Option<u64>,
    ) -> CompositionPlan {
        let data = lexicon();
        let seconds_total = f32::max(duration_seconds as f32, LONG_FORM_THRESHOLD);
        let templates = select_long_templates(data, seconds_total);
        let beats_per_bar = beats_per_bar(DEFAULT_TIME_SIGNATURE);
        let tempo_hint = tempo_hint(seconds_total, &templates, beats_per_bar);
        let tempo_bpm = select_tempo(tempo_hint);
        let seconds_per_bar = (60.0 / tempo_bpm as f32) * beats_per_bar as f32;

        let bars = allocate_bars(&templates, seconds_total, seconds_per_bar);

        let base_seed = seed.unwrap_or_else(|| deterministic_seed(prompt));
        let prompt_fold = casefold(prompt.as_ref());
        let profile = match_genre_profile(&prompt_fold, data);

        let descriptor =
            build_theme_descriptor(prompt, &prompt_fold, &templates, profile, base_seed, data);
        let palette =
            categorise_instrumentation(&descriptor, &prompt_fold, profile, base_seed, data);
        let offsets = palette_offsets(&palette, base_seed);
        let orchestrations = plan_orchestrations(&templates, &palette, &offsets);
        let key = select_key(base_seed);

        let mut sections: Vec<CompositionSection> = Vec::with_capacity(templates.len());
        let mut total_bars: u16 = 0;

        for (index, template) in templates.iter().enumerate() {
            let arrangement = describe_orchestration(&orchestrations[index]);
            let section_prompt = render_prompt(
                template.prompt_template,
                prompt,
                &descriptor,
                index,
                &arrangement,
                &data.defaults,
            );
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
        let data = lexicon();
        let seconds_total = f32::max(duration_seconds as f32, SHORT_MIN_TOTAL_SECONDS);
        let templates = select_short_templates(data, seconds_total);
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
        let prompt_fold = casefold(prompt.as_ref());
        let profile = match_genre_profile(&prompt_fold, data);

        let descriptor =
            build_theme_descriptor(prompt, &prompt_fold, &templates, profile, base_seed, data);
        let palette =
            categorise_instrumentation(&descriptor, &prompt_fold, profile, base_seed, data);
        let offsets = palette_offsets(&palette, base_seed);
        let orchestrations = plan_orchestrations(&templates, &palette, &offsets);
        let key = select_key(base_seed);

        let mut sections: Vec<CompositionSection> = Vec::with_capacity(templates.len());
        let mut total_bars: u16 = 0;
        let mut total_duration = 0.0f32;

        for (index, template) in templates.iter().enumerate() {
            let arrangement = describe_orchestration(&orchestrations[index]);
            let section_prompt = render_prompt(
                template.prompt_template,
                prompt,
                &descriptor,
                index,
                &arrangement,
                &data.defaults,
            );
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
    prompt_fold: &str,
    profile: Option<&GenreProfile>,
    _seed: u64,
    _data: &Lexicon,
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

    if category_keywords("vocals").iter().any(|keyword| prompt_fold.contains(keyword)) {
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
    palette.rhythm.extend(profile.layers.rhythm.iter().map(|value| (*value).to_string()));
    palette.bass.extend(profile.layers.bass.iter().map(|value| (*value).to_string()));
    palette.harmony.extend(profile.layers.harmony.iter().map(|value| (*value).to_string()));
    palette.lead.extend(profile.layers.lead.iter().map(|value| (*value).to_string()));
    palette.textures.extend(profile.layers.textures.iter().map(|value| (*value).to_string()));
    palette.vocals.extend(profile.layers.vocals.iter().map(|value| (*value).to_string()));
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
    orchestration.rhythm =
        select_layer("rhythm", &palette.rhythm, counts.rhythm, offsets.rhythm, section_index);
    orchestration.bass =
        select_layer("bass", &palette.bass, counts.bass, offsets.bass, section_index);
    orchestration.harmony =
        select_layer("harmony", &palette.harmony, counts.harmony, offsets.harmony, section_index);
    orchestration.lead =
        select_layer("lead", &palette.lead, counts.lead, offsets.lead, section_index);
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
        select_layer("vocals", &palette.vocals, counts.vocals, offsets.vocals, section_index)
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
    (0..count).map(|index| source[(offset + index) % source.len()].to_string()).collect()
}

fn fallback_instrumentation(data: &Lexicon, seed: u64, count: usize) -> Vec<String> {
    if data.defaults.instrumentation.is_empty() {
        return vec!["blended instrumentation".to_string()];
    }
    let count = count.max(1).min(data.defaults.instrumentation.len());
    let offset = stable_offset("instrumentation", seed);
    cycle_slice_str(data.defaults.instrumentation.as_slice(), count, offset)
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
            "lead", "guitar", "solo", "brass", "sax", "horn", "trumpet", "violin", "viola",
            "fiddle", "flute", "clarinet", "oboe",
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
        "vocals" => &["wordless vocal pads", "ethereal choirs", "layered vocal oohs"],
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
    prompt_fold: &str,
    templates: &[SectionTemplate],
    profile: Option<&GenreProfile>,
    base_seed: u64,
    data: &Lexicon,
) -> ThemeDescriptor {
    let mut instrumentation = extract_keywords(prompt_fold, &data.instrument_keywords);
    if let Some(profile) = profile {
        for item in &profile.instrumentation {
            if !instrumentation.iter().any(|existing| existing == item) {
                instrumentation.push((*item).to_string());
            }
        }
    }
    if instrumentation.is_empty() {
        instrumentation = fallback_instrumentation(data, base_seed, 3);
    }

    let rhythm = derive_rhythm(prompt_fold, templates, profile, data);
    let motif = derive_motif(prompt);
    let texture = derive_texture(prompt_fold, &instrumentation, profile, data);
    let dynamic_curve = templates
        .iter()
        .enumerate()
        .map(|(index, template)| {
            dynamic_label(&template.role, &template.energy, index, templates.len())
        })
        .collect();

    ThemeDescriptor { motif, instrumentation, rhythm, dynamic_curve, texture: Some(texture) }
}

fn extract_keywords(prompt_fold: &str, mapping: &[KeywordMapping]) -> Vec<String> {
    let mut results = Vec::new();
    for entry in mapping {
        if prompt_fold.contains(entry.folded_term)
            && !results.iter().any(|existing| existing == entry.descriptor)
        {
            results.push(entry.descriptor.to_string());
        }
    }
    results
}

fn derive_rhythm(
    prompt_fold: &str,
    templates: &[SectionTemplate],
    profile: Option<&GenreProfile>,
    data: &Lexicon,
) -> String {
    for entry in &data.rhythm_keywords {
        if prompt_fold.contains(entry.folded_term) {
            return entry.descriptor.to_string();
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
    prompt_fold: &str,
    instrumentation: &[String],
    profile: Option<&GenreProfile>,
    data: &Lexicon,
) -> String {
    for entry in &data.texture_keywords {
        if prompt_fold.contains(entry.folded_term) {
            return entry.descriptor.to_string();
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
    data.defaults.texture.to_string()
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
    defaults: &Defaults,
) -> String {
    let instrumentation_text = if descriptor.instrumentation.is_empty() {
        defaults.instrumentation.join(", ")
    } else {
        descriptor.instrumentation.join(", ")
    };
    let dynamic = descriptor
        .dynamic_curve
        .get(index)
        .cloned()
        .unwrap_or_else(|| "flowing dynamic".to_string());
    let texture = descriptor.texture.clone().unwrap_or_else(|| defaults.texture.to_string());
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

fn select_long_templates(data: &Lexicon, duration_seconds: f32) -> Vec<SectionTemplate> {
    data.templates.select_long(duration_seconds)
}

fn select_short_templates(data: &Lexicon, duration_seconds: f32) -> Vec<SectionTemplate> {
    data.templates.select_short(duration_seconds)
}

fn select_motif_template(data: &Lexicon) -> SectionTemplate {
    let primary = data
        .templates
        .select_short(0.0)
        .into_iter()
        .find(|tpl| matches!(tpl.role, SectionRole::Motif));
    if let Some(template) = primary {
        return template;
    }
    let fallback = data
        .templates
        .select_short(SHORT_MIN_TOTAL_SECONDS)
        .into_iter()
        .find(|tpl| matches!(tpl.role, SectionRole::Motif));
    if let Some(template) = fallback {
        return template;
    }
    SectionTemplate {
        role: SectionRole::Motif,
        label: "Motif",
        energy: SectionEnergy::Medium,
        base_bars: 6,
        min_bars: 4,
        max_bars: 8,
        prompt_template:
            "Present the {motif} motif clearly, keeping {instrumentation} tight around the {rhythm} while {dynamic} grows.",
        transition: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_plan_with_at_least_three_sections() {
        let planner = CompositionPlanner::new();
        let requested = 16;
        let plan =
            planner.build_plan("dreamy piano", requested, Some(42), GenerationMode::FullTrack);
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
        let plan = planner.build_plan("short clip", 8, None, GenerationMode::FullTrack);
        assert!(plan.sections.len() <= 2);
        assert!(plan.sections.iter().any(|section| matches!(section.role, SectionRole::Motif)));
        assert!(plan.sections.iter().all(|section| section.target_seconds >= 2.0));
        assert!(plan.theme.is_some());
        assert!(plan.sections.iter().all(|section| section.motif_directive.is_some()));
    }

    #[test]
    fn builds_motif_only_plan() {
        let planner = CompositionPlanner::new();
        let plan = planner.build_plan("motif test", 12, Some(7), GenerationMode::Motif);
        assert_eq!(plan.sections.len(), 1);
        let section = &plan.sections[0];
        assert!(matches!(section.role, SectionRole::Motif));
        assert!(section.target_seconds >= 2.0);
        assert!(section.target_seconds <= MOTIF_MAX_TOTAL_SECONDS + 1.0);
        assert!(section.motif_directive.is_some());
        assert!(plan.theme.is_some());
    }
}
