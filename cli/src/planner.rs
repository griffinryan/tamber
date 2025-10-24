use crate::types::{
    CompositionPlan, CompositionSection, SectionEnergy, SectionRole, ThemeDescriptor,
};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const PLAN_VERSION: &str = "v2";
const MIN_TEMPO: u16 = 60;
const MAX_TEMPO: u16 = 140;
const DEFAULT_TIME_SIGNATURE: &str = "4/4";
const MIN_SECTION_SECONDS: f32 = 5.0;
const MIN_TOTAL_SECONDS: f32 = 12.0;

const ROLE_MIN_BEATS: &[(SectionRole, u16)] = &[
    (SectionRole::Intro, 4),
    (SectionRole::Motif, 8),
    (SectionRole::Development, 8),
    (SectionRole::Bridge, 4),
    (SectionRole::Resolution, 4),
    (SectionRole::Outro, 4),
];

const ADD_PRIORITY: &[SectionRole] = &[
    SectionRole::Motif,
    SectionRole::Development,
    SectionRole::Bridge,
    SectionRole::Resolution,
    SectionRole::Intro,
    SectionRole::Outro,
];

const REMOVE_PRIORITY: &[SectionRole] = &[
    SectionRole::Outro,
    SectionRole::Intro,
    SectionRole::Bridge,
    SectionRole::Resolution,
    SectionRole::Development,
    SectionRole::Motif,
];

const INSTRUMENT_KEYWORDS: &[(&str, &str)] = &[
    ("piano", "warm piano"),
    ("keys", "soft keys"),
    ("synthwave", "retro synth layers"),
    ("synth", "lush synth pads"),
    ("modular", "modular synth textures"),
    ("guitar", "ambient guitar"),
    ("bass", "deep bass"),
    ("drum", "tight drums"),
    ("percussion", "organic percussion"),
    ("string", "layered strings"),
    ("violin", "expressive strings"),
    ("cello", "warm cello"),
    ("choir", "airy choir voices"),
    ("vocal", "ethereal vocals"),
    ("brass", "smooth brass"),
    ("sax", "saxophone lead"),
    ("flute", "breathy flute"),
    ("ambient", "atmospheric textures"),
    ("lofi", "dusty keys"),
];

const RHYTHM_KEYWORDS: &[(&str, &str)] = &[
    ("waltz", "gentle 3/4 sway"),
    ("swing", "swinging groove"),
    ("hip hop", "laid-back hip hop beat"),
    ("boom bap", "boom-bap pulse"),
    ("house", "four-on-the-floor pulse"),
    ("techno", "driving techno rhythm"),
    ("trance", "rolling trance rhythm"),
    ("trap", "stuttered trap beat"),
    ("downtempo", "downtempo pulse"),
    ("breakbeat", "syncopated breakbeat"),
    ("bossa", "bossa nova sway"),
    ("reggae", "off-beat reggae groove"),
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
];

const DEFAULT_INSTRUMENTATION: &[&str] = &["blended instrumentation"];
const DEFAULT_TEXTURE: &str = "immersive atmosphere";

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
    bars: u8,
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
        let seconds_total = f32::max(duration_seconds as f32, MIN_TOTAL_SECONDS);
        let templates = prune_templates(seconds_total, select_templates(duration_seconds));
        let raw_bars: u16 = templates.iter().map(|tpl| tpl.bars as u16).sum();
        let raw_tempo = if seconds_total > 0.0 {
            (240.0 * raw_bars as f32 / seconds_total).round() as u16
        } else {
            90
        };
        let tempo_bpm = select_tempo(raw_tempo);
        let seconds_per_beat = 60.0 / tempo_bpm as f32;

        let base_seed = seed.unwrap_or_else(|| deterministic_seed(prompt));
        let key = select_key(base_seed);

        let beats = allocate_beats(&templates, seconds_total, seconds_per_beat);
        let total_bars = ((beats.iter().sum::<u32>() as f32) / 4.0).round() as u16;
        let total_bars = total_bars.max(1);
        let seconds_per_section: Vec<f32> =
            beats.iter().map(|count| *count as f32 * seconds_per_beat).collect();

        let descriptor = build_theme_descriptor(prompt, &templates);
        let mut sections: Vec<CompositionSection> = Vec::with_capacity(templates.len());

        for (index, (template, section_seconds)) in
            templates.iter().zip(seconds_per_section).enumerate()
        {
            let section_prompt =
                render_prompt(template.prompt_template, prompt, &descriptor, index);
            let target_seconds = section_seconds.max(MIN_SECTION_SECONDS);
            let (motif_directive, variation_axes, cadence_hint) =
                directives_for_role(&template.role);

            sections.push(CompositionSection {
                section_id: format!("s{:02}", index),
                role: template.role.clone(),
                label: template.label.to_string(),
                prompt: section_prompt,
                bars: template.bars,
                target_seconds,
                energy: template.energy.clone(),
                model_id: None,
                seed_offset: Some(index as i32),
                transition: template.transition.map(|text| text.to_string()),
                motif_directive,
                variation_axes,
                cadence_hint,
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
    let mut hasher = DefaultHasher::new();
    prompt.hash(&mut hasher);
    hasher.finish()
}

fn prune_templates(
    seconds_total: f32,
    mut templates: Vec<SectionTemplate>,
) -> Vec<SectionTemplate> {
    while templates.len() > 1 && seconds_total < MIN_SECTION_SECONDS * templates.len() as f32 {
        let drop_index = template_to_drop(&templates);
        templates.remove(drop_index);
    }
    templates
}

fn select_tempo(raw_tempo: u16) -> u16 {
    let clamped = raw_tempo.clamp(MIN_TEMPO, MAX_TEMPO);
    let mut best = clamped;
    let mut best_error =
        if raw_tempo > clamped { raw_tempo - clamped } else { clamped - raw_tempo };
    for delta in 1..16 {
        for candidate in [clamped.saturating_sub(delta), clamped.saturating_add(delta)] {
            if candidate < MIN_TEMPO || candidate > MAX_TEMPO {
                continue;
            }
            let error =
                if raw_tempo > candidate { raw_tempo - candidate } else { candidate - raw_tempo };
            if error < best_error {
                best = candidate;
                best_error = error;
            }
        }
    }
    best
}

fn template_to_drop(templates: &[SectionTemplate]) -> usize {
    fn role_priority(role: &SectionRole) -> u8 {
        match role {
            SectionRole::Outro => 0,
            SectionRole::Intro => 1,
            SectionRole::Bridge => 2,
            SectionRole::Resolution => 3,
            SectionRole::Development => 4,
            SectionRole::Motif => 5,
        }
    }

    let mut best_index = 0usize;
    let mut best_key = (role_priority(&templates[0].role), templates[0].bars, 0usize);

    for (index, template) in templates.iter().enumerate().skip(1) {
        let key = (role_priority(&template.role), template.bars, index);
        if key < best_key {
            best_index = index;
            best_key = key;
        }
    }

    best_index
}

fn role_min_beats(role: &SectionRole, seconds_per_beat: f32) -> u32 {
    let role_min = ROLE_MIN_BEATS
        .iter()
        .find_map(|(candidate, beats)| if candidate == role { Some(*beats as u32) } else { None })
        .unwrap_or(4);
    let min_by_seconds = if seconds_per_beat > 0.0 {
        (MIN_SECTION_SECONDS / seconds_per_beat).ceil() as u32
    } else {
        4
    };
    std::cmp::max(role_min, min_by_seconds)
}

fn allocate_beats(
    templates: &[SectionTemplate],
    seconds_total: f32,
    seconds_per_beat: f32,
) -> Vec<u32> {
    if templates.is_empty() {
        return Vec::new();
    }

    let total_beats_raw: f32 =
        templates.iter().map(|template| template.bars as f32 * 4.0).sum::<f32>().max(1.0);
    let min_beats: Vec<u32> =
        templates.iter().map(|template| role_min_beats(&template.role, seconds_per_beat)).collect();
    let target_beats = {
        let desired = (seconds_total / seconds_per_beat).round() as i32;
        let minimum: u32 = min_beats.iter().copied().sum();
        let desired = std::cmp::max(desired, 1) as u32;
        std::cmp::max(minimum, desired)
    };

    let scale = target_beats as f32 / total_beats_raw;
    let mut provisional: Vec<u32> = templates
        .iter()
        .zip(min_beats.iter())
        .map(|(template, min_count)| {
            let scaled = (template.bars as f32 * 4.0 * scale).round() as i32;
            let scaled = std::cmp::max(scaled, 1);
            std::cmp::max(*min_count as i32, scaled) as u32
        })
        .collect();

    rebalance_beats(templates, &mut provisional, &min_beats, target_beats);
    provisional
}

fn rebalance_beats(
    templates: &[SectionTemplate],
    beats: &mut [u32],
    min_beats: &[u32],
    target_beats: u32,
) {
    let mut total: i32 = beats.iter().sum::<u32>() as i32;
    let target = target_beats as i32;

    let expand_priority = |priority: &[SectionRole]| -> Vec<usize> {
        let mut order = Vec::new();
        for role in priority {
            for (index, template) in templates.iter().enumerate() {
                if &template.role == role && !order.contains(&index) {
                    order.push(index);
                }
            }
        }
        if order.len() < templates.len() {
            for index in 0..templates.len() {
                if !order.contains(&index) {
                    order.push(index);
                }
            }
        }
        order
    };

    let add_order = expand_priority(ADD_PRIORITY);
    let remove_order = expand_priority(REMOVE_PRIORITY);

    let mut safety = 0;
    while total < target && safety < 4096 {
        let mut modified = false;
        for idx in &add_order {
            beats[*idx] += 1;
            total += 1;
            modified = true;
            if total >= target {
                break;
            }
        }
        if !modified {
            break;
        }
        safety += 1;
    }

    safety = 0;
    while total > target && safety < 4096 {
        let mut modified = false;
        for idx in &remove_order {
            if beats[*idx] > min_beats[*idx] {
                beats[*idx] -= 1;
                total -= 1;
                modified = true;
                if total <= target {
                    break;
                }
            }
        }
        if !modified {
            break;
        }
        safety += 1;
    }
}

fn build_theme_descriptor(prompt: &str, templates: &[SectionTemplate]) -> ThemeDescriptor {
    let prompt_lower = prompt.to_lowercase();
    let mut instrumentation = extract_keywords(&prompt_lower, INSTRUMENT_KEYWORDS);
    if instrumentation.is_empty() {
        instrumentation = DEFAULT_INSTRUMENTATION.iter().map(|item| (*item).to_string()).collect();
    }

    let rhythm = derive_rhythm(&prompt_lower, templates);
    let motif = derive_motif(prompt);
    let texture = derive_texture(&prompt_lower, &instrumentation);
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

fn derive_rhythm(prompt_lower: &str, templates: &[SectionTemplate]) -> String {
    for (keyword, label) in RHYTHM_KEYWORDS {
        if prompt_lower.contains(keyword) {
            return (*label).to_string();
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

fn derive_texture(prompt_lower: &str, instrumentation: &[String]) -> String {
    for (keyword, label) in TEXTURE_KEYWORDS {
        if prompt_lower.contains(keyword) {
            return (*label).to_string();
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

    let mut rendered = template.to_string();
    rendered = rendered.replace("{prompt}", prompt);
    rendered = rendered.replace("{motif}", descriptor.motif.as_str());
    rendered = rendered.replace("{instrumentation}", instrumentation_text.as_str());
    rendered = rendered.replace("{rhythm}", descriptor.rhythm.as_str());
    rendered = rendered.replace("{texture}", texture.as_str());
    rendered = rendered.replace("{dynamic}", dynamic.as_str());
    rendered.trim().to_string()
}

fn select_templates(duration_seconds: u8) -> Vec<SectionTemplate> {
    if duration_seconds >= 24 {
        vec![
            SectionTemplate {
                role: SectionRole::Intro,
                label: "Arrival",
                energy: SectionEnergy::Low,
                bars: 4,
                prompt_template: "Set a {texture} scene for {prompt} by introducing the {motif} motif with {instrumentation} over a {rhythm} pulse.",
                transition: Some("Fade in layers"),
            },
            SectionTemplate {
                role: SectionRole::Motif,
                label: "Statement",
                energy: SectionEnergy::Medium,
                bars: 8,
                prompt_template: "Deliver the core {motif} motif through {instrumentation}, keeping the {rhythm} driving as {dynamic} begins.",
                transition: Some("Build momentum"),
            },
            SectionTemplate {
                role: SectionRole::Development,
                label: "Development",
                energy: SectionEnergy::High,
                bars: 8,
                prompt_template: "Evolve the {motif} motif with adventurous variations, letting {instrumentation} weave syncopations over the {rhythm} while {dynamic} unfolds.",
                transition: Some("Evolve harmonies"),
            },
            SectionTemplate {
                role: SectionRole::Bridge,
                label: "Bridge",
                energy: SectionEnergy::Medium,
                bars: 4,
                prompt_template: "Shift the palette gently, morphing {instrumentation} into new colours yet echoing the {motif} motif within the {rhythm} feel.",
                transition: Some("Prepare resolution"),
            },
            SectionTemplate {
                role: SectionRole::Resolution,
                label: "Resolution",
                energy: SectionEnergy::Medium,
                bars: 4,
                prompt_template: "Guide the {motif} motif toward resolution, using {instrumentation} to ease the {rhythm} while highlighting {dynamic}.",
                transition: Some("Return home"),
            },
            SectionTemplate {
                role: SectionRole::Outro,
                label: "Release",
                energy: SectionEnergy::Low,
                bars: 4,
                prompt_template: "Let the {motif} motif dissolve into ambience as {instrumentation} softens atop the {rhythm}, allowing {dynamic} to close the journey.",
                transition: Some("Fade to silence"),
            },
        ]
    } else if duration_seconds >= 16 {
        vec![
            SectionTemplate {
                role: SectionRole::Intro,
                label: "Lead-in",
                energy: SectionEnergy::Low,
                bars: 4,
                prompt_template: "Open gently with {instrumentation}, introducing the {motif} motif against a {rhythm} pulse that hints at {prompt}.",
                transition: Some("Invite motif"),
            },
            SectionTemplate {
                role: SectionRole::Motif,
                label: "Motif A",
                energy: SectionEnergy::Medium,
                bars: 8,
                prompt_template: "Present the {motif} motif clearly, keeping {instrumentation} tight around the {rhythm} while {dynamic} grows.",
                transition: Some("Increase energy"),
            },
            SectionTemplate {
                role: SectionRole::Development,
                label: "Variation",
                energy: SectionEnergy::High,
                bars: 6,
                prompt_template: "Develop the {motif} motif with rhythmic twists, letting {instrumentation} ride the {rhythm} as {dynamic} intensifies.",
                transition: Some("Soften textures"),
            },
            SectionTemplate {
                role: SectionRole::Resolution,
                label: "Cadence",
                energy: SectionEnergy::Medium,
                bars: 4,
                prompt_template: "Ease the energy back, guiding {instrumentation} to resolve the {motif} motif and settle the {rhythm} with {dynamic}.",
                transition: Some("Release"),
            },
            SectionTemplate {
                role: SectionRole::Outro,
                label: "Tail",
                energy: SectionEnergy::Low,
                bars: 2,
                prompt_template: "Conclude with a gentle echo of the {motif} motif, letting {instrumentation} and the {rhythm} fade as {dynamic} sighs out.",
                transition: Some("Fade"),
            },
        ]
    } else {
        vec![
            SectionTemplate {
                role: SectionRole::Intro,
                label: "Intro",
                energy: SectionEnergy::Low,
                bars: 2,
                prompt_template: "Set a delicate entrance, introducing the {motif} motif with {instrumentation} over a {rhythm} that nods to {prompt}.",
                transition: Some("Introduce motif"),
            },
            SectionTemplate {
                role: SectionRole::Motif,
                label: "Motif",
                energy: SectionEnergy::Medium,
                bars: 6,
                prompt_template: "Deliver the {motif} motif in full, keeping {instrumentation} aligned with the {rhythm} while {dynamic} expands.",
                transition: Some("Lift energy"),
            },
            SectionTemplate {
                role: SectionRole::Resolution,
                label: "Resolve",
                energy: SectionEnergy::Medium,
                bars: 4,
                prompt_template: "Resolve the story with {instrumentation}, letting the {motif} motif relax across the {rhythm} as {dynamic} softens.",
                transition: Some("Fade out"),
            },
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_plan_with_at_least_three_sections() {
        let planner = CompositionPlanner::new();
        let plan = planner.build_plan("dreamy piano", 16, Some(42));
        assert!(plan.sections.len() >= 3);
        assert_eq!(plan.version, PLAN_VERSION);
        assert!(plan.total_duration_seconds > 0.0);
        assert!(plan.total_duration_seconds >= MIN_TOTAL_SECONDS);
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
