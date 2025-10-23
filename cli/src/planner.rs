use crate::types::{CompositionPlan, CompositionSection, SectionEnergy, SectionRole};
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

        let mut sections: Vec<CompositionSection> = Vec::with_capacity(templates.len());

        for (index, (template, section_seconds)) in
            templates.iter().zip(seconds_per_section).enumerate()
        {
            let section_prompt =
                template.prompt_template.replace("{prompt}", prompt).trim().to_string();
            let target_seconds = section_seconds.max(MIN_SECTION_SECONDS);

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
            });
        }

        CompositionPlan {
            version: PLAN_VERSION.to_string(),
            tempo_bpm,
            time_signature: DEFAULT_TIME_SIGNATURE.to_string(),
            key,
            total_bars,
            total_duration_seconds: sections.iter().map(|s| s.target_seconds).sum(),
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

fn select_templates(duration_seconds: u8) -> Vec<SectionTemplate> {
    if duration_seconds >= 24 {
        vec![
            SectionTemplate {
                role: SectionRole::Intro,
                label: "Arrival",
                energy: SectionEnergy::Low,
                bars: 4,
                prompt_template: "Set the stage with a hushed texture hinting at {prompt}.",
                transition: Some("Fade in layers"),
            },
            SectionTemplate {
                role: SectionRole::Motif,
                label: "Statement",
                energy: SectionEnergy::Medium,
                bars: 8,
                prompt_template: "Introduce a memorable motif expressing {prompt}.",
                transition: Some("Build momentum"),
            },
            SectionTemplate {
                role: SectionRole::Development,
                label: "Development",
                energy: SectionEnergy::High,
                bars: 8,
                prompt_template:
                    "Expand the motif with syncopation and evolving harmony around {prompt}.",
                transition: Some("Evolve harmonies"),
            },
            SectionTemplate {
                role: SectionRole::Bridge,
                label: "Bridge",
                energy: SectionEnergy::Medium,
                bars: 4,
                prompt_template: "Introduce contrast with modal colors echoing {prompt}.",
                transition: Some("Prepare resolution"),
            },
            SectionTemplate {
                role: SectionRole::Resolution,
                label: "Resolution",
                energy: SectionEnergy::Medium,
                bars: 4,
                prompt_template: "Resolve tension with satisfying cadences fulfilling {prompt}.",
                transition: Some("Return home"),
            },
            SectionTemplate {
                role: SectionRole::Outro,
                label: "Release",
                energy: SectionEnergy::Low,
                bars: 4,
                prompt_template: "Let the textures dissolve, echoing the atmosphere of {prompt}.",
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
                prompt_template: "Open gently and establish the mood of {prompt}.",
                transition: Some("Invite motif"),
            },
            SectionTemplate {
                role: SectionRole::Motif,
                label: "Motif A",
                energy: SectionEnergy::Medium,
                bars: 8,
                prompt_template: "Present a clear melodic phrase inspired by {prompt}.",
                transition: Some("Increase energy"),
            },
            SectionTemplate {
                role: SectionRole::Development,
                label: "Variation",
                energy: SectionEnergy::High,
                bars: 6,
                prompt_template:
                    "Develop the motif with rhythmic motion and harmonic color tied to {prompt}.",
                transition: Some("Soften textures"),
            },
            SectionTemplate {
                role: SectionRole::Resolution,
                label: "Cadence",
                energy: SectionEnergy::Medium,
                bars: 4,
                prompt_template: "Resolve back to calm, letting {prompt} settle.",
                transition: Some("Release"),
            },
            SectionTemplate {
                role: SectionRole::Outro,
                label: "Tail",
                energy: SectionEnergy::Low,
                bars: 2,
                prompt_template: "Conclude with a gentle echo of {prompt}.",
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
                prompt_template: "Set a delicate entrance referencing {prompt}.",
                transition: Some("Introduce motif"),
            },
            SectionTemplate {
                role: SectionRole::Motif,
                label: "Motif",
                energy: SectionEnergy::Medium,
                bars: 6,
                prompt_template: "Deliver a simple, emotive motif capturing {prompt}.",
                transition: Some("Lift energy"),
            },
            SectionTemplate {
                role: SectionRole::Resolution,
                label: "Resolve",
                energy: SectionEnergy::Medium,
                bars: 4,
                prompt_template: "Resolve with warm chords that fulfill {prompt}.",
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
    }

    #[test]
    fn collapses_short_duration_into_minimum_sections() {
        let planner = CompositionPlanner::new();
        let plan = planner.build_plan("short clip", 8, None);
        assert!(plan.sections.len() <= 2);
        assert!(plan.sections.iter().any(|section| matches!(section.role, SectionRole::Motif)));
        assert!(plan.sections.iter().all(|section| section.target_seconds >= 2.0));
    }
}
