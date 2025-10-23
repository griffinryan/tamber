use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::types::{CompositionPlan, CompositionSection, SectionEnergy, SectionRole};

const PLAN_VERSION: &str = "v1";
const MIN_TEMPO: u16 = 60;
const MAX_TEMPO: u16 = 140;
const DEFAULT_TIME_SIGNATURE: &str = "4/4";

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
        let templates = select_templates(duration_seconds);
        let total_bars: u16 = templates.iter().map(|tpl| tpl.bars as u16).sum();
        let seconds_total = duration_seconds.max(4) as f32;
        let seconds_per_bar = seconds_total / total_bars as f32;
        let raw_tempo = (240.0 * total_bars as f32 / seconds_total).round() as u16;
        let tempo_bpm = raw_tempo.clamp(MIN_TEMPO, MAX_TEMPO);

        let base_seed = seed.unwrap_or_else(|| deterministic_seed(prompt));
        let key = select_key(base_seed);

        let mut sections: Vec<CompositionSection> = Vec::with_capacity(templates.len());
        let mut accumulated: f32 = 0.0;

        for (index, template) in templates.iter().enumerate() {
            let mut section_seconds = template.bars as f32 * seconds_per_bar;
            if index == templates.len() - 1 {
                let planned_total: f32 =
                    sections.iter().map(|s| s.target_seconds).sum::<f32>() + section_seconds;
                let delta = seconds_total - planned_total;
                section_seconds += delta;
            }

            let section_prompt =
                template.prompt_template.replace("{prompt}", prompt).trim().to_string();

            sections.push(CompositionSection {
                section_id: format!("s{:02}", index),
                role: template.role.clone(),
                label: template.label.to_string(),
                prompt: section_prompt,
                bars: template.bars,
                target_seconds: section_seconds.max(2.0),
                energy: template.energy.clone(),
                model_id: None,
                seed_offset: Some(index as i32),
                transition: template.transition.map(|text| text.to_string()),
            });
            accumulated += section_seconds;
        }

        CompositionPlan {
            version: PLAN_VERSION.to_string(),
            tempo_bpm,
            time_signature: DEFAULT_TIME_SIGNATURE.to_string(),
            key,
            total_bars,
            total_duration_seconds: accumulated,
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
    }
}
