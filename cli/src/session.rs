use crate::types::{ClipLayer, SessionSummary};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub const LAYER_ORDER: [ClipLayer; 6] = [
    ClipLayer::Rhythm,
    ClipLayer::Bass,
    ClipLayer::Harmony,
    ClipLayer::Lead,
    ClipLayer::Textures,
    ClipLayer::Vocals,
];

pub const MAX_SCENES: usize = 3;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClipSlotStatus {
    Empty,
    Pending,
    Rendering,
    Ready,
    Failed,
}

#[derive(Debug, Clone)]
pub struct ClipSlot {
    #[allow(dead_code)]
    pub layer: ClipLayer,
    #[allow(dead_code)]
    pub scene_index: usize,
    pub job_id: Option<String>,
    pub prompt: Option<String>,
    pub status: ClipSlotStatus,
    pub artifact: Option<PathBuf>,
    pub duration_seconds: Option<f32>,
    pub bars: Option<u8>,
}

impl ClipSlot {
    pub fn new(layer: ClipLayer, scene_index: usize) -> Self {
        Self {
            layer,
            scene_index,
            job_id: None,
            prompt: None,
            status: ClipSlotStatus::Empty,
            artifact: None,
            duration_seconds: None,
            bars: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SessionScene {
    pub index: usize,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneSnapshot {
    pub index: usize,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSnapshot {
    pub session_id: Option<String>,
    pub scenes: Vec<SceneSnapshot>,
    pub focused_layer: ClipLayer,
    pub focused_scene: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_layer: Option<ClipLayer>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_scene: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct SessionView {
    scenes: Vec<SessionScene>,
    slots: HashMap<(ClipLayer, usize), ClipSlot>,
    focused_layer: ClipLayer,
    focused_scene: usize,
    pending_jobs: HashMap<String, (ClipLayer, usize)>,
    playing: HashMap<ClipLayer, usize>,
    active: Option<(ClipLayer, usize)>,
}

impl SessionView {
    pub fn new() -> Self {
        let mut view = Self::default();
        for index in 0..MAX_SCENES {
            view.ensure_scene(index);
        }
        view
    }

    pub fn focused(&self) -> (ClipLayer, usize) {
        (self.focused_layer, self.focused_scene)
    }

    pub fn focus_next_layer(&mut self) {
        let idx = LAYER_ORDER.iter().position(|layer| *layer == self.focused_layer).unwrap_or(0);
        let next = (idx + 1) % LAYER_ORDER.len();
        self.focused_layer = LAYER_ORDER[next];
    }

    pub fn focus_prev_layer(&mut self) {
        let idx = LAYER_ORDER.iter().position(|layer| *layer == self.focused_layer).unwrap_or(0);
        let prev = if idx == 0 { LAYER_ORDER.len() - 1 } else { idx - 1 };
        self.focused_layer = LAYER_ORDER[prev];
    }

    pub fn focus_next_scene(&mut self) {
        if self.scenes.is_empty() {
            return;
        }
        let idx = self.scenes.iter().position(|scene| scene.index == self.focused_scene);
        let next = idx.map(|i| (i + 1) % self.scenes.len()).unwrap_or(0);
        self.focused_scene = self.scenes[next].index;
    }

    pub fn focus_prev_scene(&mut self) {
        if self.scenes.is_empty() {
            return;
        }
        let idx = self.scenes.iter().position(|scene| scene.index == self.focused_scene);
        let prev = idx.map(|i| if i == 0 { self.scenes.len() - 1 } else { i - 1 }).unwrap_or(0);
        self.focused_scene = self.scenes[prev].index;
    }

    pub fn focus_scene(&mut self, index: usize) {
        if index >= MAX_SCENES {
            return;
        }
        self.ensure_scene(index);
        self.focused_scene = index;
    }

    pub fn focus_layer(&mut self, layer: ClipLayer) {
        self.focused_layer = layer;
    }

    pub fn ensure_scene(&mut self, index: usize) {
        if index >= MAX_SCENES {
            return;
        }
        if self.scenes.iter().all(|scene| scene.index != index) {
            let name = format!("Scene {}", index + 1);
            self.scenes.push(SessionScene { index, name });
            self.scenes.sort_by_key(|scene| scene.index);
        }
        for &layer in &LAYER_ORDER {
            self.slots.entry((layer, index)).or_insert_with(|| ClipSlot::new(layer, index));
        }
    }

    pub fn add_scene_after_focus(&mut self) -> Option<usize> {
        if self.scenes.len() >= MAX_SCENES {
            return None;
        }
        let mut candidate = 0;
        while candidate < MAX_SCENES && self.scenes.iter().any(|scene| scene.index == candidate) {
            candidate += 1;
        }
        if candidate >= MAX_SCENES {
            return None;
        }
        self.ensure_scene(candidate);
        Some(candidate)
    }

    pub fn rename_scene(&mut self, index: usize, name: String) {
        if let Some(scene) = self.scenes.iter_mut().find(|scene| scene.index == index) {
            scene.name = name;
        }
    }

    pub fn scenes(&self) -> &[SessionScene] {
        &self.scenes
    }

    pub fn scene_name(&self, index: usize) -> String {
        self.scenes
            .iter()
            .find(|scene| scene.index == index)
            .map(|scene| scene.name.clone())
            .unwrap_or_else(|| scene_name(index))
    }

    pub fn restore(&mut self, snapshot: &SessionSnapshot) {
        self.scenes.clear();
        self.slots.clear();
        self.pending_jobs.clear();
        self.playing.clear();
        for index in 0..MAX_SCENES {
            self.ensure_scene(index);
        }
        for scene in &snapshot.scenes {
            if scene.index >= MAX_SCENES {
                continue;
            }
            if let Some(existing) = self.scenes.iter_mut().find(|s| s.index == scene.index) {
                existing.name = scene.name.clone();
            }
        }
        self.focused_layer = snapshot.focused_layer;
        self.focused_scene = snapshot.focused_scene.min(MAX_SCENES.saturating_sub(1));
        let active_scene = snapshot.active_scene.unwrap_or(self.focused_scene);
        if let Some(layer) = snapshot.active_layer {
            if active_scene < MAX_SCENES {
                self.active = Some((layer, active_scene));
            } else {
                self.active = None;
            }
        } else {
            self.active = None;
        }
    }

    pub fn snapshot(&self, session_id: Option<String>) -> SessionSnapshot {
        let scenes = self
            .scenes
            .iter()
            .map(|scene| SceneSnapshot { index: scene.index, name: scene.name.clone() })
            .collect();
        let active = self.active.filter(|(_, scene)| *scene < MAX_SCENES);
        SessionSnapshot {
            session_id,
            scenes,
            focused_layer: self.focused_layer,
            focused_scene: self.focused_scene.min(MAX_SCENES.saturating_sub(1)),
            active_layer: active.map(|(layer, _)| layer),
            active_scene: active.map(|(_, scene)| scene),
        }
    }

    pub fn slot(&self, layer: ClipLayer, scene_index: usize) -> &ClipSlot {
        self.slots.get(&(layer, scene_index)).expect("slot must exist")
    }

    pub fn slot_mut(&mut self, layer: ClipLayer, scene_index: usize) -> &mut ClipSlot {
        self.slots.entry((layer, scene_index)).or_insert_with(|| ClipSlot::new(layer, scene_index))
    }

    pub fn mark_pending(
        &mut self,
        layer: ClipLayer,
        scene_index: usize,
        job_id: String,
        prompt: String,
    ) {
        self.ensure_scene(scene_index);
        let slot = self.slot_mut(layer, scene_index);
        slot.job_id = Some(job_id.clone());
        slot.prompt = Some(prompt);
        slot.status = ClipSlotStatus::Pending;
        slot.artifact = None;
        self.pending_jobs.insert(job_id, (layer, scene_index));
    }

    pub fn mark_status(&mut self, job_id: &str, status: ClipSlotStatus) {
        if let Some((layer, scene)) = self.pending_jobs.get(job_id).copied() {
            let slot = self.slot_mut(layer, scene);
            let is_failed = matches!(status, ClipSlotStatus::Failed);
            slot.status = status;
            if is_failed {
                slot.artifact = None;
            }
        }
    }

    pub fn mark_ready(
        &mut self,
        job_id: &str,
        artifact_path: &Path,
        duration_seconds: f32,
        bars: Option<u8>,
    ) {
        if let Some((layer, scene)) = self.pending_jobs.get(job_id).copied() {
            let slot = self.slot_mut(layer, scene);
            slot.status = ClipSlotStatus::Ready;
            slot.artifact = Some(artifact_path.to_path_buf());
            slot.duration_seconds = Some(duration_seconds);
            slot.bars = bars;
        }
    }

    pub fn mark_failed(&mut self, job_id: &str) {
        self.mark_status(job_id, ClipSlotStatus::Failed);
    }

    pub fn clear_job(&mut self, job_id: &str) {
        if let Some((layer, scene)) = self.pending_jobs.remove(job_id) {
            let slot = self.slot_mut(layer, scene);
            if slot.job_id.as_deref() == Some(job_id) {
                slot.job_id = None;
            }
        }
    }

    pub fn apply_summary(&mut self, summary: &SessionSummary) {
        for clip in &summary.clips {
            let Some(layer) = clip.layer else { continue };
            let scene_index = clip.scene_index.unwrap_or(0) as usize;
            if scene_index < MAX_SCENES {
                self.ensure_scene(scene_index);
            }
            let slot = self.slot_mut(layer, scene_index);
            slot.prompt = Some(clip.prompt.clone());
            slot.job_id = Some(clip.job_id.clone());
            slot.status = match clip.state {
                crate::types::JobState::Queued => ClipSlotStatus::Pending,
                crate::types::JobState::Running => ClipSlotStatus::Rendering,
                crate::types::JobState::Succeeded => ClipSlotStatus::Ready,
                crate::types::JobState::Failed => ClipSlotStatus::Failed,
            };
            slot.duration_seconds = clip.duration_seconds;
            slot.bars = clip.bars;
            slot.artifact = clip.artifact_path.as_ref().map(PathBuf::from);
        }
        if let Some(scene) = self.scenes.iter().find(|scene| scene.index == self.focused_scene) {
            self.focused_scene = scene.index;
        } else if let Some(first) = self.scenes.first() {
            self.focused_scene = first.index;
        }
    }

    pub fn layer_playing(&self, layer: ClipLayer) -> Option<usize> {
        self.playing.get(&layer).copied()
    }

    pub fn set_playing(&mut self, layer: ClipLayer, scene_index: usize) {
        if scene_index >= MAX_SCENES {
            return;
        }
        self.playing.insert(layer, scene_index);
        self.focused_layer = layer;
        self.focused_scene = scene_index;
    }

    pub fn clear_playing(&mut self, layer: ClipLayer) {
        self.playing.remove(&layer);
    }

    #[allow(dead_code)]
    pub fn clear_all_playing(&mut self) {
        self.playing.clear();
    }

    pub fn pending_scene_for_job(&self, job_id: &str) -> Option<(ClipLayer, usize)> {
        self.pending_jobs.get(job_id).copied()
    }

    pub fn active_target(&self) -> Option<(ClipLayer, usize)> {
        self.active
    }

    pub fn toggle_active_target(&mut self, layer: ClipLayer, scene_index: usize) -> bool {
        if scene_index >= MAX_SCENES {
            return false;
        }
        self.ensure_scene(scene_index);
        let target = Some((layer, scene_index));
        if self.active == target {
            self.active = None;
            false
        } else {
            self.active = target;
            true
        }
    }
}

impl Default for SessionView {
    fn default() -> Self {
        Self {
            scenes: Vec::new(),
            slots: HashMap::new(),
            focused_layer: ClipLayer::Rhythm,
            focused_scene: 0,
            pending_jobs: HashMap::new(),
            playing: HashMap::new(),
            active: None,
        }
    }
}

pub fn clip_layer_label(layer: ClipLayer) -> &'static str {
    match layer {
        ClipLayer::Rhythm => "Rhythm",
        ClipLayer::Bass => "Bass",
        ClipLayer::Harmony => "Harmony",
        ClipLayer::Lead => "Lead",
        ClipLayer::Textures => "Textures",
        ClipLayer::Vocals => "Vocals",
    }
}

pub fn scene_name(index: usize) -> String {
    format!("Scene {}", index + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handles_session_view_limits_to_three_scenes() {
        let mut view = SessionView::new();
        assert_eq!(view.scenes().len(), MAX_SCENES);
        assert!(view.scenes().iter().map(|scene| scene.index).eq(0..MAX_SCENES));
        assert!(view.add_scene_after_focus().is_none());

        let mut seen_slots = 0;
        for layer in LAYER_ORDER {
            for scene_index in 0..MAX_SCENES {
                let slot = view.slot(layer, scene_index);
                assert!(slot.prompt.is_none());
                seen_slots += 1;
            }
        }
        assert_eq!(seen_slots, LAYER_ORDER.len() * MAX_SCENES);
    }

    #[test]
    fn handles_focus_scene_ignores_out_of_range() {
        let mut view = SessionView::new();
        let initial = view.focused();
        view.focus_scene(MAX_SCENES + 2);
        assert_eq!(view.focused(), initial);
    }

    #[test]
    fn handles_restore_clamps_snapshot_scenes() {
        let mut view = SessionView::new();
        let snapshot = SessionSnapshot {
            session_id: Some("session-123".to_string()),
            scenes: vec![
                SceneSnapshot { index: 0, name: "Alpha".to_string() },
                SceneSnapshot { index: 1, name: "Beta".to_string() },
                SceneSnapshot { index: 2, name: "Gamma".to_string() },
                SceneSnapshot { index: 5, name: "Overflow".to_string() },
            ],
            focused_layer: ClipLayer::Lead,
            focused_scene: 4,
            active_layer: Some(ClipLayer::Bass),
            active_scene: Some(4),
        };

        view.restore(&snapshot);

        assert_eq!(view.scenes().len(), MAX_SCENES);
        assert_eq!(view.scene_name(0), "Alpha");
        assert_eq!(view.scene_name(1), "Beta");
        assert_eq!(view.scene_name(2), "Gamma");
        assert_eq!(view.focused(), (ClipLayer::Lead, MAX_SCENES - 1));
        assert!(view.active_target().is_none());
    }
}
