use crate::session::SessionSnapshot;
use anyhow::{anyhow, Context, Result};
use directories::ProjectDirs;
use std::fs;
use std::path::PathBuf;

const SNAPSHOT_FILE: &str = "session.json";

fn snapshot_path() -> Result<PathBuf> {
    let dirs = ProjectDirs::from("com", "Timbre", "Timbre")
        .ok_or_else(|| anyhow!("unable to determine config directory"))?;
    let path = dirs.config_dir().join(SNAPSHOT_FILE);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create config directory {}", parent.display()))?;
    }
    Ok(path)
}

pub fn load_snapshot() -> Result<Option<SessionSnapshot>> {
    let path = snapshot_path()?;
    if !path.exists() {
        return Ok(None);
    }
    let data = fs::read_to_string(&path)
        .with_context(|| format!("failed to read session snapshot at {}", path.display()))?;
    let snapshot = serde_json::from_str(&data)
        .with_context(|| format!("failed to parse session snapshot {}", path.display()))?;
    Ok(Some(snapshot))
}

pub fn save_snapshot(snapshot: &SessionSnapshot) -> Result<()> {
    let path = snapshot_path()?;
    let data =
        serde_json::to_string_pretty(snapshot).context("failed to encode session snapshot")?;
    fs::write(&path, data)
        .with_context(|| format!("failed to write session snapshot to {}", path.display()))
}
