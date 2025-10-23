use anyhow::{anyhow, Context, Result};
use directories::ProjectDirs;
use serde::Deserialize;
use std::{
    env, fs,
    path::{Path, PathBuf},
};

const CONFIG_FILE_NAME: &str = "config.toml";
const ENV_CONFIG_PATH: &str = "TIMBRE_CONFIG_PATH";
const ENV_WORKER_URL: &str = "TIMBRE_WORKER_URL";
const ENV_DEFAULT_MODEL: &str = "TIMBRE_DEFAULT_MODEL";
const ENV_DEFAULT_DURATION: &str = "TIMBRE_DEFAULT_DURATION";
const ENV_ARTIFACT_DIR: &str = "TIMBRE_ARTIFACT_DIR";

#[derive(Debug, Clone)]
pub struct AppConfig {
    worker_url: Option<String>,
    default_model_id: String,
    default_duration_seconds: u8,
    artifact_dir: PathBuf,
}

impl AppConfig {
    pub fn load() -> Result<Self> {
        let mut config = Self::default();

        if let Ok(path) = Self::default_config_path() {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).with_context(|| {
                    format!("failed to create config directory {}", parent.display())
                })?;
            }
        }

        if let Some(path) = config_file_override()? {
            if path.exists() {
                let partial = read_partial(&path)?;
                config.apply_partial(partial);
            }
        } else {
            let path = Self::default_config_path()?;
            if path.exists() {
                let partial = read_partial(&path)?;
                config.apply_partial(partial);
            }
        }

        config.apply_env()?;
        Ok(config)
    }

    pub fn worker_url(&self) -> Option<&str> {
        self.worker_url.as_deref()
    }

    pub fn default_model_id(&self) -> &str {
        &self.default_model_id
    }

    pub fn default_duration_seconds(&self) -> u8 {
        self.default_duration_seconds
    }

    pub fn artifact_dir(&self) -> &PathBuf {
        &self.artifact_dir
    }

    pub fn default_config_path() -> Result<PathBuf> {
        let dirs = ProjectDirs::from("com", "Timbre", "Timbre")
            .ok_or_else(|| anyhow!("unable to determine config directory"))?;
        Ok(dirs.config_dir().join(CONFIG_FILE_NAME))
    }

    fn apply_partial(&mut self, partial: PartialConfig) {
        if let Some(url) = partial.worker_url {
            self.worker_url = Some(url);
        }
        if let Some(model_id) = partial.default_model_id {
            self.default_model_id = model_id;
        }
        if let Some(duration) = partial.default_duration_seconds {
            self.default_duration_seconds = duration;
        }
        if let Some(dir) = partial.artifact_dir {
            self.artifact_dir = dir;
        }
    }

    fn apply_env(&mut self) -> Result<()> {
        if let Ok(value) = env::var(ENV_WORKER_URL) {
            if value.trim().is_empty() {
                self.worker_url = None;
            } else {
                self.worker_url = Some(value);
            }
        }
        if let Ok(value) = env::var(ENV_DEFAULT_MODEL) {
            if !value.trim().is_empty() {
                self.default_model_id = value;
            }
        }
        if let Ok(value) = env::var(ENV_DEFAULT_DURATION) {
            if !value.trim().is_empty() {
                let parsed = value
                    .parse::<u8>()
                    .context("TIMBRE_DEFAULT_DURATION must be an integer between 1-30")?;
                self.default_duration_seconds = parsed;
            }
        }
        if let Ok(value) = env::var(ENV_ARTIFACT_DIR) {
            if !value.trim().is_empty() {
                self.artifact_dir = PathBuf::from(value);
            }
        }
        Ok(())
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            worker_url: None,
            default_model_id: "riffusion-v1".into(),
            default_duration_seconds: 24,
            artifact_dir: default_artifact_dir(),
        }
    }
}

fn config_file_override() -> Result<Option<PathBuf>> {
    if let Some(value) = env::var_os(ENV_CONFIG_PATH) {
        if value.is_empty() {
            return Ok(None);
        }
        let path = PathBuf::from(value);
        if path.is_file() {
            return Ok(Some(path));
        }
        if path.ends_with(CONFIG_FILE_NAME) {
            return Ok(Some(path));
        }
        if path.is_dir() {
            return Ok(Some(path.join(CONFIG_FILE_NAME)));
        }
        return Ok(Some(path));
    }
    Ok(None)
}

fn read_partial(path: &Path) -> Result<PartialConfig> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("failed to read config file at {}", path.display()))?;
    let partial: PartialConfig =
        toml::from_str(&contents).with_context(|| format!("failed to parse {}", path.display()))?;
    Ok(partial)
}

fn default_artifact_dir() -> PathBuf {
    env::var_os("HOME")
        .map(PathBuf::from)
        .map(|home| home.join("Music").join("Timbre"))
        .unwrap_or_else(|| PathBuf::from("./artifacts"))
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct PartialConfig {
    worker_url: Option<String>,
    default_model_id: Option<String>,
    default_duration_seconds: Option<u8>,
    artifact_dir: Option<PathBuf>,
}
