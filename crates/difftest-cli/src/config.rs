use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct DifftestConfig {
    pub model: Option<String>,
    pub device: Option<String>,
    pub test_dir: Option<String>,
    pub generator: Option<String>,
    pub output: Option<String>,
    pub html: Option<String>,
    pub junit: Option<String>,
    pub markdown: Option<String>,
    pub comfyui_url: Option<String>,
    pub workflow: Option<String>,
    pub provider: Option<String>,
    pub api_key: Option<String>,
    pub endpoint: Option<String>,
    pub baseline_dir: Option<String>,
    pub timeout: Option<u64>,
    pub image_timeout: Option<u64>,
    pub incremental: Option<bool>,
    pub filter: Option<String>,
    pub test: Option<Vec<String>>,
    pub retry: Option<RetryConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub base_delay: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: 1.0,
        }
    }
}

/// Wrapper for pyproject.toml [tool.difftest] section.
#[derive(Debug, Deserialize)]
struct PyProjectToml {
    tool: Option<PyProjectTool>,
}

#[derive(Debug, Deserialize)]
struct PyProjectTool {
    difftest: Option<DifftestConfig>,
}

pub fn load_config() -> DifftestConfig {
    // 1. Try difftest.toml
    let difftest_toml = Path::new("difftest.toml");
    if difftest_toml.exists() {
        if let Ok(content) = std::fs::read_to_string(difftest_toml) {
            match toml::from_str::<DifftestConfig>(&content) {
                Ok(config) => {
                    eprintln!("Using config from difftest.toml");
                    return config;
                }
                Err(e) => {
                    eprintln!("Warning: failed to parse difftest.toml: {e}");
                }
            }
        }
    }

    // 2. Try pyproject.toml [tool.difftest]
    let pyproject_toml = Path::new("pyproject.toml");
    if pyproject_toml.exists() {
        if let Ok(content) = std::fs::read_to_string(pyproject_toml) {
            if let Ok(pyproject) = toml::from_str::<PyProjectToml>(&content) {
                if let Some(tool) = pyproject.tool {
                    if let Some(config) = tool.difftest {
                        eprintln!("Using config from pyproject.toml [tool.difftest]");
                        return config;
                    }
                }
            }
        }
    }

    // 3. Return defaults
    DifftestConfig::default()
}

/// Build the generator config HashMap from resolved values.
pub fn build_generator_config(
    model: &str,
    device: &str,
    comfyui_url: &Option<String>,
    workflow: &Option<String>,
    provider: &Option<String>,
    api_key: &Option<String>,
    endpoint: &Option<String>,
    retry: &Option<RetryConfig>,
    image_timeout: &Option<u64>,
) -> HashMap<String, String> {
    let mut config = HashMap::new();
    if !model.is_empty() {
        config.insert("model_id".to_string(), model.to_string());
    }
    config.insert("device".to_string(), device.to_string());
    if let Some(ref url) = comfyui_url {
        config.insert("comfyui_url".to_string(), url.clone());
    }
    if let Some(ref wf) = workflow {
        config.insert("workflow_path".to_string(), wf.clone());
    }
    if let Some(ref p) = provider {
        config.insert("provider".to_string(), p.clone());
    }
    if let Some(ref key) = api_key {
        config.insert("api_key".to_string(), key.clone());
    }
    if let Some(ref ep) = endpoint {
        config.insert("endpoint".to_string(), ep.clone());
    }
    if let Some(ref r) = retry {
        config.insert("max_retries".to_string(), r.max_retries.to_string());
        config.insert("base_delay".to_string(), r.base_delay.to_string());
    }
    if let Some(t) = image_timeout {
        config.insert("image_timeout".to_string(), t.to_string());
    }
    config
}

/// Resolve an Option<String> CLI arg with config fallback and hardcoded default.
pub fn resolve_string(cli: &Option<String>, config: &Option<String>, default: &str) -> String {
    cli.as_deref()
        .or(config.as_deref())
        .unwrap_or(default)
        .to_string()
}

pub fn resolve_option(cli: &Option<String>, config: &Option<String>) -> Option<String> {
    cli.clone().or_else(|| config.clone())
}

pub fn resolve_bool(cli: bool, config: &Option<bool>) -> bool {
    if cli {
        true
    } else {
        config.unwrap_or(false)
    }
}
