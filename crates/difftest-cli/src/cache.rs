use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct CacheManifest {
    pub version: u32,
    pub entries: HashMap<String, CacheEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub image_path: String,
    pub model_id: String,
    pub prompt: String,
    pub seed: u64,
    pub generator: String,
    pub created_at: String,
}

impl CacheManifest {
    pub fn load(path: &Path) -> Self {
        if path.exists() {
            if let Ok(content) = std::fs::read_to_string(path) {
                if let Ok(manifest) = serde_json::from_str::<CacheManifest>(&content) {
                    return manifest;
                }
            }
        }
        Self {
            version: 1,
            entries: HashMap::new(),
        }
    }

    pub fn save(&self, path: &Path) -> difftest_core::error::Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn cache_key(model_id: &str, prompt: &str, seed: u64, generator: &str) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(model_id.as_bytes());
        hasher.update(b"|");
        hasher.update(prompt.as_bytes());
        hasher.update(b"|");
        hasher.update(seed.to_le_bytes());
        hasher.update(b"|");
        hasher.update(generator.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    pub fn lookup(&self, key: &str) -> Option<&CacheEntry> {
        self.entries.get(key)
    }

    pub fn insert(&mut self, key: String, entry: CacheEntry) {
        self.entries.insert(key, entry);
    }
}
