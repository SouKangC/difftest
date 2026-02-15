use thiserror::Error;

#[derive(Error, Debug)]
pub enum DifftestError {
    #[error("Configuration error: {field} = {value}: {reason}")]
    Config {
        field: String,
        value: String,
        reason: String,
    },
    #[error("Test discovery failed: {0}")]
    Discovery(String),
    #[error("Image generation failed: {0}")]
    Generation(String),
    #[error("Metric computation failed for '{metric}': {message}")]
    Metric { metric: String, message: String },
    #[error("Suite timeout exceeded ({timeout_seconds}s)")]
    SuiteTimeout { timeout_seconds: u64 },
    #[error("Configuration file error: {0}")]
    ConfigFile(String),
    #[error("Cache error: {0}")]
    Cache(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Serialization(#[from] serde_json::Error),
    #[error(transparent)]
    Database(#[from] rusqlite::Error),
}

pub type Result<T> = std::result::Result<T, DifftestError>;
