use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSuite {
    pub tests: Vec<TestCase>,
    pub config: SuiteConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCase {
    pub name: String,
    pub prompts: Vec<String>,
    pub seeds: Vec<u64>,
    pub metrics: Vec<MetricSpec>,
    pub thresholds: HashMap<String, f64>,
    pub test_type: TestType,
    pub baseline_dir: Option<String>,
    pub reference_dir: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TestType {
    Quality,
    VisualRegression,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSpec {
    pub name: String,
    pub compute_in: ComputeBackend,
    pub category: MetricCategory,
    pub direction: MetricDirection,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ComputeBackend {
    Rust,
    Python,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MetricCategory {
    PerSample,
    Batch,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MetricDirection {
    HigherIsBetter,
    LowerIsBetter,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuiteConfig {
    pub output_dir: PathBuf,
    pub model_id: String,
    pub device: String,
    pub generator: String,
    pub generator_config: HashMap<String, String>,
}

impl Default for SuiteConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from(".difftest/outputs"),
            model_id: String::new(),
            device: "cpu".to_string(),
            generator: "diffusers".to_string(),
            generator_config: HashMap::new(),
        }
    }
}
