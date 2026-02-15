use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::suite::{MetricDirection, TestCase};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuiteResult {
    pub results: Vec<TestResult>,
    pub total_passed: usize,
    pub total_failed: usize,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub name: String,
    pub passed: bool,
    pub metrics: HashMap<String, MetricResult>,
    pub images: Vec<GeneratedImage>,
    pub duration_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricResult {
    pub per_sample: Vec<f64>,
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub threshold: f64,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedImage {
    pub path: String,
    pub prompt: String,
    pub seed: u64,
}

#[derive(Debug, Clone)]
pub struct RunConfig {
    pub model_id: String,
    pub device: String,
    pub output_dir: String,
    pub test_dir: String,
}

impl MetricResult {
    pub fn from_scores(scores: Vec<f64>, threshold: f64) -> Self {
        Self::from_scores_with_direction(scores, threshold, &MetricDirection::HigherIsBetter)
    }

    pub fn from_scores_with_direction(
        scores: Vec<f64>,
        threshold: f64,
        direction: &MetricDirection,
    ) -> Self {
        let mean = if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f64>() / scores.len() as f64
        };
        let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let passed = match direction {
            MetricDirection::HigherIsBetter => mean >= threshold,
            MetricDirection::LowerIsBetter => mean <= threshold,
        };

        Self {
            per_sample: scores,
            mean,
            min,
            max,
            threshold,
            passed,
        }
    }
}

impl SuiteResult {
    pub fn from_results(results: Vec<TestResult>, duration_ms: u64) -> Self {
        let total_passed = results.iter().filter(|r| r.passed).count();
        let total_failed = results.len() - total_passed;

        Self {
            results,
            total_passed,
            total_failed,
            duration_ms,
        }
    }
}

impl TestResult {
    pub fn from_metrics(
        test_case: &TestCase,
        metrics: HashMap<String, MetricResult>,
        images: Vec<GeneratedImage>,
        duration_ms: u64,
    ) -> Self {
        let passed = metrics.values().all(|m| m.passed);

        Self {
            name: test_case.name.clone(),
            passed,
            metrics,
            images,
            duration_ms,
        }
    }
}
