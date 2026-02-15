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
    #[serde(default)]
    pub ci_lower: Option<f64>,
    #[serde(default)]
    pub ci_upper: Option<f64>,
    #[serde(default)]
    pub std_dev: Option<f64>,
    #[serde(default)]
    pub sample_count: usize,
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

/// Hardcoded t-critical values for 95% confidence interval (two-tailed).
/// Maps degrees of freedom (df = n - 1) to t-critical value.
fn t_critical_95(df: usize) -> f64 {
    match df {
        1 => 12.706,
        2 => 4.303,
        3 => 3.182,
        4 => 2.776,
        5 => 2.571,
        6 => 2.447,
        7 => 2.365,
        8 => 2.306,
        9 => 2.262,
        10 => 2.228,
        11 => 2.201,
        12 => 2.179,
        13 => 2.160,
        14 => 2.145,
        15 => 2.131,
        16 => 2.120,
        17 => 2.110,
        18 => 2.101,
        19 => 2.093,
        20 => 2.086,
        21 => 2.080,
        22 => 2.074,
        23 => 2.069,
        24 => 2.064,
        25 => 2.060,
        26 => 2.056,
        27 => 2.052,
        28 => 2.048,
        29 => 2.045,
        30 => 2.042,
        31..=40 => 2.021,
        41..=60 => 2.000,
        61..=120 => 1.980,
        _ => 1.960, // z-value for large samples
    }
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
        let n = scores.len();
        let mean = if n == 0 {
            0.0
        } else {
            scores.iter().sum::<f64>() / n as f64
        };
        let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let passed = match direction {
            MetricDirection::HigherIsBetter => mean >= threshold,
            MetricDirection::LowerIsBetter => mean <= threshold,
        };

        // Compute standard deviation and confidence interval
        let (std_dev, ci_lower, ci_upper) = if n >= 2 {
            let variance = scores.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
            let sd = variance.sqrt();
            let df = n - 1;
            let t = t_critical_95(df);
            let margin = t * (sd / (n as f64).sqrt());
            (Some(sd), Some(mean - margin), Some(mean + margin))
        } else {
            (None, None, None)
        };

        Self {
            per_sample: scores,
            mean,
            min,
            max,
            threshold,
            passed,
            ci_lower,
            ci_upper,
            std_dev,
            sample_count: n,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ci_with_5_samples() {
        let scores = vec![0.8, 0.85, 0.9, 0.82, 0.88];
        let result = MetricResult::from_scores(scores, 0.7);

        assert!(result.ci_lower.is_some());
        assert!(result.ci_upper.is_some());
        assert!(result.std_dev.is_some());

        let ci_lower = result.ci_lower.unwrap();
        let ci_upper = result.ci_upper.unwrap();
        assert!(ci_lower < result.mean, "ci_lower {} should be < mean {}", ci_lower, result.mean);
        assert!(ci_upper > result.mean, "ci_upper {} should be > mean {}", ci_upper, result.mean);
    }

    #[test]
    fn test_ci_single_sample_is_none() {
        let scores = vec![0.85];
        let result = MetricResult::from_scores(scores, 0.7);

        assert!(result.ci_lower.is_none());
        assert!(result.ci_upper.is_none());
        assert!(result.std_dev.is_none());
        assert_eq!(result.sample_count, 1);
    }

    #[test]
    fn test_ci_zero_samples() {
        let scores: Vec<f64> = vec![];
        let result = MetricResult::from_scores(scores, 0.7);

        assert!(result.ci_lower.is_none());
        assert!(result.ci_upper.is_none());
        assert!(result.std_dev.is_none());
        assert_eq!(result.sample_count, 0);
        assert_eq!(result.mean, 0.0);
    }

    #[test]
    fn test_sample_count_correct() {
        let scores = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let result = MetricResult::from_scores(scores, 0.0);
        assert_eq!(result.sample_count, 5);
    }

    #[test]
    fn test_std_dev_correct() {
        // Known values: [2, 4, 4, 4, 5, 5, 7, 9]
        // Mean = 5.0, Sample std dev = sqrt(32/7) ≈ 2.13809
        let scores = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let result = MetricResult::from_scores(scores, 0.0);

        let std_dev = result.std_dev.unwrap();
        let expected = (32.0_f64 / 7.0).sqrt();
        assert!(
            (std_dev - expected).abs() < 1e-10,
            "std_dev {} should be close to {}",
            std_dev,
            expected
        );
    }
}
