use serde::{Deserialize, Serialize};

use crate::storage::LatestRun;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    pub run_a: RunInfo,
    pub run_b: RunInfo,
    pub comparisons: Vec<MetricComparison>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunInfo {
    pub id: i64,
    pub model_name: String,
    pub timestamp: String,
}

impl From<&LatestRun> for RunInfo {
    fn from(r: &LatestRun) -> Self {
        Self {
            id: r.id,
            model_name: r.model_name.clone(),
            timestamp: r.timestamp.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricComparison {
    pub test_name: String,
    pub metric_name: String,
    pub mean_a: f64,
    pub mean_b: f64,
    pub diff: f64,
    pub effect_size: f64,
    pub t_statistic: f64,
    pub p_significance: String,
    pub significant: bool,
    pub winner: Option<String>,
}

/// Compute Welch's t-test for two independent samples.
/// Returns (t_statistic, degrees_of_freedom).
pub fn welch_t_test(a: &[f64], b: &[f64]) -> (f64, f64) {
    let n_a = a.len() as f64;
    let n_b = b.len() as f64;

    if n_a < 2.0 || n_b < 2.0 {
        return (0.0, 0.0);
    }

    let mean_a = a.iter().sum::<f64>() / n_a;
    let mean_b = b.iter().sum::<f64>() / n_b;

    let var_a = a.iter().map(|x| (x - mean_a).powi(2)).sum::<f64>() / (n_a - 1.0);
    let var_b = b.iter().map(|x| (x - mean_b).powi(2)).sum::<f64>() / (n_b - 1.0);

    let se = (var_a / n_a + var_b / n_b).sqrt();
    if se == 0.0 {
        return (0.0, 0.0);
    }

    let t = (mean_a - mean_b) / se;

    // Welch-Satterthwaite degrees of freedom
    let num = (var_a / n_a + var_b / n_b).powi(2);
    let denom = (var_a / n_a).powi(2) / (n_a - 1.0) + (var_b / n_b).powi(2) / (n_b - 1.0);
    let df = if denom == 0.0 { 0.0 } else { num / denom };

    (t, df)
}

/// Compute Cohen's d effect size for two independent samples.
pub fn cohens_d(a: &[f64], b: &[f64]) -> f64 {
    let n_a = a.len() as f64;
    let n_b = b.len() as f64;

    if n_a < 2.0 || n_b < 2.0 {
        return 0.0;
    }

    let mean_a = a.iter().sum::<f64>() / n_a;
    let mean_b = b.iter().sum::<f64>() / n_b;

    let var_a = a.iter().map(|x| (x - mean_a).powi(2)).sum::<f64>() / (n_a - 1.0);
    let var_b = b.iter().map(|x| (x - mean_b).powi(2)).sum::<f64>() / (n_b - 1.0);

    // Pooled standard deviation
    let pooled_var = ((n_a - 1.0) * var_a + (n_b - 1.0) * var_b) / (n_a + n_b - 2.0);
    let pooled_sd = pooled_var.sqrt();

    if pooled_sd == 0.0 {
        return 0.0;
    }

    (mean_a - mean_b) / pooled_sd
}

/// Determine significance level from t-statistic and degrees of freedom.
/// Uses the same t-critical table approach as MetricResult CI computation.
pub fn significance_level(t_stat: f64, df: f64) -> (String, bool) {
    let abs_t = t_stat.abs();
    let df_int = df.round() as usize;

    if df_int == 0 {
        return ("n.s.".to_string(), false);
    }

    // t-critical values for common significance levels (two-tailed)
    // We check p<0.001, p<0.01, p<0.05 thresholds
    let (t_001, t_01, t_05) = match df_int {
        1 => (636.619, 63.657, 12.706),
        2 => (31.599, 9.925, 4.303),
        3 => (12.924, 5.841, 3.182),
        4 => (8.610, 4.604, 2.776),
        5 => (6.869, 4.032, 2.571),
        6 => (5.959, 3.707, 2.447),
        7 => (5.408, 3.499, 2.365),
        8 => (5.041, 3.355, 2.306),
        9 => (4.781, 3.250, 2.262),
        10 => (4.587, 3.169, 2.228),
        11..=15 => (4.221, 2.977, 2.131),
        16..=20 => (3.922, 2.845, 2.086),
        21..=30 => (3.646, 2.750, 2.042),
        31..=60 => (3.460, 2.660, 2.000),
        61..=120 => (3.373, 2.617, 1.980),
        _ => (3.291, 2.576, 1.960),
    };

    if abs_t >= t_001 {
        ("p<0.001".to_string(), true)
    } else if abs_t >= t_01 {
        ("p<0.01".to_string(), true)
    } else if abs_t >= t_05 {
        ("p<0.05".to_string(), true)
    } else {
        ("n.s.".to_string(), false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_welch_t_test_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (t, _df) = welch_t_test(&a, &b);
        assert!((t).abs() < 1e-10, "identical samples should have t ≈ 0");
    }

    #[test]
    fn test_welch_t_test_different() {
        let a = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (t, df) = welch_t_test(&a, &b);
        assert!(t > 0.0, "A has higher mean, t should be positive");
        assert!(df > 0.0, "df should be positive");
    }

    #[test]
    fn test_cohens_d_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let d = cohens_d(&a, &b);
        assert!((d).abs() < 1e-10, "identical samples should have d ≈ 0");
    }

    #[test]
    fn test_cohens_d_large_effect() {
        let a = vec![10.0, 11.0, 12.0];
        let b = vec![1.0, 2.0, 3.0];
        let d = cohens_d(&a, &b);
        assert!(d > 0.8, "large effect size expected, got {}", d);
    }

    #[test]
    fn test_significance_not_significant() {
        let (sig, is_sig) = significance_level(0.5, 10.0);
        assert_eq!(sig, "n.s.");
        assert!(!is_sig);
    }

    #[test]
    fn test_significance_p05() {
        // t=2.3 with df=10 should be p<0.05
        let (sig, is_sig) = significance_level(2.3, 10.0);
        assert_eq!(sig, "p<0.05");
        assert!(is_sig);
    }
}
