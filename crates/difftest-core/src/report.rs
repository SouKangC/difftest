use crate::error;
use crate::runner::SuiteResult;
use std::path::Path;

pub fn generate_console_report(result: &SuiteResult) {
    println!();
    for test_result in &result.results {
        let status = if test_result.passed {
            "\x1b[32m✓ PASSED\x1b[0m"
        } else {
            "\x1b[31m✗ FAILED\x1b[0m"
        };

        let metrics_summary: Vec<String> = test_result
            .metrics
            .iter()
            .map(|(name, m)| {
                if let (Some(ci_lo), Some(ci_hi)) = (m.ci_lower, m.ci_upper) {
                    format!("{}={:.3} [{:.3}, {:.3}]", name, m.mean, ci_lo, ci_hi)
                } else {
                    format!("{}={:.3}", name, m.mean)
                }
            })
            .collect();

        println!(
            "  {} {} ({})",
            status,
            test_result.name,
            metrics_summary.join(", ")
        );
    }
    println!();
    println!(
        "Results: \x1b[32m{} passed\x1b[0m, \x1b[31m{} failed\x1b[0m ({}ms)",
        result.total_passed, result.total_failed, result.duration_ms
    );
}

pub fn generate_json_report(
    result: &SuiteResult,
    path: &Path,
) -> error::Result<()> {
    let json = serde_json::to_string_pretty(result)?;
    std::fs::write(path, json)?;
    Ok(())
}
