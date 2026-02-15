use std::fmt::Write as FmtWrite;
use std::path::Path;

use clap::Args;

use difftest_core::compare::{
    cohens_d, significance_level, welch_t_test, ComparisonResult, MetricComparison, RunInfo,
};
use difftest_core::error::DifftestError;
use difftest_core::storage::Storage;

#[derive(Args)]
pub struct CompareArgs {
    /// First run ID to compare
    #[arg(long)]
    run_a: i64,

    /// Second run ID to compare
    #[arg(long)]
    run_b: i64,

    /// Write JSON comparison results to this path
    #[arg(long)]
    output: Option<String>,

    /// Write Markdown comparison report to this path
    #[arg(long)]
    markdown: Option<String>,
}

pub fn execute(args: CompareArgs) -> difftest_core::error::Result<()> {
    let db_path = Path::new(".difftest/results.db");
    if !db_path.exists() {
        return Err(DifftestError::Generation(
            "No results database found at .difftest/results.db. Run `difftest run` first."
                .to_string(),
        ));
    }

    let storage = Storage::new(db_path)?;

    let run_a = storage
        .get_run(args.run_a)?
        .ok_or_else(|| DifftestError::Generation(format!("Run #{} not found", args.run_a)))?;
    let run_b = storage
        .get_run(args.run_b)?
        .ok_or_else(|| DifftestError::Generation(format!("Run #{} not found", args.run_b)))?;

    let samples_a = storage.get_run_samples(args.run_a)?;
    let samples_b = storage.get_run_samples(args.run_b)?;

    // Find shared (test_name, metric_name) pairs
    let mut comparisons = Vec::new();
    for (key, values_a) in &samples_a {
        if let Some(values_b) = samples_b.get(key) {
            let (test_name, metric_name) = key;

            let mean_a = values_a.iter().sum::<f64>() / values_a.len() as f64;
            let mean_b = values_b.iter().sum::<f64>() / values_b.len() as f64;

            let (t_stat, df) = welch_t_test(values_a, values_b);
            let effect = cohens_d(values_a, values_b);
            let (p_sig, significant) = significance_level(t_stat, df);

            let winner = if significant {
                if mean_a > mean_b {
                    Some("A".to_string())
                } else {
                    Some("B".to_string())
                }
            } else {
                None
            };

            comparisons.push(MetricComparison {
                test_name: test_name.clone(),
                metric_name: metric_name.clone(),
                mean_a,
                mean_b,
                diff: mean_a - mean_b,
                effect_size: effect,
                t_statistic: t_stat,
                p_significance: p_sig,
                significant,
                winner,
            });
        }
    }

    // Sort by test_name then metric_name for consistent output
    comparisons.sort_by(|a, b| {
        a.test_name
            .cmp(&b.test_name)
            .then(a.metric_name.cmp(&b.metric_name))
    });

    let result = ComparisonResult {
        run_a: RunInfo::from(&run_a),
        run_b: RunInfo::from(&run_b),
        comparisons: comparisons.clone(),
    };

    // Console output
    println!(
        "\nComparing Run #{} ({}) vs Run #{} ({})\n",
        run_a.id, run_a.model_name, run_b.id, run_b.model_name
    );

    let mut current_test = String::new();
    for comp in &comparisons {
        if comp.test_name != current_test {
            current_test = comp.test_name.clone();
            println!("{}", current_test);
        }

        let sig_marker = match comp.p_significance.as_str() {
            "p<0.001" => "***",
            "p<0.01" => "**",
            "p<0.05" => "*",
            _ => "",
        };

        let winner_str = match &comp.winner {
            Some(w) => format!("\u{2192} {} wins", w),
            None => "\u{2192} No significant difference".to_string(),
        };

        println!(
            "  {:<14} {:.3} vs {:.3}  (d={:.2}, {}{})   {}",
            format!("{}:", comp.metric_name),
            comp.mean_a,
            comp.mean_b,
            comp.effect_size,
            comp.p_significance,
            sig_marker,
            winner_str,
        );
    }

    println!("\nLegend: * p<0.05  ** p<0.01  *** p<0.001");

    // JSON output
    if let Some(ref output_path) = args.output {
        let json = serde_json::to_string_pretty(&result)
            .map_err(|e| DifftestError::Generation(e.to_string()))?;
        std::fs::write(output_path, json)?;
        println!("\nJSON comparison written to {output_path}");
    }

    // Markdown output
    if let Some(ref md_path) = args.markdown {
        let md = generate_markdown(&result);
        if let Some(parent) = Path::new(md_path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(md_path, md)?;
        println!("Markdown comparison written to {md_path}");
    }

    Ok(())
}

fn generate_markdown(result: &ComparisonResult) -> String {
    let mut md = String::new();

    let _ = writeln!(
        md,
        "# A/B Comparison: Run #{} vs Run #{}\n",
        result.run_a.id, result.run_b.id
    );
    let _ = writeln!(
        md,
        "- **Run A**: {} (#{}, {})",
        result.run_a.model_name, result.run_a.id, result.run_a.timestamp
    );
    let _ = writeln!(
        md,
        "- **Run B**: {} (#{}, {})\n",
        result.run_b.model_name, result.run_b.id, result.run_b.timestamp
    );

    md.push_str("| Test | Metric | Mean A | Mean B | Cohen's d | Significance | Winner |\n");
    md.push_str("|------|--------|--------|--------|-----------|--------------|--------|\n");

    for comp in &result.comparisons {
        let winner = match &comp.winner {
            Some(w) => w.as_str(),
            None => "\u{2014}",
        };
        let _ = writeln!(
            md,
            "| {} | {} | {:.4} | {:.4} | {:.2} | {} | {} |",
            comp.test_name,
            comp.metric_name,
            comp.mean_a,
            comp.mean_b,
            comp.effect_size,
            comp.p_significance,
            winner,
        );
    }

    md.push_str("\n---\n*Generated by [difftest](https://github.com/SouKangC/difftest)*\n");
    md
}
