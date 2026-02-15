use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use clap::Args;
use pyo3::prelude::*;
use pyo3::types::PyList;

use difftest_core::report;
use difftest_core::runner::{GeneratedImage, MetricResult, SuiteResult, TestResult};
use difftest_core::suite::{MetricCategory, TestType};

#[derive(Args)]
pub struct RunArgs {
    /// HuggingFace model ID or local path
    #[arg(long)]
    model: String,

    /// Device to run on (cuda:0, mps, cpu)
    #[arg(long, default_value = "cpu")]
    device: String,

    /// Directory containing test_*.py files
    #[arg(long, default_value = "tests/")]
    test_dir: String,

    /// Write JSON results to this path
    #[arg(long)]
    output: Option<String>,

    /// Write HTML report to this path
    #[arg(long)]
    html: Option<String>,

    /// Write JUnit XML report to this path
    #[arg(long)]
    junit: Option<String>,

    /// Write Markdown summary to this path
    #[arg(long)]
    markdown: Option<String>,
}

pub fn execute(args: RunArgs) -> Result<(), Box<dyn std::error::Error>> {
    let suite_start = Instant::now();

    let html_path = args.html.clone();
    let junit_path = args.junit.clone();
    let markdown_path = args.markdown.clone();
    let model_id = args.model.clone();
    let device = args.device.clone();

    let suite_result: SuiteResult = Python::attach(|py| -> PyResult<SuiteResult> {
        // Add the python/ directory to Python path so `difftest` package is importable
        let sys = py.import("sys")?;
        let py_path: Bound<'_, PyList> = sys.getattr("path")?.cast_into()?;
        py_path.insert(0, "python")?;

        // Discover tests
        println!("Discovering tests in {}...", args.test_dir);
        let suite = crate::bridge::discover_and_build_suite(
            py,
            &args.test_dir,
            &args.model,
            &args.device,
            ".difftest/outputs",
        )?;

        if suite.tests.is_empty() {
            println!("No tests found.");
            return Ok(SuiteResult::from_results(vec![], 0));
        }
        println!("Found {} test(s)", suite.tests.len());

        // Collect unique metric names for initialization
        let mut required_metrics: Vec<String> = suite
            .tests
            .iter()
            .flat_map(|t| t.metrics.iter().map(|m| m.name.clone()))
            .collect();
        required_metrics.sort();
        required_metrics.dedup();

        // Initialize generator and metrics
        println!("Loading model {}...", args.model);
        let runner =
            crate::bridge::PyTestRunner::new(py, &args.model, &args.device, &required_metrics)?;

        // Run each test
        let mut results = Vec::new();
        for test_case in &suite.tests {
            let test_start = Instant::now();
            print!("Running {}... ", test_case.name);

            let test_result = match test_case.test_type {
                TestType::VisualRegression => {
                    run_visual_regression(py, &runner, test_case, &suite)?
                }
                TestType::Quality => {
                    run_quality_test(py, &runner, test_case, &suite)?
                }
            };

            let test_duration = test_start.elapsed().as_millis() as u64;
            let test_result = TestResult {
                duration_ms: test_duration,
                ..test_result
            };

            if test_result.passed {
                println!("\x1b[32m✓ PASSED\x1b[0m");
            } else {
                println!("\x1b[31m✗ FAILED\x1b[0m");
            }

            results.push(test_result);
        }

        let suite_duration = suite_start.elapsed().as_millis() as u64;
        Ok(SuiteResult::from_results(results, suite_duration))
    })?;

    // Console report
    report::generate_console_report(&suite_result);

    // JSON report
    if let Some(output_path) = &args.output {
        report::generate_json_report(&suite_result, Path::new(output_path))?;
        println!("JSON report written to {output_path}");
    }

    // HTML report
    if let Some(html_path) = &html_path {
        difftest_core::html_report::generate_html_report(&suite_result, Path::new(html_path))?;
        println!("HTML report written to {html_path}");
    }

    // JUnit XML report
    if let Some(junit_path) = &junit_path {
        difftest_core::junit::generate_junit_xml(&suite_result, Path::new(junit_path))?;
        println!("JUnit XML report written to {junit_path}");
    }

    // Markdown report
    if let Some(markdown_path) = &markdown_path {
        difftest_core::markdown::generate_markdown_report(
            &suite_result,
            Path::new(markdown_path),
        )?;
        println!("Markdown report written to {markdown_path}");
    }

    // SQLite storage
    let db_dir = Path::new(".difftest");
    if !db_dir.exists() {
        std::fs::create_dir_all(db_dir)?;
    }
    let storage = difftest_core::storage::Storage::new(&db_dir.join("results.db"))?;
    let run_id = storage.save_run(&model_id, &device, &suite_result)?;
    println!("Results saved to .difftest/results.db (run #{run_id})");

    // Exit code
    if suite_result.total_failed > 0 {
        std::process::exit(1);
    }

    Ok(())
}

fn run_quality_test(
    py: Python<'_>,
    runner: &crate::bridge::PyTestRunner,
    test_case: &difftest_core::suite::TestCase,
    suite: &difftest_core::suite::TestSuite,
) -> PyResult<TestResult> {
    let mut all_images = Vec::new();
    let mut metric_scores: HashMap<String, Vec<f64>> = HashMap::new();

    // Separate per-sample and batch metrics
    let per_sample_metrics: Vec<_> = test_case
        .metrics
        .iter()
        .filter(|m| m.category == MetricCategory::PerSample)
        .collect();
    let batch_metrics: Vec<_> = test_case
        .metrics
        .iter()
        .filter(|m| m.category == MetricCategory::Batch)
        .collect();

    for metric_spec in &test_case.metrics {
        metric_scores.insert(metric_spec.name.clone(), Vec::new());
    }

    for prompt in &test_case.prompts {
        for &seed in &test_case.seeds {
            let image_path = runner.generate_image(
                py,
                prompt,
                seed,
                suite.config.output_dir.to_str().unwrap_or(".difftest/outputs"),
            )?;

            all_images.push(GeneratedImage {
                path: image_path.clone(),
                prompt: prompt.clone(),
                seed,
            });

            // Compute per-sample metrics
            for metric_spec in &per_sample_metrics {
                let score = runner.compute_metric(
                    py,
                    &metric_spec.name,
                    &image_path,
                    Some(prompt.as_str()),
                    None,
                )?;
                metric_scores
                    .get_mut(&metric_spec.name)
                    .unwrap()
                    .push(score);
            }
        }
    }

    // Compute batch metrics (e.g. FID)
    for metric_spec in &batch_metrics {
        let generated_paths: Vec<String> = all_images.iter().map(|i| i.path.clone()).collect();

        // Collect reference paths from reference_dir
        let reference_paths: Vec<String> = if let Some(ref_dir) = &test_case.reference_dir {
            std::fs::read_dir(ref_dir)
                .map(|entries| {
                    entries
                        .filter_map(|e| e.ok())
                        .filter(|e| {
                            e.path()
                                .extension()
                                .map(|ext| ext == "png" || ext == "jpg" || ext == "jpeg")
                                .unwrap_or(false)
                        })
                        .map(|e| e.path().to_string_lossy().to_string())
                        .collect()
                })
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        let score =
            runner.compute_batch_metric(py, &metric_spec.name, &generated_paths, &reference_paths)?;
        metric_scores
            .get_mut(&metric_spec.name)
            .unwrap()
            .push(score);
    }

    let mut metric_results = HashMap::new();
    for metric_spec in &test_case.metrics {
        let scores = metric_scores.remove(&metric_spec.name).unwrap_or_default();
        let threshold = test_case
            .thresholds
            .get(&metric_spec.name)
            .copied()
            .unwrap_or(0.0);
        metric_results.insert(
            metric_spec.name.clone(),
            MetricResult::from_scores_with_direction(scores, threshold, &metric_spec.direction),
        );
    }

    Ok(TestResult::from_metrics(
        test_case,
        metric_results,
        all_images,
        0,
    ))
}

fn run_visual_regression(
    py: Python<'_>,
    runner: &crate::bridge::PyTestRunner,
    test_case: &difftest_core::suite::TestCase,
    suite: &difftest_core::suite::TestSuite,
) -> PyResult<TestResult> {
    let baseline_dir = test_case
        .baseline_dir
        .as_deref()
        .unwrap_or("baselines/");

    let mut all_images = Vec::new();
    let mut ssim_scores: Vec<f64> = Vec::new();
    let mut missing_baselines = false;

    for prompt in &test_case.prompts {
        for &seed in &test_case.seeds {
            // Generate current image
            let image_path = runner.generate_image(
                py,
                prompt,
                seed,
                suite.config.output_dir.to_str().unwrap_or(".difftest/outputs"),
            )?;

            all_images.push(GeneratedImage {
                path: image_path.clone(),
                prompt: prompt.clone(),
                seed,
            });

            // Look up baseline
            match runner.load_baseline_path(py, &test_case.name, prompt, seed, baseline_dir)? {
                Some(baseline_path) => {
                    let score = runner.compute_metric(
                        py,
                        "ssim",
                        &image_path,
                        Some(prompt.as_str()),
                        Some(&baseline_path),
                    )?;
                    ssim_scores.push(score);
                }
                None => {
                    eprintln!(
                        "  Warning: no baseline for {} (prompt='{}', seed={}). Run `difftest baseline save` first.",
                        test_case.name, prompt, seed
                    );
                    missing_baselines = true;
                    ssim_scores.push(0.0);
                }
            }
        }
    }

    let threshold = test_case
        .thresholds
        .get("ssim")
        .copied()
        .unwrap_or(0.85);

    let mut metric_result = MetricResult::from_scores(ssim_scores, threshold);
    if missing_baselines {
        metric_result.passed = false;
    }

    let mut metric_results = HashMap::new();
    metric_results.insert("ssim".to_string(), metric_result);

    Ok(TestResult::from_metrics(
        test_case,
        metric_results,
        all_images,
        0,
    ))
}
