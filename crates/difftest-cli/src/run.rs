use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use clap::Args;
use pyo3::prelude::*;
use pyo3::types::PyList;

use difftest_core::error::DifftestError;
use difftest_core::report;
use difftest_core::runner::{GeneratedImage, MetricResult, SuiteResult, TestResult};
use difftest_core::suite::{MetricCategory, TestType};

use crate::config::{self, DifftestConfig};

#[derive(Args)]
pub struct RunArgs {
    /// HuggingFace model ID or local path (required for diffusers generator)
    #[arg(long)]
    model: Option<String>,

    /// Device to run on (cuda:0, mps, cpu)
    #[arg(long)]
    device: Option<String>,

    /// Directory containing test_*.py files
    #[arg(long)]
    test_dir: Option<String>,

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

    /// Generator backend: diffusers, comfyui, api
    #[arg(long)]
    generator: Option<String>,

    /// ComfyUI server URL (for --generator comfyui)
    #[arg(long)]
    comfyui_url: Option<String>,

    /// ComfyUI workflow JSON file path (for --generator comfyui)
    #[arg(long)]
    workflow: Option<String>,

    /// API provider: fal, replicate, custom (for --generator api)
    #[arg(long)]
    provider: Option<String>,

    /// API key (for --generator api; falls back to DIFFTEST_API_KEY env var)
    #[arg(long)]
    api_key: Option<String>,

    /// API endpoint URL (for --generator api with custom provider)
    #[arg(long)]
    endpoint: Option<String>,

    /// Substring filter on test names
    #[arg(long)]
    filter: Option<String>,

    /// Run specific test(s) by exact name (repeatable)
    #[arg(long = "test", value_name = "NAME")]
    test_names: Vec<String>,

    /// List matched tests without executing them
    #[arg(long)]
    dry_run: bool,

    /// Suite-level timeout in seconds
    #[arg(long)]
    timeout: Option<u64>,

    /// Per-image generation timeout in seconds
    #[arg(long)]
    image_timeout: Option<u64>,

    /// Reuse cached images when inputs haven't changed
    #[arg(long)]
    incremental: bool,

    /// Minimum number of samples required for a metric to pass
    #[arg(long)]
    min_samples: Option<usize>,
}

pub fn execute(args: RunArgs, cfg: &DifftestConfig) -> difftest_core::error::Result<()> {
    let suite_start = Instant::now();

    // Resolve CLI > config > default
    let model = config::resolve_string(&args.model, &cfg.model, "");
    let device = config::resolve_string(&args.device, &cfg.device, "cpu");
    let test_dir = config::resolve_string(&args.test_dir, &cfg.test_dir, "tests/");
    let generator_name = config::resolve_string(&args.generator, &cfg.generator, "diffusers");
    let output = config::resolve_option(&args.output, &cfg.output);
    let html_path = config::resolve_option(&args.html, &cfg.html);
    let junit_path = config::resolve_option(&args.junit, &cfg.junit);
    let markdown_path = config::resolve_option(&args.markdown, &cfg.markdown);
    let comfyui_url = config::resolve_option(&args.comfyui_url, &cfg.comfyui_url);
    let workflow = config::resolve_option(&args.workflow, &cfg.workflow);
    let provider = config::resolve_option(&args.provider, &cfg.provider);
    let api_key = config::resolve_option(&args.api_key, &cfg.api_key);
    let endpoint = config::resolve_option(&args.endpoint, &cfg.endpoint);
    let timeout = args.timeout.or(cfg.timeout);
    let image_timeout = args.image_timeout.or(cfg.image_timeout);
    let incremental = config::resolve_bool(args.incremental, &cfg.incremental);
    let min_samples = args.min_samples.or(cfg.min_samples);
    let filter = config::resolve_option(&args.filter, &cfg.filter);
    let mut test_names = args.test_names.clone();
    if test_names.is_empty() {
        if let Some(ref cfg_tests) = cfg.test {
            test_names = cfg_tests.clone();
        }
    }
    let generator_config = config::build_generator_config(
        &model,
        &device,
        &comfyui_url,
        &workflow,
        &provider,
        &api_key,
        &endpoint,
        &cfg.retry,
        &image_timeout,
    );

    // Incremental cache
    let mut cache = if incremental {
        Some(crate::cache::CacheManifest::load(Path::new(".difftest/cache.json")))
    } else {
        None
    };

    let suite_result: SuiteResult = Python::attach(|py| -> PyResult<SuiteResult> {
        // Add the python/ directory to Python path so `difftest` package is importable
        let sys = py.import("sys")?;
        let py_path: Bound<'_, PyList> = sys.getattr("path")?.cast_into()?;
        py_path.insert(0, "python")?;

        // Discover tests
        println!("Discovering tests in {}...", test_dir);
        let suite = crate::bridge::discover_and_build_suite(
            py,
            &test_dir,
            &model,
            &device,
            ".difftest/outputs",
            &generator_name,
            &generator_config,
        )?;

        // Filter tests
        let filtered_tests = filter_tests(suite.tests, filter.as_deref(), &test_names);

        if filtered_tests.is_empty() {
            println!("No tests found.");
            return Ok(SuiteResult::from_results(vec![], 0));
        }
        println!("Found {} test(s)", filtered_tests.len());

        // Dry-run: list tests and exit
        if args.dry_run {
            println!("\nDry-run mode — tests that would run:\n");
            for t in &filtered_tests {
                let test_type = match t.test_type {
                    TestType::Quality => "quality",
                    TestType::VisualRegression => "visual_regression",
                };
                let metrics: Vec<&str> = t.metrics.iter().map(|m| m.name.as_str()).collect();
                println!(
                    "  {} (type={}, prompts={}, seeds={}, metrics=[{}])",
                    t.name,
                    test_type,
                    t.prompts.len(),
                    t.seeds.len(),
                    metrics.join(", "),
                );
            }
            println!();
            return Ok(SuiteResult::from_results(vec![], 0));
        }

        // Rebuild suite with filtered tests
        let suite = difftest_core::suite::TestSuite {
            tests: filtered_tests,
            config: suite.config,
        };

        // Collect unique metric names for initialization
        let mut required_metrics: Vec<String> = suite
            .tests
            .iter()
            .flat_map(|t| t.metrics.iter().map(|m| m.name.clone()))
            .collect();
        required_metrics.sort();
        required_metrics.dedup();

        // Initialize generator and metrics
        if generator_name == "diffusers" {
            println!("Loading model {}...", model);
        } else {
            println!("Initializing {} generator...", generator_name);
        }
        let runner =
            crate::bridge::PyTestRunner::new(py, &generator_name, &generator_config, &required_metrics)?;

        // Run each test
        let mut results = Vec::new();
        for test_case in &suite.tests {
            // Suite timeout check
            if let Some(timeout_secs) = timeout {
                if suite_start.elapsed().as_secs() >= timeout_secs {
                    println!(
                        "\x1b[33m⚠ Suite timeout ({timeout_secs}s) exceeded, skipping remaining tests\x1b[0m"
                    );
                    break;
                }
            }

            let test_start = Instant::now();
            print!("Running {}... ", test_case.name);

            let test_result = match test_case.test_type {
                TestType::VisualRegression => {
                    run_visual_regression(py, &runner, test_case, &suite, &mut cache, min_samples)?
                }
                TestType::Quality => {
                    run_quality_test(py, &runner, test_case, &suite, &mut cache, min_samples)?
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
    }).map_err(|e| DifftestError::Generation(e.to_string()))?;

    // Save cache manifest if incremental
    if let Some(ref cache) = cache {
        if let Err(e) = cache.save(Path::new(".difftest/cache.json")) {
            eprintln!("Warning: failed to save cache: {e}");
        }
    }

    // Dry-run exits early (no reports)
    if args.dry_run {
        return Ok(());
    }

    // Console report
    report::generate_console_report(&suite_result);

    // JSON report
    if let Some(ref output_path) = output {
        report::generate_json_report(&suite_result, Path::new(output_path))?;
        println!("JSON report written to {output_path}");
    }

    // HTML report
    if let Some(ref html_path) = html_path {
        difftest_core::html_report::generate_html_report(&suite_result, Path::new(html_path))?;
        println!("HTML report written to {html_path}");
    }

    // JUnit XML report
    if let Some(ref junit_path) = junit_path {
        difftest_core::junit::generate_junit_xml(&suite_result, Path::new(junit_path))?;
        println!("JUnit XML report written to {junit_path}");
    }

    // Markdown report
    if let Some(ref markdown_path) = markdown_path {
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
    let run_id = storage.save_run(&model, &device, &suite_result)?;
    println!("Results saved to .difftest/results.db (run #{run_id})");

    // Exit code
    if suite_result.total_failed > 0 {
        std::process::exit(1);
    }

    Ok(())
}

fn filter_tests(
    tests: Vec<difftest_core::suite::TestCase>,
    filter: Option<&str>,
    names: &[String],
) -> Vec<difftest_core::suite::TestCase> {
    if filter.is_none() && names.is_empty() {
        return tests;
    }
    tests
        .into_iter()
        .filter(|t| {
            if !names.is_empty() && names.contains(&t.name) {
                return true;
            }
            if let Some(f) = filter {
                return t.name.contains(f);
            }
            false
        })
        .collect()
}

fn run_quality_test(
    py: Python<'_>,
    runner: &crate::bridge::PyTestRunner,
    test_case: &difftest_core::suite::TestCase,
    suite: &difftest_core::suite::TestSuite,
    cache: &mut Option<crate::cache::CacheManifest>,
    min_samples: Option<usize>,
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
            let image_path = if let Some(ref mut manifest) = cache {
                let key = crate::cache::CacheManifest::cache_key(
                    &suite.config.model_id,
                    prompt,
                    seed,
                    &suite.config.generator,
                );
                if let Some(entry) = manifest.lookup(&key) {
                    if Path::new(&entry.image_path).exists() {
                        entry.image_path.clone()
                    } else {
                        let path = runner.generate_image(
                            py,
                            prompt,
                            seed,
                            suite.config.output_dir.to_str().unwrap_or(".difftest/outputs"),
                            test_case.negative_prompt.as_deref(),
                        )?;
                        manifest.insert(
                            key,
                            crate::cache::CacheEntry {
                                image_path: path.clone(),
                                model_id: suite.config.model_id.clone(),
                                prompt: prompt.clone(),
                                seed,
                                generator: suite.config.generator.clone(),
                                created_at: chrono::Utc::now().to_rfc3339(),
                            },
                        );
                        path
                    }
                } else {
                    let path = runner.generate_image(
                        py,
                        prompt,
                        seed,
                        suite.config.output_dir.to_str().unwrap_or(".difftest/outputs"),
                        test_case.negative_prompt.as_deref(),
                    )?;
                    manifest.insert(
                        key,
                        crate::cache::CacheEntry {
                            image_path: path.clone(),
                            model_id: suite.config.model_id.clone(),
                            prompt: prompt.clone(),
                            seed,
                            generator: suite.config.generator.clone(),
                            created_at: chrono::Utc::now().to_rfc3339(),
                        },
                    );
                    path
                }
            } else {
                runner.generate_image(
                    py,
                    prompt,
                    seed,
                    suite.config.output_dir.to_str().unwrap_or(".difftest/outputs"),
                    test_case.negative_prompt.as_deref(),
                )?
            };

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
        let mut mr = MetricResult::from_scores_with_direction(scores, threshold, &metric_spec.direction);
        if let Some(min) = min_samples {
            if mr.sample_count < min {
                mr.passed = false;
            }
        }
        metric_results.insert(metric_spec.name.clone(), mr);
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
    cache: &mut Option<crate::cache::CacheManifest>,
    min_samples: Option<usize>,
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
            // Generate current image (with cache support)
            let image_path = if let Some(ref mut manifest) = cache {
                let key = crate::cache::CacheManifest::cache_key(
                    &suite.config.model_id,
                    prompt,
                    seed,
                    &suite.config.generator,
                );
                if let Some(entry) = manifest.lookup(&key) {
                    if Path::new(&entry.image_path).exists() {
                        entry.image_path.clone()
                    } else {
                        let path = runner.generate_image(
                            py,
                            prompt,
                            seed,
                            suite.config.output_dir.to_str().unwrap_or(".difftest/outputs"),
                            test_case.negative_prompt.as_deref(),
                        )?;
                        manifest.insert(
                            key,
                            crate::cache::CacheEntry {
                                image_path: path.clone(),
                                model_id: suite.config.model_id.clone(),
                                prompt: prompt.clone(),
                                seed,
                                generator: suite.config.generator.clone(),
                                created_at: chrono::Utc::now().to_rfc3339(),
                            },
                        );
                        path
                    }
                } else {
                    let path = runner.generate_image(
                        py,
                        prompt,
                        seed,
                        suite.config.output_dir.to_str().unwrap_or(".difftest/outputs"),
                        test_case.negative_prompt.as_deref(),
                    )?;
                    manifest.insert(
                        key,
                        crate::cache::CacheEntry {
                            image_path: path.clone(),
                            model_id: suite.config.model_id.clone(),
                            prompt: prompt.clone(),
                            seed,
                            generator: suite.config.generator.clone(),
                            created_at: chrono::Utc::now().to_rfc3339(),
                        },
                    );
                    path
                }
            } else {
                runner.generate_image(
                    py,
                    prompt,
                    seed,
                    suite.config.output_dir.to_str().unwrap_or(".difftest/outputs"),
                    test_case.negative_prompt.as_deref(),
                )?
            };

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
    if let Some(min) = min_samples {
        if metric_result.sample_count < min {
            metric_result.passed = false;
        }
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
