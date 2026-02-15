use std::time::Instant;

use clap::{Args, Subcommand};
use pyo3::prelude::*;
use pyo3::types::PyList;

use difftest_core::error::DifftestError;
use difftest_core::suite::TestType;

use crate::config::{self, DifftestConfig};

#[derive(Args)]
pub struct BaselineArgs {
    #[command(subcommand)]
    pub command: BaselineCommand,
}

#[derive(Subcommand)]
pub enum BaselineCommand {
    /// Save baseline images for visual regression tests
    Save(BaselineSaveArgs),
    /// Update existing baselines (alias for save, overwrites)
    Update(BaselineSaveArgs),
}

#[derive(Args, Clone)]
pub struct BaselineSaveArgs {
    /// HuggingFace model ID or local path (required for diffusers generator)
    #[arg(long)]
    model: Option<String>,

    /// Device to run on (cuda:0, mps, cpu)
    #[arg(long)]
    device: Option<String>,

    /// Directory containing test_*.py files
    #[arg(long)]
    test_dir: Option<String>,

    /// Directory to store baseline images
    #[arg(long)]
    baseline_dir: Option<String>,

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
}

pub fn execute(args: BaselineArgs, cfg: &DifftestConfig) -> difftest_core::error::Result<()> {
    let save_args = match args.command {
        BaselineCommand::Save(a) | BaselineCommand::Update(a) => a,
    };

    let model = config::resolve_string(&save_args.model, &cfg.model, "");
    let device = config::resolve_string(&save_args.device, &cfg.device, "cpu");
    let test_dir = config::resolve_string(&save_args.test_dir, &cfg.test_dir, "tests/");
    let baseline_dir = config::resolve_string(&save_args.baseline_dir, &cfg.baseline_dir, "baselines/");
    let generator_name = config::resolve_string(&save_args.generator, &cfg.generator, "diffusers");
    let comfyui_url = config::resolve_option(&save_args.comfyui_url, &cfg.comfyui_url);
    let workflow = config::resolve_option(&save_args.workflow, &cfg.workflow);
    let provider = config::resolve_option(&save_args.provider, &cfg.provider);
    let api_key = config::resolve_option(&save_args.api_key, &cfg.api_key);
    let endpoint = config::resolve_option(&save_args.endpoint, &cfg.endpoint);
    let filter = config::resolve_option(&save_args.filter, &cfg.filter);
    let mut test_names = save_args.test_names.clone();
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
        &cfg.image_timeout,
    );

    Python::attach(|py| -> PyResult<()> {
        let sys = py.import("sys")?;
        let py_path: Bound<'_, PyList> = sys.getattr("path")?.cast_into()?;
        py_path.insert(0, "python")?;

        // Discover tests — only visual_regression tests need baselines
        println!("Discovering visual regression tests in {}...", test_dir);
        let suite = crate::bridge::discover_and_build_suite(
            py,
            &test_dir,
            &model,
            &device,
            ".difftest/outputs",
            &generator_name,
            &generator_config,
        )?;

        let mut vr_tests: Vec<_> = suite
            .tests
            .iter()
            .filter(|t| t.test_type == TestType::VisualRegression)
            .collect();

        // Apply filtering
        if !test_names.is_empty() || filter.is_some() {
            vr_tests.retain(|t| {
                if !test_names.is_empty() && test_names.contains(&t.name) {
                    return true;
                }
                if let Some(ref f) = filter {
                    return t.name.contains(f.as_str());
                }
                false
            });
        }

        if vr_tests.is_empty() {
            println!("No visual regression tests found.");
            return Ok(());
        }
        println!("Found {} visual regression test(s)", vr_tests.len());

        // Initialize generator
        if generator_name == "diffusers" {
            println!("Loading model {}...", model);
        } else {
            println!("Initializing {} generator...", generator_name);
        }
        let runner = crate::bridge::PyTestRunner::new(py, &generator_name, &generator_config, &[])?;

        for test_case in &vr_tests {
            let start = Instant::now();
            println!("Generating baselines for {}...", test_case.name);

            let test_baseline_dir = test_case
                .baseline_dir
                .as_deref()
                .unwrap_or(&baseline_dir);

            let mut images: Vec<(String, String, u64)> = Vec::new();

            for prompt in &test_case.prompts {
                for &seed in &test_case.seeds {
                    let image_path = runner.generate_image(
                        py,
                        prompt,
                        seed,
                        suite.config.output_dir.to_str().unwrap_or(".difftest/outputs"),
                        test_case.negative_prompt.as_deref(),
                    )?;
                    images.push((image_path, prompt.clone(), seed));
                }
            }

            let saved = runner.save_baselines(py, &test_case.name, &images, test_baseline_dir)?;
            let elapsed = start.elapsed().as_millis();
            println!(
                "  Saved {} baseline image(s) to {}/{} ({}ms)",
                saved.len(),
                test_baseline_dir,
                test_case.name,
                elapsed
            );
        }

        println!("Baselines saved successfully.");
        Ok(())
    }).map_err(|e| DifftestError::Generation(e.to_string()))?;

    Ok(())
}
