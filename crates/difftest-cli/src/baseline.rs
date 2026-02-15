use std::collections::HashMap;
use std::time::Instant;

use clap::{Args, Subcommand};
use pyo3::prelude::*;
use pyo3::types::PyList;

use difftest_core::suite::TestType;

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
    #[arg(long, default_value = "")]
    model: String,

    /// Device to run on (cuda:0, mps, cpu)
    #[arg(long, default_value = "cpu")]
    device: String,

    /// Directory containing test_*.py files
    #[arg(long, default_value = "tests/")]
    test_dir: String,

    /// Directory to store baseline images
    #[arg(long, default_value = "baselines/")]
    baseline_dir: String,

    /// Generator backend: diffusers, comfyui, api
    #[arg(long, default_value = "diffusers")]
    generator: String,

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
}

fn build_generator_config(args: &BaselineSaveArgs) -> HashMap<String, String> {
    let mut config = HashMap::new();
    if !args.model.is_empty() {
        config.insert("model_id".to_string(), args.model.clone());
    }
    config.insert("device".to_string(), args.device.clone());
    if let Some(ref url) = args.comfyui_url {
        config.insert("comfyui_url".to_string(), url.clone());
    }
    if let Some(ref wf) = args.workflow {
        config.insert("workflow_path".to_string(), wf.clone());
    }
    if let Some(ref provider) = args.provider {
        config.insert("provider".to_string(), provider.clone());
    }
    if let Some(ref key) = args.api_key {
        config.insert("api_key".to_string(), key.clone());
    }
    if let Some(ref ep) = args.endpoint {
        config.insert("endpoint".to_string(), ep.clone());
    }
    config
}

pub fn execute(args: BaselineArgs) -> Result<(), Box<dyn std::error::Error>> {
    let save_args = match args.command {
        BaselineCommand::Save(a) | BaselineCommand::Update(a) => a,
    };

    let generator_name = save_args.generator.clone();
    let generator_config = build_generator_config(&save_args);

    Python::attach(|py| -> PyResult<()> {
        let sys = py.import("sys")?;
        let py_path: Bound<'_, PyList> = sys.getattr("path")?.cast_into()?;
        py_path.insert(0, "python")?;

        // Discover tests — only visual_regression tests need baselines
        println!("Discovering visual regression tests in {}...", save_args.test_dir);
        let suite = crate::bridge::discover_and_build_suite(
            py,
            &save_args.test_dir,
            &save_args.model,
            &save_args.device,
            ".difftest/outputs",
            &generator_name,
            &generator_config,
        )?;

        let vr_tests: Vec<_> = suite
            .tests
            .iter()
            .filter(|t| t.test_type == TestType::VisualRegression)
            .collect();

        if vr_tests.is_empty() {
            println!("No visual regression tests found.");
            return Ok(());
        }
        println!("Found {} visual regression test(s)", vr_tests.len());

        // Initialize generator
        if generator_name == "diffusers" {
            println!("Loading model {}...", save_args.model);
        } else {
            println!("Initializing {} generator...", generator_name);
        }
        let runner = crate::bridge::PyTestRunner::new(py, &generator_name, &generator_config, &[])?;

        for test_case in &vr_tests {
            let start = Instant::now();
            println!("Generating baselines for {}...", test_case.name);

            let baseline_dir = test_case
                .baseline_dir
                .as_deref()
                .unwrap_or(&save_args.baseline_dir);

            let mut images: Vec<(String, String, u64)> = Vec::new();

            for prompt in &test_case.prompts {
                for &seed in &test_case.seeds {
                    let image_path = runner.generate_image(
                        py,
                        prompt,
                        seed,
                        suite.config.output_dir.to_str().unwrap_or(".difftest/outputs"),
                    )?;
                    images.push((image_path, prompt.clone(), seed));
                }
            }

            let saved = runner.save_baselines(py, &test_case.name, &images, baseline_dir)?;
            let elapsed = start.elapsed().as_millis();
            println!(
                "  Saved {} baseline image(s) to {}/{} ({}ms)",
                saved.len(),
                baseline_dir,
                test_case.name,
                elapsed
            );
        }

        println!("Baselines saved successfully.");
        Ok(())
    })?;

    Ok(())
}
