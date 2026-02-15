use clap::{Args, Subcommand};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

#[derive(Args)]
pub struct AgentArgs {
    #[command(subcommand)]
    pub command: AgentCommands,
}

#[derive(Subcommand)]
pub enum AgentCommands {
    /// Design a test suite using LLM
    Design {
        /// Description of the model to test
        #[arg(long)]
        model_description: String,

        /// Number of tests to generate
        #[arg(long, default_value = "5")]
        num_tests: u32,

        /// LLM provider: claude, openai, local
        #[arg(long, default_value = "claude")]
        llm_provider: String,

        /// LLM model override (provider-specific)
        #[arg(long)]
        llm_model: Option<String>,

        /// LLM API key (falls back to provider-specific env var)
        #[arg(long)]
        llm_api_key: Option<String>,

        /// Output directory for generated test file
        #[arg(long, default_value = ".")]
        output_dir: String,
    },
    /// Diagnose failures from the latest test run
    Diagnose {
        /// Test name to diagnose (omit for all failed tests)
        #[arg(long)]
        test_name: Option<String>,

        /// LLM provider: claude, openai, local
        #[arg(long, default_value = "claude")]
        llm_provider: String,

        /// LLM model override
        #[arg(long)]
        llm_model: Option<String>,

        /// LLM API key
        #[arg(long)]
        llm_api_key: Option<String>,

        /// Path to the difftest SQLite database
        #[arg(long, default_value = ".difftest/results.db")]
        db_path: String,
    },
    /// Track metric regressions over time
    Track {
        /// Test name to track (omit for all tests)
        #[arg(long)]
        test_name: Option<String>,

        /// Number of historical runs to analyze
        #[arg(long, default_value = "20")]
        limit: u32,

        /// LLM provider: claude, openai, local
        #[arg(long, default_value = "claude")]
        llm_provider: String,

        /// LLM model override
        #[arg(long)]
        llm_model: Option<String>,

        /// LLM API key
        #[arg(long)]
        llm_api_key: Option<String>,

        /// Path to the difftest SQLite database
        #[arg(long, default_value = ".difftest/results.db")]
        db_path: String,
    },
}

pub fn execute(args: AgentArgs) -> Result<(), Box<dyn std::error::Error>> {
    match args.command {
        AgentCommands::Design {
            model_description,
            num_tests,
            llm_provider,
            llm_model,
            llm_api_key,
            output_dir,
        } => execute_design(
            &model_description,
            num_tests,
            &llm_provider,
            &llm_model,
            &llm_api_key,
            &output_dir,
        ),
        AgentCommands::Diagnose {
            test_name,
            llm_provider,
            llm_model,
            llm_api_key,
            db_path,
        } => execute_diagnose(
            test_name.as_deref(),
            &llm_provider,
            &llm_model,
            &llm_api_key,
            &db_path,
        ),
        AgentCommands::Track {
            test_name,
            limit,
            llm_provider,
            llm_model,
            llm_api_key,
            db_path,
        } => execute_track(
            test_name.as_deref(),
            limit,
            &llm_provider,
            &llm_model,
            &llm_api_key,
            &db_path,
        ),
    }
}

fn execute_design(
    model_description: &str,
    num_tests: u32,
    llm_provider: &str,
    llm_model: &Option<String>,
    llm_api_key: &Option<String>,
    output_dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Designing test suite for: {model_description}");
    println!("Using LLM provider: {llm_provider}");

    Python::attach(|py| -> PyResult<()> {
        let sys = py.import("sys")?;
        let py_path: Bound<'_, PyList> = sys.getattr("path")?.cast_into()?;
        py_path.insert(0, "python")?;

        let bridge = py.import("difftest.agent.bridge")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("provider_name", llm_provider)?;
        kwargs.set_item("model_description", model_description)?;
        kwargs.set_item("num_tests", num_tests)?;
        kwargs.set_item("output_dir", output_dir)?;
        if let Some(ref key) = llm_api_key {
            kwargs.set_item("api_key", key)?;
        }
        if let Some(ref m) = llm_model {
            kwargs.set_item("model", m)?;
        }

        let results = bridge.call_method("bridge_design", (), Some(&kwargs))?;
        let result_list: Bound<'_, PyList> = results.cast_into()?;

        if result_list.is_empty() {
            println!("No tests designed.");
            return Ok(());
        }

        // Print output path from first result
        let first = result_list.get_item(0)?;
        let output_path: String = first.get_item("output_path")?.extract()?;
        println!("Generated test file: {output_path}");

        println!("\nDesigned {} test(s):", result_list.len());
        for (i, item) in result_list.iter().enumerate() {
            let name: String = item.get_item("name")?.extract()?;
            let rationale: String = item.get_item("rationale")?.extract()?;
            println!("  {}. {} — {}", i + 1, name, rationale);
        }

        Ok(())
    })?;

    Ok(())
}

fn execute_diagnose(
    test_name: Option<&str>,
    llm_provider: &str,
    llm_model: &Option<String>,
    llm_api_key: &Option<String>,
    db_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Diagnosing test failures...");
    println!("Using LLM provider: {llm_provider}");

    Python::attach(|py| -> PyResult<()> {
        let sys = py.import("sys")?;
        let py_path: Bound<'_, PyList> = sys.getattr("path")?.cast_into()?;
        py_path.insert(0, "python")?;

        let bridge = py.import("difftest.agent.bridge")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("provider_name", llm_provider)?;
        kwargs.set_item("db_path", db_path)?;
        if let Some(name) = test_name {
            kwargs.set_item("test_name", name)?;
        }
        if let Some(ref key) = llm_api_key {
            kwargs.set_item("api_key", key)?;
        }
        if let Some(ref m) = llm_model {
            kwargs.set_item("model", m)?;
        }

        let results = bridge.call_method("bridge_diagnose", (), Some(&kwargs))?;
        let result_list: Bound<'_, PyList> = results.cast_into()?;

        if result_list.is_empty() {
            println!("No failures found in the latest run.");
            return Ok(());
        }

        for item in result_list.iter() {
            let tname: String = item.get_item("test_name")?.extract()?;
            let summary: String = item.get_item("summary")?.extract()?;
            let root_cause: String = item.get_item("root_cause")?.extract()?;
            let severity: String = item.get_item("severity")?.extract()?;
            let suggestions: Vec<String> = item.get_item("suggestions")?.extract()?;

            println!("\n--- {} [{}] ---", tname, severity.to_uppercase());
            println!("Summary: {summary}");
            println!("Root cause: {root_cause}");
            if !suggestions.is_empty() {
                println!("Suggestions:");
                for s in &suggestions {
                    println!("  - {s}");
                }
            }
        }

        Ok(())
    })?;

    Ok(())
}

fn execute_track(
    test_name: Option<&str>,
    limit: u32,
    llm_provider: &str,
    llm_model: &Option<String>,
    llm_api_key: &Option<String>,
    db_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Tracking metric regressions...");
    println!("Using LLM provider: {llm_provider}");

    Python::attach(|py| -> PyResult<()> {
        let sys = py.import("sys")?;
        let py_path: Bound<'_, PyList> = sys.getattr("path")?.cast_into()?;
        py_path.insert(0, "python")?;

        let bridge = py.import("difftest.agent.bridge")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("provider_name", llm_provider)?;
        kwargs.set_item("db_path", db_path)?;
        kwargs.set_item("limit", limit)?;
        if let Some(name) = test_name {
            kwargs.set_item("test_name", name)?;
        }
        if let Some(ref key) = llm_api_key {
            kwargs.set_item("api_key", key)?;
        }
        if let Some(ref m) = llm_model {
            kwargs.set_item("model", m)?;
        }

        let results = bridge.call_method("bridge_track", (), Some(&kwargs))?;
        let result_list: Bound<'_, PyList> = results.cast_into()?;

        if result_list.is_empty() {
            println!("No historical data found.");
            return Ok(());
        }

        println!("\n{} metric trend(s) analyzed:", result_list.len());
        for item in result_list.iter() {
            let tname: String = item.get_item("test_name")?.extract()?;
            let mname: String = item.get_item("metric_name")?.extract()?;
            let trend: String = item.get_item("trend")?.extract()?;
            let analysis: String = item.get_item("analysis")?.extract()?;
            let alert: bool = item.get_item("alert")?.extract()?;

            let alert_marker = if alert { " [ALERT]" } else { "" };
            println!("\n  {tname} / {mname}: {trend}{alert_marker}");
            println!("    {analysis}");
        }

        Ok(())
    })?;

    Ok(())
}
