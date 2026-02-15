mod agent;
mod baseline;
mod bridge;
mod cache;
mod compare;
mod config;
mod run;

use clap::{Parser, Subcommand};
use difftest_core::error::DifftestError;

#[derive(Parser)]
#[command(name = "difftest", about = "pytest for diffusion models")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run test suite against a model
    Run(run::RunArgs),
    /// Manage baseline images for visual regression tests
    Baseline(baseline::BaselineArgs),
    /// LLM-powered test design, diagnosis, and regression tracking
    Agent(agent::AgentArgs),
    /// Compare two runs with statistical tests (Welch's t-test, Cohen's d)
    Compare(compare::CompareArgs),
}

fn main() {
    let cli = Cli::parse();
    let cfg = config::load_config();

    let result = match cli.command {
        Commands::Run(args) => run::execute(args, &cfg),
        Commands::Baseline(args) => baseline::execute(args, &cfg),
        Commands::Agent(args) => agent::execute(args),
        Commands::Compare(args) => compare::execute(args),
    };

    if let Err(e) = result {
        match &e {
            DifftestError::Config { field, value, reason } => {
                eprintln!("Configuration error: {field} = {value}: {reason}");
            }
            DifftestError::SuiteTimeout { timeout_seconds } => {
                eprintln!("Suite timeout exceeded ({timeout_seconds}s)");
            }
            DifftestError::ConfigFile(msg) => {
                eprintln!("Config file error: {msg}");
            }
            _ => {
                eprintln!("Error: {e}");
            }
        }
        std::process::exit(2);
    }
}
