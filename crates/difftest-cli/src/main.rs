mod agent;
mod baseline;
mod bridge;
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
}

fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        Commands::Run(args) => run::execute(args),
        Commands::Baseline(args) => baseline::execute(args),
        Commands::Agent(args) => agent::execute(args),
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
