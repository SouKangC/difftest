mod bridge;
mod run;

use clap::{Parser, Subcommand};

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
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run(args) => {
            if let Err(e) = run::execute(args) {
                eprintln!("Error: {e}");
                std::process::exit(2);
            }
        }
    }
}
