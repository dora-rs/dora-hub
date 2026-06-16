//! `dora-index-ci` — enforcement + auto-merge gate for the node-index/ catalog.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use dora_index_ci::{append_only, decide, namespace, validate};

#[derive(Parser)]
#[command(about = "Enforcement + auto-merge gate for the dora-hub node-index/ catalog")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Validate the catalog: schema, pins, path/name, sibling package, symlinks.
    Validate {
        /// The catalog root (the `node-index/` directory).
        #[arg(long, default_value = "node-index")]
        root: PathBuf,
    },
    /// Enforce the append-only rule on published version files (§7.5).
    AppendOnly {
        #[arg(long)]
        base: String,
    },
    /// Screen newly-claimed namespaces: reserved + confusable (§7.4).
    Namespace {
        #[arg(long)]
        base: String,
    },
    /// Decide whether a PR may auto-merge: MERGE / HOLD (§7.5).
    Decide {
        #[arg(long)]
        author: String,
        #[arg(long)]
        base: String,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Validate { root } => validate::run(&root),
        Command::AppendOnly { base } => append_only::run(&base),
        Command::Namespace { base } => namespace::run(&base),
        Command::Decide { author, base } => decide::run(&author, &base),
    };
    match result {
        Ok(code) => ExitCode::from(code as u8),
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::from(2)
        }
    }
}
