//! `dora-index-ci` — enforcement + auto-merge gate for the node-index/ catalog.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use dora_index_ci::{append_only, decide, identity, integrity, namespace, reachability, validate};

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
    /// Verify a new namespace matches the author's GitHub identity (§7.4).
    Identity {
        #[arg(long)]
        author: String,
        #[arg(long)]
        base: String,
    },
    /// Re-check that every pinned source still fetches (P3.4, periodic).
    Reachability {
        /// The catalog root (the `node-index/` directory).
        #[arg(long, default_value = "node-index")]
        root: PathBuf,
    },
    /// Re-check that entries still match their pinned source (P3.4, periodic).
    IntegrityAudit {
        /// The catalog root (the `node-index/` directory).
        #[arg(long, default_value = "node-index")]
        root: PathBuf,
        /// Audit only the first N entries (sorted); omit to audit all.
        #[arg(long)]
        sample: Option<usize>,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Validate { root } => validate::run(&root),
        Command::AppendOnly { base } => append_only::run(&base),
        Command::Namespace { base } => namespace::run(&base),
        Command::Decide { author, base } => decide::run(&author, &base),
        Command::Identity { author, base } => identity::run(&author, &base),
        Command::Reachability { root } => reachability::run(&root),
        Command::IntegrityAudit { root, sample } => integrity::run(&root, sample),
    };
    match result {
        Ok(code) => ExitCode::from(code as u8),
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::from(2)
        }
    }
}
