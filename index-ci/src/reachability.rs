//! Pin reachability re-check (spec P3.4, §7.5/§11): for every git-pinned catalog
//! entry, confirm the remote and the exact pinned `rev` still fetch. Catches
//! deleted repos, force-pushed-away commits, and otherwise-rotted sources.
//!
//! This *surfaces a report* rather than mutating entries — it exits non-zero
//! when a source has rotted (so a scheduled run goes red / can file an issue)
//! but never edits the index. Yanked versions are skipped (a yanked version's
//! source is allowed to disappear); binary-only entries have no git rev to check.

use std::path::Path;

use crate::model::git_pins;
use crate::{catalog, git};

pub fn run(root: &Path) -> eyre::Result<i32> {
    let entries = catalog::version_entries(root)?;
    let mut checked = 0usize;
    let mut binary_only = 0usize;
    let mut failed: Vec<String> = Vec::new();

    for ce in &entries {
        if ce.entry.yanked {
            continue;
        }
        // re-check every git-pinned level — the primary source *and* each
        // `fallback-git` (a binary-primary entry can still pin a git fallback)
        let pins = git_pins(&ce.entry.source);
        if pins.is_empty() {
            binary_only += 1;
            continue;
        }
        for pin in pins {
            checked += 1;
            if let Err(e) = git::shallow_fetch(pin.git, pin.rev) {
                failed.push(format!("{}: {e:#}", super::audit_site(&ce.rel, pin.depth)));
            }
        }
    }

    for f in &failed {
        println!("::error::{f}");
    }

    if checked == 0 {
        println!(
            "reachability: OK (no git-pinned entries to check; {binary_only} binary-only skipped)"
        );
        return Ok(0);
    }
    if failed.is_empty() {
        println!("reachability: OK ({checked} source(s) reachable)");
        Ok(0)
    } else {
        println!(
            "reachability: {} of {checked} source(s) UNREACHABLE",
            failed.len()
        );
        Ok(1)
    }
}
