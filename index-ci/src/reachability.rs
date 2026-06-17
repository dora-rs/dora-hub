//! Pin reachability re-check (spec P3.4, §7.5/§11): for every git-pinned catalog
//! entry, confirm the remote and the exact pinned `rev` still fetch. Catches
//! deleted repos, force-pushed-away commits, and otherwise-rotted sources.
//!
//! This *surfaces a report* rather than mutating entries — it exits non-zero
//! when a source has rotted (so a scheduled run goes red / can file an issue)
//! but never edits the index. Yanked versions are skipped (a yanked version's
//! source is allowed to disappear); binary-only entries have no git rev to check.

use std::path::Path;

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
        let (git_url, rev) = match (&ce.entry.source.git, &ce.entry.source.rev) {
            (Some(g), Some(r)) => (g, r),
            _ => {
                binary_only += 1;
                continue;
            }
        };
        checked += 1;
        if let Err(e) = git::shallow_fetch(git_url, rev) {
            failed.push(format!("{}: {e:#}", ce.rel));
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
