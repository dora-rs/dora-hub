//! Enforcement + auto-merge gate for the dora-hub `node-index/` catalog (§7).
//!
//! Subcommands mirroring the spec's machine-enforced rules:
//! - `validate`        — schema (serde), pins, path/name, sibling package, symlinks;
//! - `append-only`     — published versions are immutable (§7.5);
//! - `namespace`       — reserved + confusable screening of new claims (§7.4);
//! - `decide`          — the auto-merge MERGE/HOLD verdict for the bot (§7.5);
//! - `reachability`    — pinned sources still fetch (P3.4, periodic);
//! - `integrity-audit` — entries still match their pinned source (P3.4, periodic).

pub mod append_only;
pub mod catalog;
pub mod decide;
pub mod git;
pub mod integrity;
pub mod model;
pub mod namespace;
pub mod reachability;
pub mod validate;

/// Label an audit finding by its source-chain position: the entry path for the
/// primary source, or `<rel> (fallback N)` for a `fallback-git` level.
pub fn audit_site(rel: &str, depth: usize) -> String {
    if depth == 0 {
        rel.to_owned()
    } else {
        format!("{rel} (fallback {depth})")
    }
}

/// A valid namespace/name path segment of a package key (mirrors
/// `dora-hub-client`'s `index::is_valid_key_part`): a bounded `[A-Za-z0-9._-]`
/// token, non-empty, not starting with `.` (so `..`/dotfiles can't traverse).
pub fn is_valid_key_part(part: &str) -> bool {
    !part.is_empty()
        && part.len() <= 64
        && part
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
        && !part.starts_with('.')
}
