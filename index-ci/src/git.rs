//! Thin wrappers over the `git` CLI, run in the current working directory.

use std::path::Path;
use std::process::Command;

use eyre::{Context, ContextCompat, bail};

/// Run `git <args>` and return stdout, erroring on a non-zero exit.
pub fn git(args: &[&str]) -> eyre::Result<String> {
    let out = Command::new("git")
        .args(args)
        .output()
        .with_context(|| format!("failed to run `git {}`", args.join(" ")))?;
    if !out.status.success() {
        bail!(
            "`git {}` failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// merge-base(base, HEAD).
pub fn merge_base(base: &str) -> eyre::Result<String> {
    Ok(git(&["merge-base", base, "HEAD"])?.trim().to_owned())
}

/// A changed file in `from..HEAD`: its status letter, (new) path, and the old
/// path for a rename/copy.
pub struct Change {
    pub status: char,
    pub path: String,
    pub old_path: Option<String>,
}

/// `git diff --name-status -M <from> HEAD`, parsed.
pub fn changed_files(from: &str) -> eyre::Result<Vec<Change>> {
    let out = git(&["diff", "--name-status", "-M", from, "HEAD"])?;
    let mut changes = Vec::new();
    for line in out.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        // rename/copy: "<R|C><score>\told\tnew" — the new path is the last column
        let status = cols[0].chars().next().unwrap_or('?');
        let path = cols.last().copied().unwrap_or("").to_owned();
        let old_path = (matches!(status, 'R' | 'C') && cols.len() >= 3).then(|| cols[1].to_owned());
        changes.push(Change {
            status,
            path,
            old_path,
        });
    }
    Ok(changes)
}

/// Contents of `<ref>:<path>`, or `None` if the path doesn't exist at that ref.
pub fn show(reference: &str, path: &str) -> eyre::Result<Option<String>> {
    let out = Command::new("git")
        .args(["show", &format!("{reference}:{path}")])
        .output()
        .context("failed to run `git show`")?;
    if out.status.success() {
        Ok(Some(String::from_utf8_lossy(&out.stdout).into_owned()))
    } else {
        Ok(None)
    }
}

/// Shallow-fetch a single commit into a throwaway repo so the audit subcommands
/// can re-check reachability and inspect committed files. Errors if the remote
/// or that exact `rev` is unreachable (a deleted repo, a force-pushed-away
/// commit, a rotted source).
///
/// Self-guards against argument injection before shelling out: `url` must not
/// start with `-` (so `git` can't read it as an option) and `rev` must be a hex
/// object id. Callers pass entries the `validate` subcommand already accepted;
/// this is a belt-and-suspenders second check.
pub fn shallow_fetch(url: &str, rev: &str) -> eyre::Result<tempfile::TempDir> {
    if url.is_empty() || url.starts_with('-') || url.contains(|c: char| c.is_control()) {
        bail!("refusing to fetch suspicious url `{url}`");
    }
    if rev.is_empty() || !rev.chars().all(|c| c.is_ascii_hexdigit()) {
        bail!("refusing to fetch non-hex rev `{rev}`");
    }
    let dir = tempfile::tempdir().context("failed to create temp dir for fetch")?;
    let path = dir.path().to_str().context("temp dir path is not UTF-8")?;
    git(&["-C", path, "init", "-q"])?;
    git(&["-C", path, "fetch", "--depth", "1", "-q", url, rev])
        .with_context(|| format!("`{url}` @ {} is unreachable", &rev[..rev.len().min(12)]))?;
    Ok(dir)
}

/// Contents of `<rev>:<path>` inside an already-fetched repo `dir`, or `None` if
/// that path doesn't exist at the rev.
pub fn show_in(dir: &Path, rev: &str, path: &str) -> eyre::Result<Option<String>> {
    let repo = dir.to_str().context("repo path is not UTF-8")?;
    let out = Command::new("git")
        .args(["-C", repo, "show", &format!("{rev}:{path}")])
        .output()
        .context("failed to run `git show`")?;
    Ok(out
        .status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).into_owned()))
}

/// Namespaces present under `node-index/` at `<ref>` (dirs with a package.yml).
pub fn namespaces_at(reference: &str) -> eyre::Result<std::collections::BTreeSet<String>> {
    let out = git(&["ls-tree", "-r", "--name-only", reference, "node-index/"])?;
    let mut namespaces = std::collections::BTreeSet::new();
    for path in out.lines() {
        let parts: Vec<&str> = path.split('/').collect();
        // node-index/<ns>/<name>/package.yml
        if parts.len() >= 4 && parts[0] == "node-index" && *parts.last().unwrap() == "package.yml" {
            namespaces.insert(parts[1].to_owned());
        }
    }
    Ok(namespaces)
}

#[cfg(test)]
mod tests {
    use super::*;

    // The fetch+show happy path is exercised end-to-end against a real local
    // repo in `tests/audit-e2e.rs`; here we only pin the cheap injection guards
    // that must reject before any `git` process is spawned.
    #[test]
    fn shallow_fetch_rejects_option_shaped_urls_and_non_hex_revs() {
        let hex = "a".repeat(40);
        assert!(shallow_fetch("--upload-pack=touch pwned", &hex).is_err());
        assert!(shallow_fetch("", &hex).is_err());
        assert!(shallow_fetch("https://example.com/r.git", "not-hex").is_err());
        assert!(shallow_fetch("https://example.com/r.git", "").is_err());
    }
}
