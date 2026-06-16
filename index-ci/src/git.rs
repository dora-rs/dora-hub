//! Thin wrappers over the `git` CLI, run in the current working directory.

use std::process::Command;

use eyre::{Context, bail};

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

/// A changed file in `from..HEAD`: its status letter and (new) path.
pub struct Change {
    pub status: char,
    pub path: String,
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
        changes.push(Change { status, path });
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
