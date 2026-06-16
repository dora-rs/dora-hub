//! The auto-merge MERGE/HOLD verdict (spec §7.5).
//!
//! The bot runs this from the **trusted base** binary against a PR's catalog
//! data. It returns MERGE only for a routine publish: the PR touches only
//! version files in an existing namespace, every change is an add/append (no
//! delete/rename), no new namespace is claimed, and every entry is authored by
//! an OWNER of its namespace — where OWNERS is read from the **base tip**, not
//! the attacker-chosen merge-base, so a former owner can't branch from an old
//! commit and self-approve.

use std::process::Command;

use semver::Version;

use crate::git::{self, namespaces_at, show};
use crate::is_valid_key_part;
use crate::model::PackageMeta;

/// `node-index/<ns>/<name>/<semver>.yml` — the only path an auto-merge may touch.
pub fn is_version_file(path: &str) -> bool {
    let parts: Vec<&str> = path.split('/').collect();
    if parts.len() != 4 || parts[0] != "node-index" {
        return false;
    }
    let Some(stem) = parts[3].strip_suffix(".yml") else {
        return false;
    };
    is_valid_key_part(parts[1]) && is_valid_key_part(parts[2]) && Version::parse(stem).is_ok()
}

/// OWNERS from the `package.yml` at `reference` (empty if absent/malformed).
pub fn owners_at(reference: &str, ns: &str, name: &str) -> eyre::Result<Vec<String>> {
    let path = format!("node-index/{ns}/{name}/package.yml");
    match show(reference, &path)? {
        Some(text) => Ok(PackageMeta::parse(&text)
            .map(|m| m.owners)
            .unwrap_or_default()),
        None => Ok(Vec::new()),
    }
}

/// Author owns the namespace directly, or via an org in the OWNERS list.
pub fn is_authorized(
    author: &str,
    owners: &[String],
    is_member: &dyn Fn(&str, &str) -> bool,
) -> bool {
    if owners.iter().any(|o| o == author) {
        return true;
    }
    owners.iter().any(|o| o != author && is_member(o, author))
}

/// True if `user` is a public member of org `org` (via `gh api`).
fn gh_is_member(org: &str, user: &str) -> bool {
    Command::new("gh")
        .args(["api", &format!("/orgs/{org}/members/{user}"), "--silent"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Return the HOLD reasons; empty means auto-mergeable. `base` is the base tip.
pub fn decide(
    author: &str,
    base: &str,
    is_member: &dyn Fn(&str, &str) -> bool,
) -> eyre::Result<Vec<String>> {
    let mb = git::merge_base(base)?;
    let files = git::changed_files(&mb)?;
    let mut reasons = Vec::new();

    // path + status guard: only added/appended version files may auto-merge
    let mut bad_paths: Vec<&str> = files
        .iter()
        .filter(|c| !is_version_file(&c.path))
        .map(|c| c.path.as_str())
        .collect();
    bad_paths.sort_unstable();
    bad_paths.dedup();
    if !bad_paths.is_empty() {
        reasons.push(format!(
            "touches files that need human review (§7.5): {}",
            bad_paths.join(", ")
        ));
    }
    let mut bad_status: Vec<&str> = files
        .iter()
        .filter(|c| c.status != 'A' && c.status != 'M')
        .map(|c| c.path.as_str())
        .collect();
    bad_status.sort_unstable();
    bad_status.dedup();
    if !bad_status.is_empty() {
        reasons.push(format!(
            "deletes/renames a published file: {}",
            bad_status.join(", ")
        ));
    }

    // new-namespace guard (§7.4) — against the base tip, so resurrecting a
    // namespace deleted on base (but present at the branch point) is flagged new
    let new_ns: Vec<String> = namespaces_at("HEAD")?
        .difference(&namespaces_at(base)?)
        .cloned()
        .collect();
    if !new_ns.is_empty() {
        reasons.push(format!("claims new namespace(s): {}", new_ns.join(", ")));
    }

    // owner guard (§7.5): every touched version entry must be owner-authored,
    // against the OWNERS as they exist on the base tip
    for change in &files {
        if !is_version_file(&change.path) {
            continue;
        }
        let parts: Vec<&str> = change.path.split('/').collect();
        let (ns, name) = (parts[1], parts[2]);
        let owners = owners_at(base, ns, name)?;
        if owners.is_empty() {
            reasons.push(format!(
                "{}: no owners on the base package.yml",
                change.path
            ));
        } else if !is_authorized(author, &owners, is_member) {
            reasons.push(format!(
                "{}: @{author} is not an owner of {ns}/{name}",
                change.path
            ));
        }
    }

    Ok(reasons)
}

/// CLI entry: print `MERGE`/`HOLD: …` and return the exit code (0 = mergeable).
pub fn run(author: &str, base: &str) -> eyre::Result<i32> {
    let reasons = decide(author, base, &gh_is_member)?;
    if reasons.is_empty() {
        println!("MERGE");
        Ok(0)
    } else {
        println!("HOLD: {}", reasons.join("; "));
        Ok(1)
    }
}
