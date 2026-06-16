//! Append-only enforcement (spec §7.5): a published version file is immutable.
//! Across a PR, the only permitted edits to an existing `<version>.yml` are
//! flipping `yanked` (+ a non-empty `yank_reason`) and *appending* late
//! `source.binary` artifacts for platforms not already pinned.

use serde_yaml::{Mapping, Value};

use crate::git::{self, merge_base, show};

/// `None` if the edit from `old` to `new` is an allowed yank/binary-add; else a
/// reason string. Operates on the parsed YAML values so it sees exactly what
/// changed, independent of the typed envelope.
pub fn allowed_version_edit(old: &Value, new: &Value) -> Option<String> {
    // a yank must carry a non-empty reason (§7.5)
    if new.get("yanked") == Some(&Value::Bool(true)) {
        let ok = new
            .get("yank_reason")
            .and_then(Value::as_str)
            .is_some_and(|r| !r.trim().is_empty());
        if !ok {
            return Some("`yanked: true` requires a non-empty `yank_reason`".into());
        }
    }

    let mut o = old.clone();
    let mut n = new.clone();
    for entry in [&mut o, &mut n] {
        if let Some(map) = entry.as_mapping_mut() {
            map.remove("yanked");
            map.remove("yank_reason");
        }
    }

    let o_bins = take_binary(&mut o);
    let n_bins = take_binary(&mut n);

    // existing binary artifacts must survive unchanged
    for b in &o_bins {
        if !n_bins.contains(b) {
            return Some("an existing `source.binary` artifact was changed or removed".into());
        }
    }
    // an appended artifact may only cover a platform not already pinned — re-adding
    // an existing platform with a different url/sha shadows the original
    let o_platforms: Vec<Option<&str>> = o_bins.iter().map(platform_of).collect();
    for b in &n_bins {
        if !o_bins.contains(b) && o_platforms.contains(&platform_of(b)) {
            return Some(
                "a `source.binary` platform already pinned was re-added (shadowing)".into(),
            );
        }
    }

    if o != n {
        return Some("the entry changed beyond `yanked`/`yank_reason` or binary additions".into());
    }
    None
}

/// Remove and return `source.binary` from a version-entry value (empty if none).
fn take_binary(entry: &mut Value) -> Vec<Value> {
    entry
        .as_mapping_mut()
        .and_then(|m| m.get_mut("source"))
        .and_then(Value::as_mapping_mut)
        .and_then(|s| s.remove("binary"))
        .and_then(|b| match b {
            Value::Sequence(seq) => Some(seq),
            _ => None,
        })
        .unwrap_or_default()
}

fn platform_of(artifact: &Value) -> Option<&str> {
    artifact.get("platform").and_then(Value::as_str)
}

fn is_version_entry(path: &str) -> bool {
    path.ends_with(".yml") && !path.ends_with("/package.yml")
}

fn under_index(path: &str) -> bool {
    path.starts_with("node-index/") && path.ends_with(".yml")
}

/// Enforce the append-only rule across `merge-base(base, HEAD)..HEAD`.
pub fn run(base: &str) -> eyre::Result<i32> {
    let mb = merge_base(base)?;
    let mut errors = 0u32;
    let changes = git::changed_files(&mb)?;

    for change in &changes {
        if !under_index(&change.path) {
            continue;
        }
        match change.status {
            'A' => {} // adding a new file is always allowed
            'D' => {
                println!(
                    "::error::{}: a published index file must not be deleted (append-only)",
                    change.path
                );
                errors += 1;
            }
            'R' | 'C' => {
                println!(
                    "::error::{}: a published index file must not be renamed/moved",
                    change.path
                );
                errors += 1;
            }
            'M' => {
                if !is_version_entry(&change.path) {
                    println!(
                        "::warning::{}: package metadata changed — needs human review",
                        change.path
                    );
                    continue;
                }
                let old = parse(show(&mb, &change.path)?, &change.path)?;
                let new = parse(show("HEAD", &change.path)?, &change.path)?;
                if let Some(reason) = allowed_version_edit(&old, &new) {
                    println!(
                        "::error::{}: {reason} — published versions are immutable (§7.5)",
                        change.path
                    );
                    errors += 1;
                }
            }
            other => {
                println!(
                    "::error::{}: unexpected change status `{other}`",
                    change.path
                );
                errors += 1;
            }
        }
    }

    if errors > 0 {
        println!("check_append_only: {errors} violation(s)");
        return Ok(1);
    }
    println!("check_append_only: OK ({} index change(s))", changes.len());
    Ok(0)
}

fn parse(yaml: Option<String>, path: &str) -> eyre::Result<Value> {
    let text = yaml.ok_or_else(|| eyre::eyre!("could not read `{path}`"))?;
    Ok(serde_yaml::from_str(&text).unwrap_or(Value::Mapping(Mapping::new())))
}
