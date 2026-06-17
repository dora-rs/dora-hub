//! Integrity spot-audit (spec P3.4, §7.5/§11): for a sample of catalog entries,
//! re-fetch the pinned source and confirm the manifest committed at that exact
//! `rev` still matches the entry's stored manifest snapshot.
//!
//! This complements the append-only rule: append-only stops a *published version
//! file* from being edited in place; this catches an entry whose manifest was
//! rewritten to disagree with what is actually committed at the pinned commit
//! (e.g. a published entry pointing at a benign commit but claiming a different
//! `entrypoint`/`runtime`).
//!
//! The entry format stores the manifest snapshot, not a content digest, so the
//! audit compares the manifest *envelope* — the trust-bearing, always-present
//! fields the resolver and `validate` rely on (`apiVersion`, `name`,
//! `namespace`, `runtime`, `entrypoint`) — rather than hashing the opaque
//! manifest body, which would false-positive on serde round-trip differences.

use std::path::Path;

use crate::model::{Manifest, git_pins};
use crate::{catalog, git};

/// The node manifest filename in a source repo (mirrors dora-core's
/// `manifest::MANIFEST_FILENAME`; `dora hub publish` reads `<subdir>/dora-node.yml`).
const MANIFEST_FILENAME: &str = "dora-node.yml";

/// `sample`: audit only the first N entries in sorted order (a cheap spot-check),
/// or `None` to audit the whole catalog.
pub fn run(root: &Path, sample: Option<usize>) -> eyre::Result<i32> {
    let mut entries = catalog::version_entries(root)?;
    entries.sort_by(|a, b| a.rel.cmp(&b.rel)); // deterministic order / sample
    if let Some(n) = sample {
        entries.truncate(n);
    }

    let mut checked = 0usize;
    let mut drift: Vec<String> = Vec::new();

    for ce in &entries {
        if ce.entry.yanked {
            continue;
        }
        // audit every git-pinned level — primary *and* each `fallback-git`,
        // which hosts the same node and must match the stored manifest too
        for pin in git_pins(&ce.entry.source) {
            let site = crate::audit_site(&ce.rel, pin.depth);
            let manifest_path = match pin.subdir {
                Some(s) => format!("{s}/{MANIFEST_FILENAME}"),
                None => MANIFEST_FILENAME.to_owned(),
            };
            checked += 1;
            let dir = match git::shallow_fetch(pin.git, pin.rev) {
                Ok(dir) => dir,
                Err(e) => {
                    drift.push(format!("{site}: source unreachable: {e:#}"));
                    continue;
                }
            };
            let committed = match git::show_in(dir.path(), pin.rev, &manifest_path)? {
                Some(committed) => committed,
                None => {
                    drift.push(format!("{site}: {manifest_path} absent at pinned commit"));
                    continue;
                }
            };
            let actual: Manifest = match serde_yaml::from_str(&committed) {
                Ok(actual) => actual,
                Err(e) => {
                    drift.push(format!(
                        "{site}: committed {MANIFEST_FILENAME} is unparseable: {e}"
                    ));
                    continue;
                }
            };
            if let Some(reason) = envelope_drift(&ce.entry.manifest, &actual) {
                drift.push(format!("{site}: {reason}"));
            }
        }
    }

    for d in &drift {
        println!("::error::{d}");
    }

    if checked == 0 {
        println!("integrity-audit: OK (no git-pinned entries to audit)");
        return Ok(0);
    }
    if drift.is_empty() {
        println!("integrity-audit: OK ({checked} entr(ies) match their pinned source)");
        Ok(0)
    } else {
        println!(
            "integrity-audit: {} of {checked} entr(ies) DRIFTED from source",
            drift.len()
        );
        Ok(1)
    }
}

/// First mismatch between the stored manifest snapshot and the manifest actually
/// committed at the pinned `rev`, or `None` if the envelope matches.
fn envelope_drift(stored: &Manifest, actual: &Manifest) -> Option<String> {
    if stored.api_version != actual.api_version {
        return Some(format!(
            "apiVersion drift: entry says {}, source has {}",
            stored.api_version, actual.api_version
        ));
    }
    if stored.name != actual.name {
        return Some(format!(
            "name drift: entry says `{}`, source has `{}`",
            stored.name, actual.name
        ));
    }
    if stored.namespace != actual.namespace {
        return Some(format!(
            "namespace drift: entry says `{}`, source has `{}`",
            stored.namespace, actual.namespace
        ));
    }
    if stored.runtime != actual.runtime {
        return Some(format!(
            "runtime drift: entry says `{}`, source has `{}`",
            stored.runtime, actual.runtime
        ));
    }
    if stored.entrypoint != actual.entrypoint {
        return Some(format!(
            "entrypoint drift: entry says `{}`, source has `{}`",
            stored.entrypoint, actual.entrypoint
        ));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest(yaml: &str) -> Manifest {
        serde_yaml::from_str(yaml).unwrap()
    }

    const BASE: &str = "apiVersion: 1\nname: a\nnamespace: ns\nruntime: rust\nentrypoint: e\n";

    #[test]
    fn detects_envelope_field_drift() {
        let stored = manifest(BASE);
        let drifted =
            manifest("apiVersion: 1\nname: b\nnamespace: ns\nruntime: rust\nentrypoint: e\n");
        assert!(
            envelope_drift(&stored, &drifted)
                .unwrap()
                .contains("name drift")
        );
    }

    #[test]
    fn ignores_opaque_body_differences() {
        // same envelope, different opaque body (`description`) → not drift
        let stored = manifest(&format!("{BASE}description: one\n"));
        let actual = manifest(&format!("{BASE}description: two\n"));
        assert!(envelope_drift(&stored, &actual).is_none());
    }
}
