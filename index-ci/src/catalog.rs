//! Shared catalog walk for the audit subcommands (`reachability`,
//! `integrity-audit`). Collects the *parseable* version entries
//! (`node-index/<ns>/<name>/<semver>.yml`); `package.yml`, symlinks, and
//! non-version files are skipped, and an unparsable entry is reported as a
//! `::warning::` and skipped — the `validate` subcommand is the gate that fails
//! a PR on a malformed entry, so the periodic audits only re-check entries that
//! already passed it.

use std::path::{Path, PathBuf};

use crate::model::IndexEntry;

/// A parsed published version entry plus its catalog-relative path for reporting.
pub struct CatalogEntry {
    pub rel: String,
    pub entry: IndexEntry,
}

/// Collect the parseable version entries under `root` (the `node-index/` dir).
pub fn version_entries(root: &Path) -> eyre::Result<Vec<CatalogEntry>> {
    let mut files = Vec::new();
    collect(root, &mut files)?;
    let mut out = Vec::new();
    for path in files {
        let parts: Vec<String> = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .components()
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect();
        // only node-index/<ns>/<name>/<file>.yml version entries
        if parts.len() != 3 || parts[2] == "package.yml" {
            continue;
        }
        let rel = parts.join("/");
        let raw = match std::fs::read_to_string(&path) {
            Ok(raw) => raw,
            Err(e) => {
                println!("::warning::{rel}: cannot read: {e}");
                continue;
            }
        };
        match IndexEntry::parse(&raw) {
            Ok(entry) => out.push(CatalogEntry { rel, entry }),
            Err(e) => println!("::warning::{rel}: skipped (invalid entry: {e})"),
        }
    }
    Ok(out)
}

/// Recursively collect regular `*.yml` files under `dir`. Symlinks are not
/// followed and not returned (the `validate` subcommand rejects them); a missing
/// directory yields an empty list so the audits no-op on an unpopulated catalog.
fn collect(dir: &Path, out: &mut Vec<PathBuf>) -> eyre::Result<()> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e.into()),
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let meta = std::fs::symlink_metadata(&path)?;
        if meta.file_type().is_symlink() {
            continue;
        }
        if meta.is_dir() {
            collect(&path, out)?;
        } else if path.extension().and_then(|e| e.to_str()) == Some("yml") {
            out.push(path);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(path: &Path, body: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, body).unwrap();
    }

    const ENTRY: &str = "manifest:\n  apiVersion: 1\n  name: n\n  namespace: acme\n  \
        runtime: rust\n  entrypoint: e\nsource:\n  git: https://example.com/r.git\n  \
        rev: 0000000000000000000000000000000000000000\n";

    #[test]
    fn collects_version_entries_and_skips_package_and_bad_files() {
        let root = tempfile::tempdir().unwrap();
        let p = root.path();
        write(&p.join("acme/n/1.0.0.yml"), ENTRY);
        write(&p.join("acme/n/package.yml"), "owners: [acme]\n");
        write(&p.join("acme/n/2.0.0.yml"), "manifest: {oops\n"); // unparsable → skipped
        write(&p.join("acme/loose.yml"), ENTRY); // wrong depth → skipped

        let entries = version_entries(p).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].rel, "acme/n/1.0.0.yml");
    }

    #[test]
    fn empty_catalog_yields_no_entries() {
        let root = tempfile::tempdir().unwrap();
        assert!(version_entries(root.path()).unwrap().is_empty());
        // a never-created node-index dir also no-ops (the scheduled audit case)
        assert!(
            version_entries(&root.path().join("missing"))
                .unwrap()
                .is_empty()
        );
    }
}
