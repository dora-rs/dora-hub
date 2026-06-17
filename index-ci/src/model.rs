//! Lean serde mirror of the node-index entry *envelope*.
//!
//! The authoritative definition is `dora-hub-client`'s
//! `index::{IndexEntry, SourceSpec, BinaryArtifact, PackageMeta}` in
//! `dora-rs/dora`. We mirror only the small, stable envelope here — not the
//! manifest internals (`NodeManifest`), which `dora hub publish` validates at
//! publish time — so this CI tool stays free of the `dora-core`/arrow/tokio
//! dependency tree. The drift surface is the five `SourceSpec` fields, which are
//! reserved and stable (spec §8.1).
//!
//! `deny_unknown_fields` on the envelope structs is the schema check: an entry
//! with a stray or misspelled key fails to deserialize.

use std::collections::BTreeMap;

use serde::Deserialize;
use serde_yaml::Value;

/// One published version: manifest snapshot + source pointer (spec §7.1).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndexEntry {
    pub manifest: Manifest,
    pub source: SourceSpec,
    #[serde(default)]
    pub published: Option<String>,
    #[serde(default)]
    pub yanked: bool,
    #[serde(default)]
    pub yank_reason: Option<String>,
}

/// The manifest fields the index envelope constrains (mirrors the old
/// `node-index-entry.schema.json` `manifest.required`). The rest of the manifest
/// body is opaque (`rest`) — the publishing CLI validates it. Missing required
/// fields fail deserialization; `apiVersion`/`entrypoint` values are checked in
/// `validate`.
#[derive(Debug, Deserialize)]
pub struct Manifest {
    #[serde(rename = "apiVersion")]
    pub api_version: u32,
    pub name: String,
    pub namespace: String,
    pub runtime: String,
    pub entrypoint: String,
    #[serde(flatten)]
    pub rest: BTreeMap<String, Value>,
}

/// The source pointer of a published version (spec §8.1).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceSpec {
    #[serde(default)]
    pub git: Option<String>,
    #[serde(default)]
    pub rev: Option<String>,
    #[serde(default)]
    pub subdir: Option<String>,
    // `Option` (not bare `Vec`) so a *present but empty* list can be rejected
    // (the old schema's `minItems: 1`) — absent is `None`, `binary: []` is `Some([])`.
    #[serde(default)]
    pub binary: Option<Vec<BinaryArtifact>>,
    #[serde(default, rename = "fallback-git")]
    pub fallback_git: Option<Box<SourceSpec>>,
}

/// Reserved binary-form artifact (spec §8.1); not consumed in v1.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BinaryArtifact {
    pub platform: String,
    pub url: String,
    pub sha256: String,
}

/// Namespace-level metadata (`package.yml`, spec §7.1/§7.4).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackageMeta {
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub repo: Option<String>,
    #[serde(default)]
    pub owners: Vec<String>,
}

/// One git-pinned level of a source's primary → `fallback-git` chain.
pub struct GitPin<'a> {
    pub git: &'a str,
    pub rev: &'a str,
    pub subdir: Option<&'a str>,
    /// 0 = the primary source; 1+ = position in the `fallback-git` chain.
    pub depth: usize,
}

/// Every git-pinned level reachable from `source`, walking the `fallback-git`
/// chain. Binary-only levels contribute nothing. The audits re-check each of
/// these — a primary *and* a fallback can independently rot, and validation
/// requires a `fallback-git` to be a full pinned source too (§8.1).
pub fn git_pins(source: &SourceSpec) -> Vec<GitPin<'_>> {
    let mut out = Vec::new();
    let mut cur = Some(source);
    let mut depth = 0;
    while let Some(s) = cur {
        if let (Some(git), Some(rev)) = (&s.git, &s.rev) {
            out.push(GitPin {
                git,
                rev,
                subdir: s.subdir.as_deref(),
                depth,
            });
        }
        cur = s.fallback_git.as_deref();
        depth += 1;
    }
    out
}

impl IndexEntry {
    pub fn parse(yaml: &str) -> eyre::Result<Self> {
        Ok(serde_yaml::from_str(yaml)?)
    }
}

impl PackageMeta {
    pub fn parse(yaml: &str) -> eyre::Result<Self> {
        Ok(serde_yaml::from_str(yaml)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(yaml: &str) -> SourceSpec {
        serde_yaml::from_str(yaml).unwrap()
    }

    #[test]
    fn git_pins_includes_a_fallback_under_a_binary_primary() {
        // a binary-primary entry can still pin a git fallback — the audits must
        // re-check it (the gap this fixes), so it shows up as a depth-1 pin
        let s = source(
            "
binary:
  - platform: x86_64-unknown-linux-gnu
    url: https://example.com/x.tgz
    sha256: 0000000000000000000000000000000000000000000000000000000000000000
fallback-git:
  git: https://example.com/r.git
  rev: 1111111111111111111111111111111111111111
  subdir: nodes/n
",
        );
        let pins = git_pins(&s);
        assert_eq!(pins.len(), 1);
        assert_eq!(pins[0].depth, 1);
        assert_eq!(pins[0].subdir, Some("nodes/n"));
    }

    #[test]
    fn git_pins_includes_primary_and_fallback() {
        let s = source(
            "
git: https://example.com/a.git
rev: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
fallback-git:
  git: https://example.com/b.git
  rev: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
",
        );
        let pins = git_pins(&s);
        assert_eq!(pins.len(), 2);
        assert_eq!((pins[0].depth, pins[1].depth), (0, 1));
    }

    #[test]
    fn git_pins_empty_for_a_pure_binary_entry() {
        let s = source(
            "
binary:
  - platform: x86_64-unknown-linux-gnu
    url: https://example.com/x.tgz
    sha256: 0000000000000000000000000000000000000000000000000000000000000000
",
        );
        assert!(git_pins(&s).is_empty());
    }
}
