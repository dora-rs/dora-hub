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
    #[serde(default)]
    pub binary: Vec<BinaryArtifact>,
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
pub struct PackageMeta {
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub repo: Option<String>,
    #[serde(default)]
    pub owners: Vec<String>,
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
