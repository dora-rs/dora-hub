//! Validate catalog entries: schema (serde), pins, path/name consistency,
//! sibling `package.yml`, source usability, subdir traversal, binary-platform
//! uniqueness, and symlink rejection.

use std::path::{Path, PathBuf};

use eyre::bail;

use crate::is_valid_key_part;
use crate::model::{IndexEntry, PackageMeta, SourceSpec};
use crate::namespace::reserved;

/// Validate the whole catalog under `root` (the `node-index/` directory).
pub fn run(root: &Path) -> eyre::Result<i32> {
    let reserved = reserved();
    let mut errors: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    let mut files = Vec::new();
    collect_yml(root, &mut files)?;
    let checked = files.len();

    for (path, is_symlink) in files {
        let rel = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .display()
            .to_string();
        // A catalog entry must be a regular file inside the catalog. A symlink
        // (escaping or dangling) would otherwise smuggle unvalidated content in.
        if is_symlink {
            errors.push(format!(
                "{rel}: catalog entries must be regular files, not symlinks"
            ));
            continue;
        }
        validate_file(root, &path, &rel, &reserved, &mut errors, &mut warnings);
    }

    for w in &warnings {
        println!("::warning::{w}");
    }
    for e in &errors {
        println!("::error::{e}");
    }
    if !errors.is_empty() {
        println!("validate: {} error(s) in {checked} file(s)", errors.len());
        return Ok(1);
    }
    println!("validate: OK ({checked} file(s) checked)");
    Ok(0)
}

fn validate_file(
    root: &Path,
    path: &Path,
    rel: &str,
    reserved: &std::collections::BTreeSet<String>,
    errors: &mut Vec<String>,
    warnings: &mut Vec<String>,
) {
    let parts: Vec<String> = path
        .strip_prefix(root)
        .unwrap_or(path)
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    if parts.len() != 3 {
        errors.push(format!(
            "{rel}: catalog files must live at node-index/<ns>/<name>/<file>.yml"
        ));
        return;
    }
    let (namespace, name, filename) = (&parts[0], &parts[1], &parts[2]);

    if !is_valid_key_part(namespace) {
        errors.push(format!("{rel}: invalid namespace `{namespace}`"));
    }
    if !is_valid_key_part(name) {
        errors.push(format!("{rel}: invalid package name `{name}`"));
    }

    let raw = match std::fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(e) => {
            errors.push(format!("{rel}: cannot read: {e}"));
            return;
        }
    };

    if filename == "package.yml" {
        match PackageMeta::parse(&raw) {
            Ok(meta) if meta.owners.is_empty() => {
                errors.push(format!("{rel}: package.yml must list at least one owner"));
            }
            Ok(_) => {}
            Err(e) => errors.push(format!("{rel}: invalid package metadata: {e}")),
        }
        if reserved.contains(namespace) {
            warnings.push(format!(
                "{rel}: namespace `{namespace}` is reserved — needs an index admin (§7.4)"
            ));
        }
        return;
    }

    // version entry: stem must be semver
    let stem = filename.strip_suffix(".yml").unwrap_or(filename);
    if semver::Version::parse(stem).is_err() {
        errors.push(format!(
            "{rel}: filename `{filename}` is not a `<semver>.yml`"
        ));
    }
    // every version needs sibling package metadata (owners) to anchor owner checks
    if !path.with_file_name("package.yml").is_file() {
        errors.push(format!(
            "{rel}: no sibling `package.yml` — every node needs package metadata"
        ));
    }

    let entry = match IndexEntry::parse(&raw) {
        Ok(entry) => entry,
        Err(e) => {
            errors.push(format!("{rel}: invalid entry: {e}"));
            return;
        }
    };
    if entry.manifest.namespace != *namespace {
        errors.push(format!(
            "{rel}: manifest.namespace `{}` != directory `{namespace}`",
            entry.manifest.namespace
        ));
    }
    if entry.manifest.name != *name {
        errors.push(format!(
            "{rel}: manifest.name `{}` != directory `{name}`",
            entry.manifest.name
        ));
    }
    if entry.manifest.api_version != 1 {
        errors.push(format!("{rel}: manifest.apiVersion must be 1"));
    }
    if entry.manifest.entrypoint.is_empty() {
        errors.push(format!("{rel}: manifest.entrypoint must not be empty"));
    }
    if let Err(e) = validate_source(&entry.source) {
        errors.push(format!("{rel}: {e}"));
    }
}

/// A usable, well-formed source: a full-hash git pin (URL scheme-checked) or a
/// non-empty binary list, with a safe `subdir` and unique binary platforms.
/// Mirrors `dora-hub-client`'s `SourceSpec::git_pin` rules.
pub fn validate_source(source: &SourceSpec) -> eyre::Result<()> {
    match (&source.git, &source.rev) {
        (Some(git), Some(rev)) => {
            validate_git_url(git)?;
            let valid = matches!(rev.len(), 40 | 64) && rev.chars().all(|c| c.is_ascii_hexdigit());
            if !valid {
                bail!("`rev` must be a full 40- or 64-char hex object id");
            }
        }
        (None, None) if !source.binary.is_empty() => {} // binary form (reserved, P2.8)
        (None, None) => bail!("no usable source: needs `git`+`rev` or a non-empty `binary`"),
        _ => bail!("incomplete git source (needs both `git` and `rev`)"),
    }
    if let Some(subdir) = &source.subdir {
        validate_subdir(subdir)?;
    }
    check_binary_platforms(source)?;
    Ok(())
}

/// Reject a `subdir` that could escape the checkout: relative, no `..`, single
/// line, no backslash.
fn validate_subdir(subdir: &str) -> eyre::Result<()> {
    if subdir.is_empty()
        || subdir.starts_with('/')
        || subdir.contains("..")
        || subdir.contains('\\')
        || subdir.contains(['\n', '\r'])
    {
        bail!("invalid `subdir` (must be relative, no `..`, single line)");
    }
    Ok(())
}

/// Validate each `binary` artifact (non-empty platform/url, 64-hex sha256) and
/// require a unique `platform` (a duplicate lets first/last-match consumers
/// resolve different artifacts). Recurses into `fallback-git`.
fn check_binary_platforms(source: &SourceSpec) -> eyre::Result<()> {
    let mut seen = std::collections::HashSet::new();
    for art in &source.binary {
        if art.platform.is_empty() || art.url.is_empty() {
            bail!("`source.binary` artifact has an empty `platform` or `url`");
        }
        if art.sha256.len() != 64 || !art.sha256.chars().all(|c| c.is_ascii_hexdigit()) {
            bail!("`source.binary` artifact `sha256` must be 64 hex chars");
        }
        if !seen.insert(art.platform.as_str()) {
            bail!("duplicate `source.binary` platform `{}`", art.platform);
        }
    }
    if let Some(fb) = &source.fallback_git {
        check_binary_platforms(fb)?;
    }
    Ok(())
}

/// Scheme-validate an *untrusted* index `source.git` (mirrors
/// `dora-hub-client::validate_git_url_untrusted`): reject argument-injection and
/// command transports, and local `file://`/absolute paths (a public index must
/// point at a remote).
fn validate_git_url(url: &str) -> eyre::Result<()> {
    const SCHEMES: &[&str] = &["https://", "ssh://", "git@"];
    let ok = SCHEMES.iter().any(|s| url.starts_with(s)) && !url.contains(|c: char| c.is_control());
    if !ok {
        bail!("invalid `source.git` URL (expected a https/ssh/git@ remote)");
    }
    Ok(())
}

/// Recursively collect `*.yml` files under `dir` as `(path, is_symlink)`,
/// including symlinks (even dangling ones) so they are rejected, not skipped.
/// Directory symlinks are not followed.
fn collect_yml(dir: &Path, out: &mut Vec<(PathBuf, bool)>) -> eyre::Result<()> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e.into()),
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let meta = std::fs::symlink_metadata(&path)?;
        let is_yml = path.extension().and_then(|e| e.to_str()) == Some("yml");
        if meta.file_type().is_symlink() {
            if is_yml {
                out.push((path, true));
            }
            continue; // never follow a symlinked dir
        }
        if meta.is_dir() {
            collect_yml(&path, out)?;
        } else if is_yml {
            out.push((path, false));
        }
    }
    Ok(())
}
