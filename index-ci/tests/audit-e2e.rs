//! End-to-end tests for the periodic audit subcommands (`reachability`,
//! `integrity-audit`) against a *real* local git source repo and a synthetic
//! catalog — no network. The pinned `git` is the local repo's path and the
//! `rev` is a real commit SHA, so `git fetch`/`git show` run for real.

use std::path::Path;
use std::process::Command;

use dora_index_ci::{integrity, reachability};
use tempfile::TempDir;

fn git(dir: &Path, args: &[&str]) {
    let ok = Command::new("git")
        .args(["-C", dir.to_str().unwrap()])
        .args(args)
        .status()
        .unwrap()
        .success();
    assert!(ok, "git {args:?} failed");
}

/// A local git repo with `dora-node.yml` (and the given body) committed; returns
/// the repo dir and the commit SHA.
fn source_repo(manifest_body: &str) -> (TempDir, String) {
    let dir = tempfile::tempdir().unwrap();
    git(dir.path(), &["init", "-q"]);
    git(dir.path(), &["config", "user.email", "t@example.com"]);
    git(dir.path(), &["config", "user.name", "tester"]);
    std::fs::write(dir.path().join("dora-node.yml"), manifest_body).unwrap();
    git(dir.path(), &["add", "."]);
    git(dir.path(), &["commit", "-q", "-m", "init"]);
    let sha = String::from_utf8(
        Command::new("git")
            .args(["-C", dir.path().to_str().unwrap(), "rev-parse", "HEAD"])
            .output()
            .unwrap()
            .stdout,
    )
    .unwrap()
    .trim()
    .to_owned();
    (dir, sha)
}

fn manifest(name: &str) -> String {
    format!("apiVersion: 1\nname: {name}\nnamespace: acme\nruntime: rust\nentrypoint: e\n")
}

/// Write `node-index/acme/<name>/<ver>.yml` + sibling `package.yml`.
fn write_entry(catalog: &Path, name: &str, ver: &str, entry_yaml: &str) {
    let dir = catalog.join("acme").join(name);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join(format!("{ver}.yml")), entry_yaml).unwrap();
    std::fs::write(dir.join("package.yml"), "owners: [acme]\n").unwrap();
}

fn entry_yaml(name: &str, git_url: &str, rev: &str, yanked: bool, binary: bool) -> String {
    let source = if binary {
        "source:\n  binary:\n    - platform: x86_64-unknown-linux-gnu\n      \
         url: https://example.com/n.tar.gz\n      \
         sha256: 0000000000000000000000000000000000000000000000000000000000000000\n"
            .to_owned()
    } else {
        format!("source:\n  git: {git_url}\n  rev: {rev}\n")
    };
    format!(
        "manifest:\n  apiVersion: 1\n  name: {name}\n  namespace: acme\n  \
         runtime: rust\n  entrypoint: e\n{source}yanked: {yanked}\n"
    )
}

#[test]
fn reachability_passes_for_a_live_pin_and_fails_for_a_rotted_one() {
    let (repo, sha) = source_repo(&manifest("n"));
    let url = repo.path().to_str().unwrap();

    let live = tempfile::tempdir().unwrap();
    write_entry(
        live.path(),
        "n",
        "1.0.0",
        &entry_yaml("n", url, &sha, false, false),
    );
    assert_eq!(reachability::run(live.path()).unwrap(), 0);

    // a force-pushed-away / never-existed commit → unreachable → exit 1
    let rotted = tempfile::tempdir().unwrap();
    let gone = "0".repeat(40);
    write_entry(
        rotted.path(),
        "n",
        "1.0.0",
        &entry_yaml("n", url, &gone, false, false),
    );
    assert_eq!(reachability::run(rotted.path()).unwrap(), 1);
}

#[test]
fn reachability_skips_yanked_and_binary_only_entries() {
    let url = "/this/path/does/not/exist.git";
    let gone = "0".repeat(40);

    // a yanked entry whose source is gone must not fail the run
    let yanked = tempfile::tempdir().unwrap();
    write_entry(
        yanked.path(),
        "n",
        "1.0.0",
        &entry_yaml("n", url, &gone, true, false),
    );
    assert_eq!(reachability::run(yanked.path()).unwrap(), 0);

    // a binary-only entry has no git rev to re-check
    let binary = tempfile::tempdir().unwrap();
    write_entry(
        binary.path(),
        "n",
        "1.0.0",
        &entry_yaml("n", "", "", false, true),
    );
    assert_eq!(reachability::run(binary.path()).unwrap(), 0);
}

#[test]
fn integrity_audit_passes_on_match_and_fails_on_rewritten_entry() {
    let (repo, sha) = source_repo(&manifest("n"));
    let url = repo.path().to_str().unwrap();

    // entry manifest matches what's committed at the pinned commit
    let ok = tempfile::tempdir().unwrap();
    write_entry(
        ok.path(),
        "n",
        "1.0.0",
        &entry_yaml("n", url, &sha, false, false),
    );
    assert_eq!(integrity::run(ok.path(), None).unwrap(), 0);

    // entry claims `name: evil` but the pinned commit's manifest says `name: n`
    let tampered = tempfile::tempdir().unwrap();
    write_entry(
        tampered.path(),
        "n",
        "1.0.0",
        &entry_yaml("evil", url, &sha, false, false),
    );
    assert_eq!(integrity::run(tampered.path(), None).unwrap(), 1);
}

#[test]
fn audits_no_op_on_an_empty_catalog() {
    let empty = tempfile::tempdir().unwrap();
    assert_eq!(reachability::run(empty.path()).unwrap(), 0);
    assert_eq!(integrity::run(empty.path(), None).unwrap(), 0);
}
