//! Integration tests for the index-CI checks. Pure-function tests run in
//! parallel; the `decide` tests use a real git repo and serialize cwd changes.

use std::collections::BTreeSet;
use std::path::Path;
use std::process::Command;
use std::sync::Mutex;

use dora_index_ci::append_only::allowed_version_edit;
use dora_index_ci::decide::{decide, is_authorized, is_version_file};
use dora_index_ci::namespace::{Tier, is_confusable, levenshtein, normalize, review_tier};
use dora_index_ci::validate::validate_source;
use serde_yaml::Value;

// ---- namespace governance ----

fn ns_set(items: &[&str]) -> BTreeSet<String> {
    items.iter().map(|s| s.to_string()).collect()
}

#[test]
fn namespace_review_tiers() {
    let existing = ns_set(&["dora-rs", "acme"]);
    let reserved = ns_set(&["dora", "dora-rs", "std", "hub"]);
    let tier = |ns: &str| {
        let mut e = existing.clone();
        e.remove(ns);
        review_tier(ns, &e, &reserved)
    };

    // every new namespace needs at least a human reviewer
    assert_eq!(tier("widgetworks").0, Tier::Human);
    assert_eq!(tier("acme-robotics").0, Tier::Human);
    // only RESERVED claims escalate to an index admin
    assert_eq!(tier("std").0, Tier::Admin);
    // confusables are human review (flagged), not admin
    let (t, why) = tier("d0ra-rs");
    assert_eq!(t, Tier::Human);
    assert!(why.contains("confusable"), "{why}");
    assert_eq!(tier("acrne").0, Tier::Human);
    assert!(tier("acrne").1.contains("confusable"));
}

#[test]
fn confusable_homoglyph_plus_edit_does_not_escape() {
    // the d0rars class: a 0->o swap AND a dropped/extra hyphen is one skeleton
    for squat in ["d0rars", "d0ra--rs", "dorars", "clora-rs"] {
        assert!(
            is_confusable(squat, "dora-rs"),
            "{squat} should be confusable"
        );
    }
    assert!(!is_confusable("acme-robotics", "acme"));
    assert!(!is_confusable("widgetworks", "dora-rs"));
    // a `.`-for-`-` swap is a lookalike too (the resolver key charset allows `.`)
    assert!(is_confusable("dora.rs", "dora-rs"));
    assert_eq!(normalize("d0rn3"), "dome");
    assert_eq!(levenshtein("acme", "acme2"), 1);
}

fn validate_entry(entry: &str) -> i32 {
    let tmp = tempfile::tempdir().unwrap();
    let pkg = tmp.path().join("acme/widget");
    std::fs::create_dir_all(&pkg).unwrap();
    std::fs::write(pkg.join("package.yml"), "owners:\n  - alice\n").unwrap();
    std::fs::write(pkg.join("1.0.0.yml"), entry).unwrap();
    dora_index_ci::validate::run(tmp.path()).unwrap()
}

#[test]
fn validate_rejects_schema_violations() {
    let full = "a".repeat(40);
    assert_eq!(validate_entry(FULL_ENTRY), 0, "clean entry passes");
    // HIGH-1: required manifest fields
    let no_runtime = format!(
        "manifest:\n  apiVersion: 1\n  name: widget\n  namespace: acme\n  entrypoint: x:main\nsource:\n  git: https://x/y\n  rev: {full}\n"
    );
    assert_eq!(validate_entry(&no_runtime), 1, "missing runtime rejected");
    assert_eq!(
        validate_entry(&FULL_ENTRY.replace("apiVersion: 1", "apiVersion: 2")),
        1,
        "apiVersion != 1"
    );
    assert_eq!(
        validate_entry(&FULL_ENTRY.replace("entrypoint: widget:main", "entrypoint: ''")),
        1,
        "empty entrypoint"
    );
    // HIGH-2: binary artifact value checks
    let bad_sha = "manifest:\n  apiVersion: 1\n  name: widget\n  namespace: acme\n  runtime: python\n  entrypoint: x:main\nsource:\n  binary:\n    - platform: linux\n      url: https://x/a\n      sha256: NOTAHASH\n";
    assert_eq!(validate_entry(bad_sha), 1, "bad binary sha256 rejected");
}

fn validate_pkg(pkg_yaml: &str) -> i32 {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("acme/widget");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("package.yml"), pkg_yaml).unwrap();
    dora_index_ci::validate::run(tmp.path()).unwrap()
}

#[test]
fn validate_package_metadata() {
    assert_eq!(
        validate_pkg("owners:\n  - alice\n"),
        0,
        "valid owner passes"
    );
    assert_eq!(validate_pkg("owners: []\n"), 1, "no owners rejected");
    // owners are decide's merge authority — each must be a valid GitHub login
    assert_eq!(
        validate_pkg("owners:\n  - bad/name\n"),
        1,
        "malformed owner rejected"
    );
    // package.yml is the owner authority — unknown fields must not slip in
    assert_eq!(
        validate_pkg("owners:\n  - alice\nbogus: 1\n"),
        1,
        "unknown field rejected"
    );
}

// ---- append-only ----

fn val(yaml: &str) -> Value {
    serde_yaml::from_str(yaml).unwrap()
}

const BASE_ENTRY: &str = "manifest:\n  name: n\n  namespace: ns\nsource:\n  git: g\n  rev: r\n  binary:\n    - platform: x\n      url: u\n      sha256: s\npublished: t\nyanked: false\n";

#[test]
fn append_only_edits() {
    let base = val(BASE_ENTRY);

    let yanked =
        val(&format!("{BASE_ENTRY}yank_reason: broken\n").replace("yanked: false", "yanked: true"));
    assert!(
        allowed_version_edit(&base, &yanked).is_none(),
        "yank+reason allowed"
    );

    let yank_no_reason = val(&BASE_ENTRY.replace("yanked: false", "yanked: true"));
    assert!(
        allowed_version_edit(&base, &yank_no_reason).is_some(),
        "yank without reason rejected"
    );

    let blank =
        val(&format!("{BASE_ENTRY}yank_reason: '   '\n").replace("yanked: false", "yanked: true"));
    assert!(
        allowed_version_edit(&base, &blank).is_some(),
        "blank reason rejected"
    );

    assert!(
        allowed_version_edit(&base, &base).is_none(),
        "no-op allowed"
    );

    let added_bin = val(&BASE_ENTRY.replace(
        "    - platform: x\n      url: u\n      sha256: s\n",
        "    - platform: x\n      url: u\n      sha256: s\n    - platform: y\n      url: u2\n      sha256: s2\n",
    ));
    assert!(
        allowed_version_edit(&base, &added_bin).is_none(),
        "appending a new platform allowed"
    );

    let changed_rev = val(&BASE_ENTRY.replace("rev: r", "rev: OTHER"));
    assert!(
        allowed_version_edit(&base, &changed_rev).is_some(),
        "changing rev rejected"
    );

    let mutated_bin =
        val(&BASE_ENTRY.replace("url: u\n      sha256: s", "url: EVIL\n      sha256: s"));
    assert!(
        allowed_version_edit(&base, &mutated_bin).is_some(),
        "mutating a binary rejected"
    );

    let shadow = val(&BASE_ENTRY.replace(
        "    - platform: x\n      url: u\n      sha256: s\n",
        "    - platform: x\n      url: u\n      sha256: s\n    - platform: x\n      url: EVIL\n      sha256: s2\n",
    ));
    assert!(
        allowed_version_edit(&base, &shadow).is_some(),
        "shadowing a pinned platform rejected"
    );

    let changed_manifest = val(&BASE_ENTRY.replace("namespace: ns", "namespace: OTHER"));
    assert!(
        allowed_version_edit(&base, &changed_manifest).is_some(),
        "changing the manifest rejected"
    );
}

// ---- source validation ----

fn source(yaml: &str) -> dora_index_ci::model::SourceSpec {
    serde_yaml::from_str(yaml).unwrap()
}

#[test]
fn source_validation() {
    let full = "a".repeat(40);
    assert!(validate_source(&source(&format!("git: https://x/y\nrev: {full}\n"))).is_ok());
    assert!(
        validate_source(&source("git: https://x/y\nrev: abc1234\n")).is_err(),
        "short rev"
    );
    assert!(
        validate_source(&source("git: ext::sh -c evil\nrev: ")).is_err(),
        "hostile git url"
    );
    assert!(
        validate_source(&source("binary: []\n")).is_err(),
        "empty binary = no source"
    );
    assert!(
        validate_source(&source(&format!(
            "git: https://x/y\nrev: {full}\nbinary: []\n"
        )))
        .is_err(),
        "a present-but-empty binary list is rejected even with a git pin"
    );
    let hash = "a".repeat(64);
    assert!(
        validate_source(&source(&format!(
            "binary:\n  - platform: linux\n    url: u\n    sha256: {hash}\n"
        )))
        .is_ok(),
        "non-empty binary ok"
    );
    // subdir traversal, incl. newline-smuggled `..`
    let with_subdir = |s: &str| format!("git: https://x/y\nrev: {full}\nsubdir: {s}\n");
    assert!(validate_source(&source(&with_subdir("node-hub/x"))).is_ok());
    assert!(validate_source(&source(&with_subdir("../etc"))).is_err());
    assert!(validate_source(&source(&with_subdir("\"ok\\n../../etc\""))).is_err());
    // duplicate binary platform (both artifacts otherwise valid)
    assert!(
        validate_source(&source(&format!(
            "binary:\n  - platform: linux\n    url: a\n    sha256: {hash}\n  - platform: linux\n    url: b\n    sha256: {hash}\n"
        )))
        .is_err(),
        "duplicate platform"
    );
    // fallback-git must itself be a valid source (recursive, full-hash pin)
    let with_fallback = |fb: &str| {
        format!("binary:\n  - platform: linux\n    url: u\n    sha256: {hash}\nfallback-git:\n{fb}")
    };
    assert!(
        validate_source(&source(&with_fallback("  rev: main\n"))).is_err(),
        "mutable/incomplete fallback-git rejected"
    );
    assert!(
        validate_source(&source(&with_fallback(&format!(
            "  git: https://x/y\n  rev: {full}\n"
        ))))
        .is_ok(),
        "valid fallback-git accepted"
    );
}

// ---- validate: catalog walk ----

const FULL_ENTRY: &str = "manifest:\n  apiVersion: 1\n  name: widget\n  namespace: acme\n  runtime: python\n  entrypoint: widget:main\nsource:\n  git: https://github.com/dora-rs/dora-hub\n  rev: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n";

#[test]
fn validate_accepts_clean_catalog_and_rejects_symlinks() {
    let tmp = tempfile::tempdir().unwrap();
    let pkg = tmp.path().join("acme/widget");
    std::fs::create_dir_all(&pkg).unwrap();
    std::fs::write(pkg.join("package.yml"), "owners:\n  - alice\n").unwrap();
    std::fs::write(pkg.join("1.0.0.yml"), FULL_ENTRY).unwrap();
    assert_eq!(
        dora_index_ci::validate::run(tmp.path()).unwrap(),
        0,
        "clean catalog passes"
    );

    // a dangling symlinked entry must be rejected, not skipped
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(tmp.path().join("missing.yml"), pkg.join("2.0.0.yml")).unwrap();
        assert_eq!(
            dora_index_ci::validate::run(tmp.path()).unwrap(),
            1,
            "symlink rejected"
        );
    }
}

// ---- decide: pure helpers ----

#[test]
fn decide_path_and_owner_guards() {
    assert!(is_version_file("node-index/acme/widget/1.2.3.yml"));
    assert!(is_version_file("node-index/acme/widget/1.0.0-rc.1.yml"));
    assert!(!is_version_file("node-index/acme/widget/package.yml"));
    assert!(!is_version_file("node-index/acme/widget/sub/1.0.0.yml"));
    assert!(!is_version_file("node-index/acme/widget/latest.yml"));
    assert!(!is_version_file("node-hub/acme/main.py"));

    let member = |org: &str, user: &str| org == "acme-org" && user == "alice";
    let owners = |s: &[&str]| s.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    assert!(is_authorized("alice", &owners(&["alice"]), &member));
    assert!(!is_authorized("mallory", &owners(&["alice"]), &member));
    assert!(is_authorized("alice", &owners(&["acme-org"]), &member));
    assert!(!is_authorized("bob", &owners(&["acme-org"]), &member));
}

// ---- decide: real git repo ----

static CWD_LOCK: Mutex<()> = Mutex::new(());

fn git(dir: &Path, args: &[&str]) {
    let out = Command::new("git")
        .current_dir(dir)
        .args(args)
        .env("GIT_AUTHOR_NAME", "t")
        .env("GIT_AUTHOR_EMAIL", "t@t")
        .env("GIT_COMMITTER_NAME", "t")
        .env("GIT_COMMITTER_EMAIL", "t@t")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn rev_parse(dir: &Path, what: &str) -> String {
    let out = Command::new("git")
        .current_dir(dir)
        .args(["rev-parse", what])
        .output()
        .unwrap();
    String::from_utf8_lossy(&out.stdout).trim().to_owned()
}

#[test]
fn append_only_flags_rename_out_of_index() {
    let tmp = tempfile::tempdir().unwrap();
    let repo = tmp.path();
    let pkg = repo.join("node-index/acme/widget");
    std::fs::create_dir_all(&pkg).unwrap();
    std::fs::write(pkg.join("package.yml"), "owners:\n  - alice\n").unwrap();
    std::fs::write(pkg.join("1.0.0.yml"), ENTRY).unwrap();
    git(repo, &["init", "-q", "-b", "main"]);
    git(repo, &["add", "-A"]);
    git(repo, &["commit", "-qm", "base"]);
    let base = rev_parse(repo, "HEAD");

    let _guard = CWD_LOCK.lock().unwrap();
    let saved = std::env::current_dir().unwrap();
    std::env::set_current_dir(repo).unwrap();
    let result = std::panic::catch_unwind(|| {
        // rename a published version file OUT of node-index/ — a silent removal
        std::fs::create_dir_all(repo.join("docs")).unwrap();
        git(
            repo,
            &["mv", "node-index/acme/widget/1.0.0.yml", "docs/moved.yml"],
        );
        git(repo, &["commit", "-qm", "rename out of index"]);
        assert_eq!(
            dora_index_ci::append_only::run(&base).unwrap(),
            1,
            "renaming a published file out of node-index/ is an append-only violation"
        );
    });
    std::env::set_current_dir(saved).unwrap();
    result.unwrap();
}

const ENTRY: &str = "manifest:\n  name: widget\n  namespace: acme\nsource:\n  git: g\n  rev: r\n";

#[test]
fn decide_integration() {
    let no_org = |_: &str, _: &str| false;
    let tmp = tempfile::tempdir().unwrap();
    let repo = tmp.path();
    let pkg = repo.join("node-index/acme/widget");
    std::fs::create_dir_all(&pkg).unwrap();
    std::fs::write(pkg.join("package.yml"), "owners:\n  - alice\n").unwrap();
    std::fs::write(pkg.join("1.0.0.yml"), ENTRY).unwrap();
    git(repo, &["init", "-q", "-b", "main"]);
    git(repo, &["add", "-A"]);
    git(repo, &["commit", "-qm", "base"]);
    let base = rev_parse(repo, "HEAD");

    let _guard = CWD_LOCK.lock().unwrap();
    let saved = std::env::current_dir().unwrap();
    std::env::set_current_dir(repo).unwrap();
    let result = std::panic::catch_unwind(|| {
        // routine publish by the owner -> MERGE
        std::fs::write(pkg.join("1.1.0.yml"), ENTRY).unwrap();
        git(repo, &["add", "-A"]);
        git(repo, &["commit", "-qm", "publish 1.1.0"]);
        assert!(
            decide("alice", &base, &no_org).unwrap().is_empty(),
            "owner publish -> MERGE"
        );
        assert!(
            decide("mallory", &base, &no_org)
                .unwrap()
                .iter()
                .any(|r| r.contains("not an owner")),
            "non-owner -> HOLD"
        );

        // new namespace -> HOLD (new ns + non-version package.yml)
        git(repo, &["checkout", "-q", &base]);
        git(repo, &["checkout", "-q", "-b", "newns"]);
        let other = repo.join("node-index/other/thing");
        std::fs::create_dir_all(&other).unwrap();
        std::fs::write(other.join("package.yml"), "owners:\n  - alice\n").unwrap();
        std::fs::write(other.join("0.1.0.yml"), ENTRY).unwrap();
        git(repo, &["add", "-A"]);
        git(repo, &["commit", "-qm", "new ns"]);
        assert!(
            decide("alice", &base, &no_org)
                .unwrap()
                .iter()
                .any(|r| r.contains("new namespace")),
            "new namespace -> HOLD"
        );

        // OWNERS edit -> HOLD (non-version path)
        git(repo, &["checkout", "-q", &base]);
        git(repo, &["checkout", "-q", "-b", "owners"]);
        std::fs::write(pkg.join("package.yml"), "owners:\n  - alice\n  - mallory\n").unwrap();
        git(repo, &["add", "-A"]);
        git(repo, &["commit", "-qm", "add owner"]);
        assert!(
            decide("alice", &base, &no_org)
                .unwrap()
                .iter()
                .any(|r| r.contains("human review")),
            "OWNERS edit -> HOLD"
        );

        // CRITICAL-1: a FORMER owner branching from an old commit must HOLD —
        // authorization reads the base tip, not the attacker-chosen merge-base
        git(repo, &["checkout", "-q", "main"]);
        let hist = repo.join("node-index/hist/tool");
        std::fs::create_dir_all(&hist).unwrap();
        std::fs::write(hist.join("package.yml"), "owners:\n  - mallory\n").unwrap();
        std::fs::write(hist.join("1.0.0.yml"), ENTRY).unwrap();
        git(repo, &["add", "-A"]);
        git(repo, &["commit", "-qm", "hist: mallory owns"]);
        let old = rev_parse(repo, "HEAD");
        std::fs::write(hist.join("package.yml"), "owners:\n  - alice\n").unwrap();
        git(repo, &["add", "-A"]);
        git(repo, &["commit", "-qm", "hist: transferred to alice"]);
        let new_tip = rev_parse(repo, "HEAD");
        git(repo, &["checkout", "-q", &old]);
        git(repo, &["checkout", "-q", "-b", "hist-attack"]);
        std::fs::write(hist.join("2.0.0.yml"), ENTRY).unwrap();
        git(repo, &["add", "-A"]);
        git(repo, &["commit", "-qm", "republish from old branch point"]);
        assert!(
            decide("mallory", &new_tip, &no_org)
                .unwrap()
                .iter()
                .any(|r| r.contains("not an owner")),
            "former owner from old commit -> HOLD"
        );
    });
    std::env::set_current_dir(saved).unwrap();
    result.unwrap();
}
