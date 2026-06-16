#!/usr/bin/env python3
"""Self-contained tests for the index-CI scripts (no pytest needed).

Run: `python3 scripts/index-ci/tests/test_index_ci.py`
Exits non-zero on the first failure.
"""

from __future__ import annotations

import contextlib
import io
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
SCRIPTS = HERE.parent
sys.path.insert(0, str(SCRIPTS))

# point the validator at the fixture tree before importing it
os.environ["DORA_INDEX_ROOT"] = str(FIXTURES)

import check_append_only as cao  # noqa: E402
import check_namespace as cns  # noqa: E402
import decide_auto_merge as dam  # noqa: E402
import validate_entries as ve  # noqa: E402

PASSED = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASSED
    if cond:
        PASSED += 1
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        raise SystemExit(1)


def run_validate(*relpaths: str) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = ve.main([str(FIXTURES / p) for p in relpaths])
    return rc, buf.getvalue()


def test_validate() -> None:
    print("validate_entries:")
    rc, _ = run_validate(
        "dora-rs/example-node/0.1.0.yml", "dora-rs/example-node/package.yml"
    )
    check("good entry + package pass", rc == 0)

    rc, out = run_validate("badrev/example-node/0.1.0.yml")
    check("short rev rejected", rc == 1 and "rev" in out, out)

    rc, out = run_validate("mismatch/example-node/0.1.0.yml")
    check("namespace/path mismatch rejected", rc == 1 and "directory" in out, out)

    rc, out = run_validate("dora-rs/example-node/latest.yml")
    check("non-semver filename rejected", rc == 1 and "semver" in out, out)

    rc, out = run_validate("noowners/example-node/package.yml")
    check("package without owners rejected", rc == 1 and "owners" in out, out)

    rc, out = run_validate("std/example-node/package.yml")
    check("reserved namespace warns but passes", rc == 0 and "reserved" in out, out)

    rc, out = run_validate("nopkg/example-node/0.1.0.yml")
    check("version without sibling package.yml rejected", rc == 1 and "package.yml" in out, out)

    rc, out = run_validate("dupplat/example-node/0.1.0.yml")
    check("duplicate binary platform rejected", rc == 1 and "duplicate" in out and "platform" in out, out)


def test_empty_binary_rejected() -> None:
    print("node-index-entry.schema empty-binary guard:")
    check("git+rev source accepted", _source_ok({"git": "g", "rev": "f" * 40}))
    check("non-empty binary source accepted", _source_ok({"binary": [{"platform": "x", "url": "u", "sha256": "a" * 64}]}))
    # an empty binary list is not a usable source — must not satisfy the anyOf
    check("empty binary list rejected", not _source_ok({"binary": []}))


def test_symlink_rejected() -> None:
    import tempfile

    print("validate_entries symlink guard:")
    with tempfile.TemporaryDirectory() as td:
        tdp = pathlib.Path(td)
        outside = tdp / "secret.yml"
        outside.write_text("manifest: {}\n")
        entry = tdp / "index" / "ns" / "pkg"
        entry.mkdir(parents=True)
        # an escaping symlink and a dangling one (missing target) — both must be
        # rejected, not silently skipped by the containment / exists() guards
        link = entry / "1.0.0.yml"
        link.symlink_to(outside)
        dangling = entry / "2.0.0.yml"
        dangling.symlink_to(tdp / "does-not-exist.yml")
        saved = ve.INDEX_ROOT
        ve.INDEX_ROOT = tdp / "index"
        try:
            # explicit-arg path
            for label, target in (("escaping", link), ("dangling", dangling)):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = ve.main([str(target)])
                check(f"{label} symlink rejected (explicit)", rc == 1 and "symlink" in buf.getvalue(), buf.getvalue())
            # whole-tree scan must surface both, not skip them
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = ve.main([])
            out = buf.getvalue()
            check("dangling symlink caught by tree scan", rc == 1 and out.count("symlink") >= 2, out)
        finally:
            ve.INDEX_ROOT = saved


def test_append_only() -> None:
    print("check_append_only.allowed_version_edit:")
    base = {
        "manifest": {"name": "n", "namespace": "ns"},
        "source": {"git": "g", "rev": "r", "binary": [{"platform": "x", "url": "u", "sha256": "s"}]},
        "published": "t",
        "yanked": False,
    }

    yanked = {**base, "yanked": True, "yank_reason": "broken"}
    check("yank flip allowed", cao.allowed_version_edit(base, yanked) is None)

    yanked_no_reason = {**base, "yanked": True}
    check("yank without reason rejected", cao.allowed_version_edit(base, yanked_no_reason) is not None)

    yanked_blank_reason = {**base, "yanked": True, "yank_reason": "   "}
    check("yank with blank reason rejected", cao.allowed_version_edit(base, yanked_blank_reason) is not None)

    unyank = {**base, "yanked": False}
    check("no-op allowed", cao.allowed_version_edit(base, unyank) is None)

    added_bin = {
        **base,
        "source": {**base["source"], "binary": base["source"]["binary"] + [{"platform": "y", "url": "u2", "sha256": "s2"}]},
    }
    check("appending a binary allowed", cao.allowed_version_edit(base, added_bin) is None)

    changed_rev = {**base, "source": {**base["source"], "rev": "OTHER"}}
    check("changing rev rejected", cao.allowed_version_edit(base, changed_rev) is not None)

    changed_bin = {
        **base,
        "source": {**base["source"], "binary": [{"platform": "x", "url": "EVIL", "sha256": "s"}]},
    }
    check("mutating an existing binary rejected", cao.allowed_version_edit(base, changed_bin) is not None)

    shadow_bin = {
        **base,
        "source": {**base["source"], "binary": base["source"]["binary"] + [{"platform": "x", "url": "EVIL", "sha256": "s2"}]},
    }
    check("re-adding a pinned platform (shadowing) rejected", cao.allowed_version_edit(base, shadow_bin) is not None)

    changed_manifest = {**base, "manifest": {"name": "n", "namespace": "OTHER"}}
    check("changing the manifest rejected", cao.allowed_version_edit(base, changed_manifest) is not None)


def _source_ok(source: dict) -> bool:
    """Validate a `source` stanza against the real entry schema (refs resolved)."""
    import jsonschema

    schema = ve.load_schema("node-index-entry.schema.json")
    doc = {
        "manifest": {
            "apiVersion": 1,
            "name": "n",
            "namespace": "ns",
            "runtime": "python",
            "entrypoint": "x:main",
        },
        "source": source,
    }
    return not list(jsonschema.Draft7Validator(schema).iter_errors(doc))


def test_subdir_no_traversal() -> None:
    print("node-index-entry.schema subdir traversal guard:")

    def subdir_ok(value: str) -> bool:
        return _source_ok({"git": "g", "rev": "f" * 40, "subdir": value})

    check("plain subdir accepted", subdir_ok("node-hub/dora-yolo"))
    check("leading-slash subdir rejected", not subdir_ok("/etc"))
    check("dotdot subdir rejected", not subdir_ok("../etc"))
    # the `..` guard must not be defeated by smuggling it onto a second line
    check("newline-smuggled dotdot rejected", not subdir_ok("ok\n../../etc"))


def test_namespace_screening() -> None:
    print("check_namespace.review_tier:")
    existing = {"dora-rs", "acme"}
    reserved = {"dora", "dora-rs", "std", "hub"}

    def tier(ns: str) -> str:
        return cns.review_tier(ns, existing - {ns}, reserved)[0]

    def reason(ns: str) -> str:
        return cns.review_tier(ns, existing - {ns}, reserved)[1]

    # every new namespace needs at least a human reviewer — never auto-merge
    check("fresh distinct namespace needs human review", tier("widgetworks") == "human")
    check("distinct longer name needs human review", tier("acme-robotics") == "human")
    # only RESERVED claims escalate to an index admin (§7.4)
    check("reserved namespace needs admin", tier("std") == "admin")
    # confusables get mandatory HUMAN review (not admin), but flagged in the
    # reason so the detection is still proven (not silently defaulting to "new")
    check("homoglyph of reserved is human review", tier("d0ra") == "human")
    check("homoglyph of reserved is flagged confusable", "confusable" in reason("d0ra"))
    check("homoglyph of existing is human review (0->o)", tier("d0ra-rs") == "human")
    check("homoglyph of existing flagged", "confusable" in reason("d0ra-rs"))
    check("rn->m confusable flagged", "confusable" in reason("acrne"))
    check("edit-distance-1 of existing flagged", "confusable" in reason("acme2"))

    # homoglyph + structural edit must not compose past the gate (the d0rars
    # class): a 0->o swap AND a dropped/extra hyphen is still one skeleton
    check("homoglyph + dropped hyphen flagged", "confusable" in reason("d0rars"))
    check("homoglyph + double hyphen flagged", "confusable" in reason("d0ra--rs"))
    check("dropped hyphen alone flagged", "confusable" in reason("dorars"))
    # bigram lookalikes: cl->d, vv->w, nn->m
    check("cl->d homoglyph flagged", "confusable" in reason("clora-rs"))

    # the homoglyph normalizer and metric the gate is built on
    check("normalize collapses 0/1/3/5 and rn", cns.normalize("d0rn3") == "dome")
    check("levenshtein basic", cns.levenshtein("acme", "acme2") == 1)
    check("levenshtein rn vs m is 2", cns.levenshtein("acrne", "acme") == 2)


def test_decide_auto_merge() -> None:
    import subprocess as sp
    import tempfile

    print("decide_auto_merge (path/owner guards):")
    check("version file accepted", dam.is_version_file("node-index/acme/widget/1.2.3.yml"))
    check("prerelease semver accepted", dam.is_version_file("node-index/acme/widget/1.0.0-rc.1.yml"))
    check("package.yml is not a version file", not dam.is_version_file("node-index/acme/widget/package.yml"))
    check("nested path rejected", not dam.is_version_file("node-index/acme/widget/sub/1.0.0.yml"))
    check("non-semver rejected", not dam.is_version_file("node-index/acme/widget/latest.yml"))
    check("outside node-index rejected", not dam.is_version_file("node-hub/acme/main.py"))

    member = {("acme-org", "alice")}
    is_member = lambda org, user: (org, user) in member  # noqa: E731
    check("direct owner authorized", dam.is_authorized("alice", ["alice"], is_member))
    check("non-owner rejected", not dam.is_authorized("mallory", ["alice"], is_member))
    check("org-member authorized", dam.is_authorized("alice", ["acme-org"], is_member))
    check("non-member of owner-org rejected", not dam.is_authorized("bob", ["acme-org"], is_member))

    print("decide_auto_merge (real two-commit repo):")
    no_org = lambda org, user: False  # noqa: E731
    with tempfile.TemporaryDirectory() as td:
        repo = pathlib.Path(td)
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
        }

        def g(*a: str) -> None:
            sp.run(["git", *a], cwd=repo, env=env, check=True, capture_output=True)

        def commit(msg: str) -> None:
            g("add", "-A")
            g("commit", "-qm", msg)

        entry = "manifest: {}\nsource: {git: g, rev: r}\n"
        g("init", "-q", "-b", "main")
        pkg = repo / "node-index" / "acme" / "widget"
        pkg.mkdir(parents=True)
        (pkg / "package.yml").write_text("owners:\n  - alice\n")
        (pkg / "1.0.0.yml").write_text(entry)
        commit("base")
        base = sp.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, env=env, capture_output=True, text=True
        ).stdout.strip()

        saved = os.getcwd()
        os.chdir(repo)
        try:
            # routine publish by the owner -> MERGE
            (pkg / "1.1.0.yml").write_text(entry)
            commit("publish 1.1.0")
            check("owner publish -> MERGE", dam.decide("alice", base, no_org) == [])
            check(
                "non-owner publish -> HOLD",
                any("not an owner" in r for r in dam.decide("mallory", base, no_org)),
            )

            # new namespace claim -> HOLD (new ns + a non-version package.yml)
            g("checkout", "-q", base)
            g("checkout", "-q", "-b", "newns")
            newpkg = repo / "node-index" / "other" / "thing"
            newpkg.mkdir(parents=True)
            (newpkg / "package.yml").write_text("owners:\n  - alice\n")
            (newpkg / "0.1.0.yml").write_text(entry)
            commit("new namespace")
            holds = dam.decide("alice", base, no_org)
            check("new namespace -> HOLD", any("new namespace" in r for r in holds))

            # editing OWNERS/package.yml -> HOLD (non-version path)
            g("checkout", "-q", base)
            g("checkout", "-q", "-b", "owners")
            (pkg / "package.yml").write_text("owners:\n  - alice\n  - mallory\n")
            commit("add owner")
            holds = dam.decide("alice", base, no_org)
            check("OWNERS edit -> HOLD", any("human review" in r for r in holds))
        finally:
            os.chdir(saved)


if __name__ == "__main__":
    test_validate()
    test_append_only()
    test_namespace_screening()
    test_decide_auto_merge()
    test_subdir_no_traversal()
    test_empty_binary_rejected()
    test_symlink_rejected()
    print(f"\nall index-ci tests passed ({PASSED} checks)")
