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

    changed_manifest = {**base, "manifest": {"name": "n", "namespace": "OTHER"}}
    check("changing the manifest rejected", cao.allowed_version_edit(base, changed_manifest) is not None)


if __name__ == "__main__":
    test_validate()
    test_append_only()
    print(f"\nall index-ci tests passed ({PASSED} checks)")
