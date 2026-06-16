#!/usr/bin/env python3
"""Enforce the append-only rule on `node-index/` (Hub P2.3, spec §7.5).

A published version file is immutable. Across a PR, index CI permits:
  - ADDING new files (new versions, packages, namespaces);
  - on an existing `<version>.yml`, ONLY flipping `yanked` (+ `yank_reason`)
    and/or ADDING late `source.binary` platform entries.

Everything else — deleting/renaming a version file, or any other edit to its
content — fails and needs an index admin. `package.yml` edits (owners/metadata)
are allowed here but flagged for human review (the bot, not this script, gates
auto-merge on them, spec §7.5).

Usage: check_append_only.py [--base <ref>]
  base defaults to origin/$GITHUB_BASE_REF, else origin/main.
Compares `<merge-base(base, HEAD)>..HEAD`.
"""

from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys

import yaml


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout


def changed_index_files(base: str) -> list[tuple[str, str, str | None]]:
    """Return (status, path, old_path) for changed files under node-index/."""
    merge_base = git("merge-base", base, "HEAD").strip()
    out = git("diff", "--name-status", "-M", f"{merge_base}", "HEAD")
    changes: list[tuple[str, str, str | None]] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        cols = line.split("\t")
        status = cols[0]
        if status.startswith("R"):  # rename: R<score>\told\tnew
            old, new = cols[1], cols[2]
            if _under_index(old) or _under_index(new):
                changes.append(("R", new, old))
        else:
            path = cols[1]
            if _under_index(path):
                changes.append((status[0], path, None))
    return changes


def _under_index(path: str) -> bool:
    return path.startswith("node-index/") and path.endswith(".yml")


def _is_version_entry(path: str) -> bool:
    return path.endswith(".yml") and not path.endswith("/package.yml")


def show(ref_path: str) -> dict:
    text = git("show", ref_path)
    doc = yaml.safe_load(text)
    return doc if isinstance(doc, dict) else {}


def allowed_version_edit(old: dict, new: dict) -> str | None:
    """None if the edit is an allowed yank/binary-add; else a reason string."""

    # a yank must carry a non-empty reason (§7.5); otherwise the yank/yank_reason
    # pair is stripped below and any yank state would pass unchecked
    if new.get("yanked") is True:
        reason = new.get("yank_reason")
        if not (isinstance(reason, str) and reason.strip()):
            return "`yanked: true` requires a non-empty `yank_reason`"

    def strip_mutable(entry: dict) -> dict:
        e = copy.deepcopy(entry)
        e.pop("yanked", None)
        e.pop("yank_reason", None)
        return e

    o, n = strip_mutable(old), strip_mutable(new)

    o_src = o.pop("source", {}) or {}
    n_src = n.pop("source", {}) or {}
    o_bins = o_src.pop("binary", []) or []
    n_bins = n_src.pop("binary", []) or []

    # existing binary artifacts must survive unchanged; new ones may be appended
    for b in o_bins:
        if b not in n_bins:
            return "an existing `source.binary` artifact was changed or removed"

    # an appended artifact may only cover a platform not already pinned —
    # re-adding an existing platform with a different url/sha shadows the
    # original (consumers that take the last match would resolve the new one)
    o_platforms = {b.get("platform") for b in o_bins}
    for b in n_bins:
        if b not in o_bins and b.get("platform") in o_platforms:
            return "a `source.binary` platform already pinned was re-added (shadowing)"

    if o_src != n_src:
        return "`source` changed beyond appending binary artifacts"
    if o != n:
        return "the entry changed beyond `yanked`/`yank_reason` or binary additions"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    default_base = (
        f"origin/{os.environ['GITHUB_BASE_REF']}"
        if os.environ.get("GITHUB_BASE_REF")
        else "origin/main"
    )
    ap.add_argument("--base", default=default_base)
    args = ap.parse_args()

    in_actions = os.environ.get("GITHUB_ACTIONS") == "true"

    def emit(level: str, msg: str) -> None:
        print(f"::{level}::{msg}" if in_actions else f"{level}: {msg}")

    try:
        changes = changed_index_files(args.base)
    except subprocess.CalledProcessError as e:
        emit("error", f"git diff against `{args.base}` failed: {e.stderr.strip()}")
        return 1

    errors = 0
    for status, path, old_path in changes:
        if status == "A":
            continue  # adding a new file is always allowed
        if status == "D":
            emit("error", f"{path}: a published index file must not be deleted (append-only)")
            errors += 1
            continue
        if status == "R":
            emit(
                "error",
                f"{old_path} -> {path}: a published index file must not be renamed/moved",
            )
            errors += 1
            continue
        if status == "M":
            if not _is_version_entry(path):
                emit("warning", f"{path}: package metadata changed — needs human review")
                continue
            merge_base = git("merge-base", args.base, "HEAD").strip()
            old = show(f"{merge_base}:{path}")
            new = show(f"HEAD:{path}")
            reason = allowed_version_edit(old, new)
            if reason:
                emit("error", f"{path}: {reason} — published versions are immutable (§7.5)")
                errors += 1
            continue
        emit("error", f"{path}: unexpected change status `{status}`")
        errors += 1

    if errors:
        print(f"\ncheck_append_only: {errors} violation(s)")
        return 1
    print(f"check_append_only: OK ({len(changes)} index change(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
