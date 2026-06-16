#!/usr/bin/env python3
"""Screen newly-claimed `node-index/` namespaces (Hub P2.3, spec §7.4).

A namespace is claimed by adding `node-index/<ns>/.../package.yml`. A *new*
claim (a namespace not present in the base branch) must clear two structural
gates before it can auto-merge:

  - **Reserved.** `<ns>` must not be on the reserved list — those belong to the
    project / type system and need an index admin (`reserved_namespaces.txt`).
  - **Confusable.** `<ns>` must not look like an existing namespace (or a
    reserved one): homoglyph-normalized equality (`d0ra-rs` == `dora-rs`,
    `acrne` == `acme`) or Levenshtein distance <= 1. Lookalikes are the
    dependency-confusion vector, so they get mandatory human review.

This does NOT verify owner identity (that the PR author owns `<ns>`) — that
needs the trusted GitHub actor + API and lands with the auto-merge bot (PR 3).

Failing a gate is not "rejected forever": it means the PR needs a human/admin,
not auto-merge. Routine publishing *within* an existing namespace never trips
this (the namespace isn't new).

Usage: check_namespace.py [--base <ref>]
  base defaults to origin/$GITHUB_BASE_REF, else origin/main.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

RESERVED_FILE = pathlib.Path(__file__).resolve().parent / "reserved_namespaces.txt"

# Visually-confusable substitutions, folded before comparison so a homoglyph
# swap and a structural edit can't be *combined* to slip past the gate (e.g.
# `d0rars` = `0`->`o` + a dropped hyphen). Multi-char lookalikes (read
# identically, differ by edit count) first, then single-char swaps. Best-effort,
# not exhaustive — it only has to make the obvious typosquats land on the same
# skeleton, and a miss flags for human review, it never auto-rejects.
_BIGRAM_LOOKALIKES = (("rn", "m"), ("cl", "d"), ("vv", "w"), ("nn", "m"))
_HOMOGLYPH_CHARS = str.maketrans(
    {"0": "o", "1": "l", "2": "z", "3": "e", "5": "s", "6": "b", "8": "b", "9": "g"}
)


def normalize(ns: str) -> str:
    """Collapse a namespace to its visual skeleton: drop separators and fold
    look-alike character groups, so confusable comparison sees through both."""
    s = ns.lower().replace("-", "").replace("_", "")
    for a, b in _BIGRAM_LOOKALIKES:
        s = s.replace(a, b)
    return s.translate(_HOMOGLYPH_CHARS)


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def is_confusable(a: str, b: str) -> bool:
    """True if `a` is a distinct-but-lookalike name for `b`. Compares the visual
    skeletons (not the raw strings) so a homoglyph swap plus a single structural
    edit still collapses to within distance 1."""
    if a == b:
        return False
    na, nb = normalize(a), normalize(b)
    return na == nb or levenshtein(na, nb) <= 1


def screen_namespace(ns: str, existing: set[str], reserved: set[str]) -> str | None:
    """None if `ns` may auto-merge; else a reason it needs human review."""
    if ns in reserved:
        return f"namespace `{ns}` is reserved — needs an index admin (§7.4)"
    for ref in sorted(existing | reserved):
        if is_confusable(ns, ref):
            return (
                f"namespace `{ns}` is confusable with `{ref}` "
                f"— needs human review (§7.4)"
            )
    return None


def load_reserved() -> set[str]:
    out = set()
    for line in RESERVED_FILE.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.add(line)
    return out


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout


def namespaces_at(ref: str) -> set[str]:
    """Namespaces present under node-index/ at `ref` (dirs with a package.yml)."""
    out = git("ls-tree", "-r", "--name-only", ref, "node-index/")
    namespaces: set[str] = set()
    for path in out.splitlines():
        parts = path.split("/")
        # node-index/<ns>/<name>/package.yml
        if len(parts) >= 4 and parts[0] == "node-index" and parts[-1] == "package.yml":
            namespaces.add(parts[1])
    return namespaces


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

    reserved = load_reserved()
    try:
        merge_base = git("merge-base", args.base, "HEAD").strip()
        base_ns = namespaces_at(merge_base)
        head_ns = namespaces_at("HEAD")
    except subprocess.CalledProcessError as e:
        emit("error", f"git inspection against `{args.base}` failed: {e.stderr.strip()}")
        return 1

    new = sorted(head_ns - base_ns)
    errors = 0
    for ns in new:
        # screen against everything that already existed plus the other new
        # claims in this same PR
        existing = (base_ns | set(new)) - {ns}
        reason = screen_namespace(ns, existing, reserved)
        if reason:
            emit("error", reason)
            errors += 1

    if errors:
        print(f"\ncheck_namespace: {errors} namespace claim(s) need human review")
        return 1
    print(f"check_namespace: OK ({len(new)} new namespace(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
