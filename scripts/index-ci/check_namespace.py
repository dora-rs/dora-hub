#!/usr/bin/env python3
"""Screen newly-claimed `node-index/` namespaces (Hub P2.3, spec §7.4).

A namespace is claimed by adding `node-index/<ns>/.../package.yml`. Per §7.4
*every* new namespace (one not present in the base branch) gets a human
reviewer — and reserved or confusable claims escalate to an index admin:

  - **Reserved.** `<ns>` on the reserved list belongs to the project / type
    system and needs an index admin (`reserved_namespaces.txt`).
  - **Confusable.** `<ns>` that looks like an existing or reserved namespace —
    homoglyph-normalized equality (`d0ra-rs` == `dora-rs`, `acrne` == `acme`)
    or Levenshtein distance <= 1 — is the dependency-confusion vector and needs
    an index admin.

This is a *routing signal, not a hard gate*: it emits a warning per new claim
and exits 0. A new namespace is never a CI failure — that would force a
legitimate human/admin to bypass a red check instead of approving-and-merging
normally. The auto-merge bot (PR 3) reads `review_tier()` from the trusted base
to decide what to withhold from auto-merge; this run only surfaces it. Routine
publishing *within* an existing namespace produces no warning (it isn't new).

This does NOT verify owner identity (that the PR author owns `<ns>`) — that
needs the trusted GitHub actor + API and lands with the bot (PR 3).

Usage: check_namespace.py [--base <ref>]
  base defaults to origin/$GITHUB_BASE_REF, else origin/main.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess

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


def review_tier(ns: str, existing: set[str], reserved: set[str]) -> tuple[str, str]:
    """Classify a *new* namespace claim. Every new namespace needs at least a
    human reviewer (§7.4) — that is the contract the auto-merge bot enforces by
    withholding auto-merge, NOT a CI failure. Reserved or confusable claims
    escalate to an index admin. Returns (tier, reason), tier in {"admin",
    "human"}."""
    if ns in reserved:
        return "admin", f"namespace `{ns}` is reserved"
    for ref in sorted(existing | reserved):
        if is_confusable(ns, ref):
            return "admin", f"namespace `{ns}` is confusable with `{ref}`"
    return "human", f"namespace `{ns}` is newly claimed"


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
    # A new namespace is never a CI *failure* — it is a routing signal. We emit
    # a warning per claim so the reviewer (and, later, the bot) sees it, and
    # exit 0 so a legitimate human/admin can approve-and-merge normally rather
    # than bypass a red check. The bot (PR 3) reads review_tier() from the
    # trusted base to decide what NOT to auto-merge.
    for ns in new:
        # classify against everything that already existed plus the other new
        # claims in this same PR
        existing = (base_ns | set(new)) - {ns}
        tier, reason = review_tier(ns, existing, reserved)
        who = "an index admin" if tier == "admin" else "a human reviewer"
        emit("warning", f"{reason} — needs {who} before merge (§7.4); not auto-merge")

    if new:
        print(f"check_namespace: {len(new)} new namespace claim(s) flagged for review (not a failure)")
        return 0
    print("check_namespace: OK (no new namespaces)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
