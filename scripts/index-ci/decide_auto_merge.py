#!/usr/bin/env python3
"""Decide whether a `node-index/` PR may auto-merge (Hub P2.3, spec §7.5).

The auto-merge bot runs this from the **trusted base checkout** (never the PR's
own copy of these scripts) against the PR's `node-index/` content. It returns
MERGE only for a *routine publish*:

  - the PR touches **only** version files (`node-index/<ns>/<name>/<semver>.yml`),
    never `package.yml`/OWNERS, the policy, the CI, or the scripts/schemas
    themselves (those are human-review categories, §7.5);
  - every change is an add or an append-only edit (no delete/rename — the
    append-only gate enforces the edit *content*, this just refuses the status);
  - no **new namespace** is claimed (new namespaces always get a human, §7.4);
  - every touched entry is authored by an **owner** of its namespace, where the
    OWNERS list is read from the *base* `package.yml` (not the PR's — a PR must
    not be able to add itself as an owner and self-approve).

Anything else is HOLD — left for a human reviewer. Schema/pin validation and the
append-only *content* rule run as separate trusted hard gates in the bot job.

Usage: decide_auto_merge.py --author <login> [--base <ref>]
  Prints `MERGE` and exits 0 when auto-mergeable; prints `HOLD: <reasons>` and
  exits 1 otherwise. Org-membership lookups use `gh api` (needs GH_TOKEN).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_namespace import git, namespaces_at  # noqa: E402  (trusted siblings)

# node-index/<ns>/<name>/<semver>.yml — the only path an auto-merge may touch.
_VERSION_FILE_RE = re.compile(
    r"^node-index/[^/]+/[^/]+/(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:[-+][0-9A-Za-z.-]+)*\.yml$"
)


def changed_files(base: str) -> list[tuple[str, str]]:
    """(status, path) for files changed in merge-base(base, HEAD)..HEAD."""
    merge_base = git("merge-base", base, "HEAD").strip()
    out = git("diff", "--name-status", "-M", merge_base, "HEAD")
    changes: list[tuple[str, str]] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        cols = line.split("\t")
        # rename/copy: "<R|C><score>\told\tnew" — new path is the last column
        changes.append((cols[0][0], cols[-1]))
    return changes


def is_version_file(path: str) -> bool:
    return bool(_VERSION_FILE_RE.match(path))


def owners_at(ref: str, ns: str, name: str) -> list[str]:
    """OWNERS list from the package.yml at `ref` (empty if absent/malformed)."""
    try:
        text = git("show", f"{ref}:node-index/{ns}/{name}/package.yml")
    except subprocess.CalledProcessError:
        return []
    doc = yaml.safe_load(text)
    owners = doc.get("owners") if isinstance(doc, dict) else None
    return [o for o in owners if isinstance(o, str)] if isinstance(owners, list) else []


def is_authorized(author: str, owners: list[str], is_member) -> bool:
    """Author owns the namespace directly, or via an org in the OWNERS list."""
    if author in owners:
        return True
    return any(is_member(o, author) for o in owners if o and o != author)


def gh_is_member(org: str, user: str) -> bool:
    """True if `user` is a public member of org `org` (404/non-org -> False)."""
    r = subprocess.run(
        ["gh", "api", f"/orgs/{org}/members/{user}", "--silent"],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0


def decide(author: str, base: str, is_member=gh_is_member) -> list[str]:
    """Return a list of HOLD reasons; empty list means auto-mergeable.

    All authorization reads (owners, existing namespaces) use `base` — the base
    *tip* — never `merge-base(base, HEAD)`. The PR chooses its own merge-base
    (its branch point), so a former owner could branch from an old commit where
    they were still listed and self-approve. `changed_files` still enumerates
    the PR's own edits via merge-base; that only widens the set (fail-safe)."""
    files = changed_files(base)
    reasons: list[str] = []

    # path + status guard: only added/appended version files may auto-merge
    bad_paths = sorted({p for _, p in files if not is_version_file(p)})
    if bad_paths:
        reasons.append(
            "touches files that need human review (§7.5): " + ", ".join(bad_paths)
        )
    bad_status = sorted({p for s, p in files if s not in ("A", "M")})
    if bad_status:
        reasons.append("deletes/renames a published file: " + ", ".join(bad_status))

    # new-namespace guard (§7.4): a new namespace always gets a human. Compare
    # against the base tip so resurrecting a namespace deleted on base (but
    # present at the branch point) is still flagged new.
    new_ns = namespaces_at("HEAD") - namespaces_at(base)
    if new_ns:
        reasons.append("claims new namespace(s): " + ", ".join(sorted(new_ns)))

    # owner guard (§7.5): every touched version entry must be owner-authored,
    # against the OWNERS list as it exists on the base tip (not the PR, and not
    # the branch point)
    for _, path in files:
        if not is_version_file(path):
            continue
        _, ns, name, _ = path.split("/")
        owners = owners_at(base, ns, name)
        if not owners:
            reasons.append(f"{path}: no owners on the base package.yml")
        elif not is_authorized(author, owners, is_member):
            reasons.append(f"{path}: @{author} is not an owner of {ns}/{name}")

    return reasons


def main() -> int:
    ap = argparse.ArgumentParser()
    default_base = (
        f"origin/{os.environ['GITHUB_BASE_REF']}"
        if os.environ.get("GITHUB_BASE_REF")
        else "origin/main"
    )
    ap.add_argument("--author", required=True)
    ap.add_argument("--base", default=default_base)
    args = ap.parse_args()

    try:
        reasons = decide(args.author, args.base)
    except subprocess.CalledProcessError as e:
        print(f"HOLD: git inspection failed: {e.stderr.strip()}")
        return 1

    if reasons:
        print("HOLD: " + "; ".join(reasons))
        return 1
    print("MERGE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
