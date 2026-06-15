#!/usr/bin/env python3
"""Validate dora-hub `node-index/` catalog entries (Hub P2.3, spec §7).

For each catalog file:
  - `<ns>/<name>/<version>.yml`  -> a version entry (manifest + source pointer)
  - `<ns>/<name>/package.yml`     -> package metadata (description, repo, owners)

it checks the JSON schema, the path<->content consistency (namespace/name match
the directory, filename is valid semver), and the key-part charset. Reserved
namespaces are reported (they need an admin, spec §7.4) but not hard-failed here
— that gating is the bot's job (PR 3).

Usage:
  validate_entries.py [FILE ...]      # validate the given files
  validate_entries.py                 # scan the whole node-index/ tree

Exits non-zero if any file fails. Designed to run in CI over the PR's changed
files, and locally over fixtures.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import sys

import jsonschema
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCHEMA_DIR = REPO_ROOT / "schemas"
# Overridable for tests (point at a fixture tree); defaults to the real catalog.
INDEX_ROOT = pathlib.Path(os.environ.get("DORA_INDEX_ROOT", REPO_ROOT / "node-index"))
RESERVED_FILE = pathlib.Path(__file__).resolve().parent / "reserved_namespaces.txt"

# X.Y.Z with optional -prerelease and +build (a pragmatic semver 2.0 subset;
# the CLI's `semver` crate is the authority, this is the CI gate).
SEMVER_RE = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)
NAMESPACE_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,38}$")
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


def load_reserved() -> set[str]:
    out = set()
    for line in RESERVED_FILE.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.add(line)
    return out


def load_schema(name: str) -> dict:
    return json.loads((SCHEMA_DIR / name).read_text())


def rel(path: pathlib.Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def index_files() -> list[pathlib.Path]:
    return sorted(p for p in INDEX_ROOT.rglob("*.yml") if p.is_file())


def validate_file(
    path: pathlib.Path,
    entry_schema: dict,
    package_schema: dict,
    reserved: set[str],
    errors: list[str],
    warnings: list[str],
) -> None:
    rp = rel(path)
    try:
        parts = path.resolve().relative_to(INDEX_ROOT).parts
    except ValueError:
        # not under node-index/ — nothing to validate
        return
    if len(parts) != 3:
        errors.append(
            f"{rp}: catalog files must live at node-index/<ns>/<name>/<file>.yml"
        )
        return
    namespace, name, filename = parts

    if not NAMESPACE_RE.match(namespace):
        errors.append(f"{rp}: invalid namespace `{namespace}` (want [a-z0-9-], <=39)")
    if not NAME_RE.match(name):
        errors.append(f"{rp}: invalid package name `{name}` (want [a-z0-9._-], <=64)")

    try:
        doc = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        errors.append(f"{rp}: not valid YAML: {e}")
        return
    if not isinstance(doc, dict):
        errors.append(f"{rp}: expected a YAML mapping")
        return

    if filename == "package.yml":
        _schema_check(package_schema, doc, rp, errors)
        if namespace in reserved:
            warnings.append(
                f"{rp}: namespace `{namespace}` is reserved — needs an index admin "
                f"(spec §7.4), not auto-merge"
            )
        return

    # version entry: filename stem must be semver
    stem = filename[:-4] if filename.endswith(".yml") else filename
    if not SEMVER_RE.match(stem):
        errors.append(f"{rp}: filename `{filename}` is not a `<semver>.yml`")
    _schema_check(entry_schema, doc, rp, errors)

    manifest = doc.get("manifest", {})
    if isinstance(manifest, dict):
        if manifest.get("namespace") != namespace:
            errors.append(
                f"{rp}: manifest.namespace `{manifest.get('namespace')}` "
                f"!= directory `{namespace}`"
            )
        if manifest.get("name") != name:
            errors.append(
                f"{rp}: manifest.name `{manifest.get('name')}` != directory `{name}`"
            )


def _schema_check(schema: dict, doc: dict, rp: str, errors: list[str]) -> None:
    validator = jsonschema.Draft7Validator(schema)
    for err in sorted(validator.iter_errors(doc), key=lambda e: e.path):
        loc = "/".join(str(p) for p in err.path) or "(root)"
        errors.append(f"{rp}: schema: {loc}: {err.message}")


def main(argv: list[str]) -> int:
    entry_schema = load_schema("node-index-entry.schema.json")
    package_schema = load_schema("node-index-package.schema.json")
    reserved = load_reserved()

    if argv:
        targets = [pathlib.Path(a) for a in argv]
    else:
        targets = index_files()

    errors: list[str] = []
    warnings: list[str] = []
    checked = 0
    for path in targets:
        if not path.exists():
            # a deleted file shows up in a changed-files list — append-only CI
            # handles deletions; nothing to validate here
            continue
        if path.suffix != ".yml":
            continue
        try:
            path.resolve().relative_to(INDEX_ROOT)
        except ValueError:
            continue
        checked += 1
        validate_file(path, entry_schema, package_schema, reserved, errors, warnings)

    for w in warnings:
        print(f"::warning::{w}" if _in_actions() else f"warning: {w}")
    for e in errors:
        print(f"::error::{e}" if _in_actions() else f"error: {e}")

    if errors:
        print(f"\nvalidate_entries: {len(errors)} error(s) in {checked} file(s)")
        return 1
    print(f"validate_entries: OK ({checked} file(s) checked)")
    return 0


def _in_actions() -> bool:
    import os

    return os.environ.get("GITHUB_ACTIONS") == "true"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
