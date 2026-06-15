# AGENTS.md

This file provides guidance to Codex and other agentic coding tools working in
this repository. It complements, but does not replace, [`CLAUDE.md`](CLAUDE.md).
If the two documents overlap, follow the stricter rule.

## Start here

`dora-hub` is the collection of reusable [dora](https://github.com/dora-rs/dora)
nodes (~60), and it also hosts the Dora Hub catalog (`node-index/`). Most nodes
are Python (built/tested with **uv**, linted with **ruff**, tested with
**pytest**); some are Rust members of the root Cargo workspace, and a few are
Rust+Python (maturin) hybrids.

> **Note:** the `node-index/` catalog and its tooling land with the catalog
> bootstrap (PR #66). Until that merges, the catalog instructions below
> (`make index-ci`, the "don't hand-edit the catalog" rule) don't yet apply.

**Read [`CLAUDE.md`](CLAUDE.md) first** — it is the canonical reference for the
repo map, build/test commands, the CI structure, and the pre-commit gates.
[`CONTRIBUTING.md`](CONTRIBUTING.md) covers how to add a node.

## Working rules

- **Surgical changes.** Touch only what the task needs and match the surrounding
  style. Don't fix unrelated warnings or reformat code you aren't changing.
- **One node at a time.** A node lives in `node-hub/<name>/`. Preserve the 1:1:1
  packaging convention: the PyPI dist name, the `[project.scripts]`
  console-script, and the `path:` a dataflow uses are all the same string.
- **Verify what you change.** `make test-node NODE=<name>` runs exactly what CI
  runs for one node (uv + ruff + pytest, or cargo). For Rust workspace changes,
  `make test-rust` and `make lint`. For `node-index/` changes, `make index-ci`.
- **Don't hand-edit the catalog** (once it exists — see the note above).
  `node-index/` entries come from `dora hub publish`; published version files
  are immutable (append-only).

## Before you finish

Run the pre-commit gates for what you touched (see CLAUDE.md → "Pre-commit
Quality Gates"): the changed node's `make test-node`, plus `cargo
fmt`/`clippy`/`test` for Rust, `make index-ci` for the catalog, and `typos`.
Don't push red.

## Don'ts

- Don't commit secrets, a node's `.venv/`, or build artifacts.
- Don't switch git branches or run destructive git commands in a shared working
  tree without saying so first.
- Don't add a `.ci-skip` / `.skip-test` marker to paper over a real failure —
  fix the node, or flag the breakage.
