# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`dora-rs/dora-hub` is the **collection of reusable [dora](https://github.com/dora-rs/dora) nodes** — ~60 ready-to-use sensors, models, transforms, and sinks you can drop into a dataflow. Nodes are independently packaged and published (Python to PyPI, Rust to crates.io); a dataflow pulls one in via `pip install dora-<name>` / a `git:` source / or a `hub:` reference.

This repo also hosts the **Dora Hub catalog** (`node-index/`) — the git-backed index that the `dora hub` CLI resolves `hub:` references against (spec: [`docs/plan-node-hub.md`](https://github.com/dora-rs/dora/blob/main/docs/plan-node-hub.md) in `dora-rs/dora`).

> **Note:** the `node-index/` catalog, its `schemas/`, `scripts/index-ci/`, and the `node-index CI` workflow land with the catalog bootstrap (PR #66). Until that merges, the index sections below don't apply and `make index-ci` is a no-op.

Two languages, one repo:
- **Python nodes** (~55) — `pyproject.toml` + a package dir, built/tested with **uv**, linted with **ruff**, tested with **pytest**.
- **Rust nodes** (~14) — members of the root **Cargo workspace**; some are Rust+Python (maturin) hybrids.

## Repository Layout

| Path | What |
|------|------|
| `node-hub/<name>/` | One node per directory — the active, maintained collection |
| `node-archive/` | Deprecated nodes, not built or tested in CI |
| `node-index/` | The Dora Hub catalog (`<ns>/<name>/<version>.yml`); see `node-index/README.md` |
| `examples/` | End-to-end dataflows exercised by CI (`examples/*/dataflow.yml`) |
| `tests/` | Workspace-level Rust integration tests |
| `benches/` | Benchmarks |
| `schemas/` | JSON schemas for the node-index entry / package format |
| `scripts/` | Repo tooling (`index-ci/` validators, `test-node.sh`) |
| `.github/workflows/` | CI (see below) |

### Anatomy of a node

**Python node** (`node-hub/dora-echo/`):
```
dora-echo/
  pyproject.toml          # name = "dora-echo"; [project.scripts] dora-echo = "dora_echo.main:main"
  dora_echo/__init__.py   # package dir (underscores)
  dora_echo/main.py       # entrypoint
  tests/test_dora_echo.py # pytest
  README.md
```
**Packaging convention:** the PyPI dist name, the `[project.scripts]` console-script, and the value a dataflow uses as `path:` are **all the same string** (`dora-echo`). New nodes must keep this 1:1:1 mapping. Lint config lives under `[tool.ruff.lint]` — see the shared baseline in the root `ruff.toml`.

**Rust node** (`node-hub/dora-rerun/`): a `Cargo.toml` workspace member (add it to the root `Cargo.toml` `members`). Rust+Python hybrids also carry a `pyproject.toml` and build a wheel with maturin.

## Build & Test Commands

### Rust workspace
```bash
# A few nodes need system libs / are heavy — CI excludes them:
cargo check  --all --exclude dora-dav1d --exclude dora-rav1e --exclude dora-qwen-omni
cargo build  --all --exclude dora-dav1d --exclude dora-rav1e --exclude dora-qwen-omni
cargo test   --all --exclude dora-dav1d --exclude dora-rav1e --exclude dora-qwen-omni
cargo clippy --all --exclude dora-dav1d --exclude dora-rav1e --exclude dora-qwen-omni
cargo fmt --all -- --check
```

### A single node (the inner loop)
Run exactly what CI runs for one node, locally:
```bash
make test-node NODE=dora-echo        # or: scripts/test-node.sh dora-echo
```
For a Python node this does `uv venv` + `uv pip install .` + `uv run ruff check .` + `uv run pytest`; for a Rust node, `cargo check/clippy/build/test`. A Rust+Python (maturin) hybrid additionally runs a `maturin build --release` (no publish). It reuses the same driver CI uses (`.github/workflows/node_hub_test.sh`), so local == CI.

### The node-index catalog
```bash
make index-ci                         # schema + append-only validators + unit tests
# or individually:
python3 scripts/index-ci/tests/test_index_ci.py
python3 scripts/index-ci/validate_entries.py
```

## Pre-commit Quality Gates (MANDATORY)

**Run these before pushing.** Remote CI is a large multi-platform, ~60-node matrix — catching failures locally saves a long round-trip.

1. **For each node you touched:** `make test-node NODE=<name>` (ruff + pytest, or cargo for Rust nodes).
2. **If you touched Rust workspace code:** `cargo fmt --all -- --check`, then `cargo clippy --all <excludes>`, then `cargo test --all <excludes>`.
3. **If you touched `node-index/`, `schemas/`, or `scripts/index-ci/`:** `make index-ci`.
4. **Always:** `typos` (config: `_typos.toml`).

`make qa` runs the repo-wide static gates (ruff on repo Python + rustfmt check + clippy + index-ci + typos) in one go. Use `make test-node NODE=<n>` for the node you changed.

## Remote CI

| Workflow | Scope |
|----------|-------|
| `ci.yml` | Rust workspace Check/Build/Test (Linux/macOS/Windows), Clippy, rustfmt, Typos, and example dataflows end-to-end |
| `node-hub-ci-cd.yml` | Per-node matrix (`node_hub_test.sh` for each `node-hub/<folder>`). PRs run the lint+test step on Linux only; macOS runs on release/dispatch. Publishes on release |
| `node-index-ci.yml` | Catalog CI (schema + append-only), **path-scoped to `node-index/**`** — never gates `node-hub/` source |
| `claude-code.yml` | `@claude` GitHub automation |

## Conventions

- **Adding a node:** see [`CONTRIBUTING.md`](CONTRIBUTING.md) — keep the name 1:1:1 (dist == console-script == `path:`), add Rust nodes to the workspace `members`, include a `tests/` directory.
- **Rust:** format with `rustfmt` defaults; fix clippy before pushing.
- **Python:** ruff for lint (baseline in root `ruff.toml`), pytest for tests, **uv** for envs (not bare `pip`).
- **Publishing to the Hub:** use `dora hub publish` (don't hand-edit `node-index/` entries); published version files are immutable (append-only, enforced by `node-index CI`).
- Discuss non-trivial changes in a GitHub issue or Discord first; don't fix unrelated warnings in a PR.
