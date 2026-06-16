# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`dora-rs/dora-hub` is the **collection of reusable [dora](https://github.com/dora-rs/dora) nodes** — ~60 ready-to-use sensors, models, transforms, and sinks you can drop into a dataflow. Nodes are independently packaged and published (Python to PyPI, Rust to crates.io); a dataflow pulls one in via `pip install dora-<name>` / a `git:` source / or a `hub:` reference.

This repo also hosts the **Dora Hub catalog** (`node-index/`) — the git-backed index that the `dora hub` CLI resolves `hub:` references against (spec: [`docs/plan-node-hub.md`](https://github.com/dora-rs/dora/blob/main/docs/plan-node-hub.md) in `dora-rs/dora`).

> **Note:** the `node-index/` catalog, the `index-ci/` Rust gate, and the `node-index CI` workflow are the Dora Hub catalog enforcement. `make index-ci` is a no-op in a tree without them.

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
| `index-ci/` | Rust crate (`dora-index-ci`) — the node-index validate / append-only / namespace / auto-merge gate |
| `scripts/` | Repo tooling (`test-node.sh`) |
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

**Rust node** (`node-hub/dora-rerun/`): a `Cargo.toml` workspace member (add it to the root `Cargo.toml` `members`). Rust+Python hybrids also carry a `pyproject.toml` and build a wheel with maturin. **Hub-spawnability:** a Rust node must take its node id from the daemon — use `DoraNode::init_flexible(NodeId::from(...))` (or `init_from_env()`), never a hard-coded `init_from_node_id(...)`, or `hub:` only works when the dataflow names the node exactly that id. (Python `Node()` reads `DORA_NODE_CONFIG` already, so it's fine.)

### Node README checklist

Every node must ship a `README.md`. `dora-node.yml` is the machine contract; the README is the human one — keep them consistent (same description, same example). Use this structure (omit a section only if truly N/A; see `node-hub/terminal-print/README.md` for the reference):

- [ ] **Title + one-line description** — matches `dora-node.yml`'s `description`.
- [ ] **Behavior** — what the node actually does (the logic), not just what it is.
- [ ] **Inputs** — each input id + type. **Declare every input a dataflow will wire**: a `hub:` build fails on any wired input not in the manifest (an empty map is *not* a wildcard). A generic sink that prints "any input" must still declare a concrete input (e.g. `data`) — don't leave `inputs` empty.
- [ ] **Outputs** — each output id + type, or "None" for sinks.
- [ ] **Environment variables** — name, type, default, meaning (mirror `dora-node.yml`'s `env`), or "None". Document only vars that **actually affect behavior**: a var the code reads with `os.getenv` but never uses is *not* part of the contract — leave it out of the manifest/README (don't imply it does something).
- [ ] **Usage** — a copy-pasteable dataflow YAML snippet wiring the node. Prefer the `hub:` form. If you show a from-source `path:`, it is the built **executable** (the manifest `entrypoint`, e.g. `target/release/<bin>`), paired with `build:` — never the package directory.
- [ ] **Build** — for workspace-member Rust nodes, note `cargo build --release --target-dir target` (package-local binary, matches `entrypoint`).

No broken relative links to examples in the upstream `dora-rs/dora` repo — inline the snippet instead.

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
make index-ci                         # cargo test + validate + append-only + namespace
# or individually:
cargo test -p dora-index-ci
cargo run -p dora-index-ci -- validate
```

## Pre-commit Quality Gates (MANDATORY)

**Run these before pushing.** Remote CI is a large multi-platform, ~60-node matrix — catching failures locally saves a long round-trip.

1. **For each node you touched:** `make test-node NODE=<name>` (ruff + pytest, or cargo for Rust nodes).
2. **If you touched Rust workspace code:** `cargo fmt --all -- --check`, then `cargo clippy --all <excludes>`, then `cargo test --all <excludes>`.
3. **If you touched `node-index/` or `index-ci/`:** `make index-ci`.
4. **Always:** `typos` (config: `_typos.toml`).

`make qa` runs the repo-wide static gates (ruff on repo Python + rustfmt check + clippy + index-ci + typos) in one go. Use `make test-node NODE=<n>` for the node you changed.

## Remote CI

| Workflow | Scope |
|----------|-------|
| `ci.yml` | Rust workspace Check/Build/Test (Linux/macOS/Windows), Clippy, rustfmt, Typos, and example dataflows end-to-end |
| `node-hub-ci-cd.yml` | Per-node matrix (`node_hub_test.sh` for each `node-hub/<folder>`). PRs run the lint+test step on Linux only; macOS runs on release/dispatch. Publishes on release |
| `node-index-ci.yml` | Catalog CI — the `dora-index-ci` Rust gate (validate + append-only + namespace), **path-scoped to `node-index/**`** — never gates `node-hub/` source |
| `index-auto-merge.yml` | Trusted-context bot: builds `dora-index-ci` from the base ref and auto-merges a routine, owner-authored publish (`pull_request_target`) |
| `claude-code.yml` | `@claude` GitHub automation |

## Conventions

- **Adding a node:** see [`CONTRIBUTING.md`](CONTRIBUTING.md) — keep the name 1:1:1 (dist == console-script == `path:`), add Rust nodes to the workspace `members`, include a `tests/` directory.
- **Rust:** format with `rustfmt` defaults; fix clippy before pushing.
- **Python:** ruff for lint (baseline in root `ruff.toml`), pytest for tests, **uv** for envs (not bare `pip`).
- **Publishing to the Hub:** use `dora hub publish` (don't hand-edit `node-index/` entries); published version files are immutable (append-only, enforced by `node-index CI`).
- Discuss non-trivial changes in a GitHub issue or Discord first; don't fix unrelated warnings in a PR.
