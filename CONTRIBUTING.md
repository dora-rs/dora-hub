# Contributing to dora-hub

`dora-hub` is the collection of reusable [dora](https://github.com/dora-rs/dora)
nodes. This guide covers adding/maintaining a node and the checks to run before
you push. See [`CLAUDE.md`](CLAUDE.md) for the repo map and command reference.

## Adding a node

A node lives in its own directory under `node-hub/<name>/`.

### Python node

```
node-hub/my-node/
  pyproject.toml          # see below
  my_node/__init__.py     # package dir — underscores
  my_node/main.py         # entrypoint with a main()
  tests/test_my_node.py   # at least an import/smoke test
  README.md
```

`pyproject.toml` must keep the **1:1:1 naming convention** — the PyPI
distribution name, the `[project.scripts]` console-script, and the string a
dataflow uses as `path:` are all the same:

```toml
[project]
name = "my-node"                       # dist name == path: in a dataflow

[project.scripts]
my-node = "my_node.main:main"          # console-script == dist name

[tool.ruff.lint]
extend-select = ["D", "UP", "PERF", "RET", "RSE", "NPY", "N", "I"]  # align with /ruff.toml
```

- Use **uv** for environments and **pytest** for tests.
- Keep your node's `[tool.ruff.lint]` aligned with the repo baseline in
  [`ruff.toml`](ruff.toml).
- A test that imports `main` and asserts it raises outside a dataflow is the
  minimum (see `node-hub/dora-echo/tests/`).

### Rust node

Add the crate to the root `Cargo.toml` `members`. Rust+Python hybrids also carry
a `pyproject.toml` and build a wheel with maturin (the CI driver handles this).

## Testing locally

Run exactly what CI runs for your node:

```bash
make test-node NODE=my-node        # uv + ruff + pytest, or cargo for a Rust node
```

The helper neutralizes any active conda/virtualenv so `uv` uses the node's own
`.venv`. For workspace-wide Rust changes:

```bash
make test-rust                     # cargo test for the workspace
make lint                          # ruff (repo tooling) + rustfmt --check + clippy
```

## Before you push

The remote matrix is large (≈60 nodes × platforms), so check locally first:

1. `make test-node NODE=<name>` for **each node you touched**.
2. If you changed **Rust workspace** code: `cargo fmt --all -- --check`, `cargo clippy`, `cargo test`.
3. If you changed **`node-index/`**: `make index-ci`.
4. `make qa` bundles the static gates (lint + index-ci + typos).

Don't fix unrelated warnings in a PR; discuss non-trivial changes in an issue or
on Discord first.

## Publishing to the Dora Hub

> The `node-index/` catalog and its CI land with the catalog bootstrap (PR #66);
> this section applies once that merges.

The `node-index/` catalog is produced by the `dora hub publish` CLI — **don't
hand-edit version entries**. Published version files are immutable (append-only,
enforced by `node-index CI`); a yank is the one allowed mutation. See
`node-index/README.md` for the format.
