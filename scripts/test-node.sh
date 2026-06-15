#!/usr/bin/env bash
# Run locally exactly what CI runs for one node (lint + tests, no publish).
#
#   scripts/test-node.sh <node-name>        e.g. scripts/test-node.sh dora-echo
#
# It reuses the same driver CI uses (.github/workflows/node_hub_test.sh) from the
# node's directory, so "passes locally" means "passes in CI". Python nodes run
# uv + ruff + pytest; Rust nodes run cargo check/clippy/build/test.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
node="${1:-}"

if [[ -z "$node" ]]; then
  echo "usage: $0 <node-name>   (a directory under node-hub/)" >&2
  echo "nodes:" >&2
  ls "$repo_root/node-hub" | sed 's/^/  /' >&2
  exit 2
fi

node_dir="$repo_root/node-hub/$node"
if [[ ! -d "$node_dir" ]]; then
  echo "error: no such node: node-hub/$node" >&2
  exit 1
fi

# The driver keys off the current directory and treats an unset GITHUB_ACTIONS as
# a local run (no disk cleanup). Set GITHUB_EVENT_NAME so its release-publish
# checks evaluate false instead of tripping `set -u` locally — never "release",
# so this can't publish.
export GITHUB_EVENT_NAME="local"

# A locally-active conda/virtualenv hijacks `uv`'s target — it would install the
# node into the active env instead of the node's own `.venv`, then `uv run` can't
# find it. Neutralize it so `uv venv` + `uv pip install .` are authoritative.
unset VIRTUAL_ENV CONDA_PREFIX CONDA_DEFAULT_ENV

echo ">> testing node-hub/$node (local)"
cd "$node_dir"
exec bash "$repo_root/.github/workflows/node_hub_test.sh"
