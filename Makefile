# dora-hub developer convenience targets. See CLAUDE.md for the full workflow.
# Rust nodes that need system libs / are heavy are excluded, matching ci.yml.
RUST_EXCLUDES := --exclude dora-dav1d --exclude dora-rav1e --exclude dora-qwen-omni

# GitHub login for the §7.4 identity gate in `index-ci`. Only consulted when the
# tree adds a NEW namespace (otherwise the gate is a no-op), so the default is
# best-effort: your gh-authenticated login. Override for a real new-namespace
# claim: `make index-ci AUTHOR=<your-login>`.
AUTHOR ?= $(shell gh api user --jq .login 2>/dev/null || echo unknown)

.PHONY: help
help:
	@echo "dora-hub dev targets:"
	@echo "  make test-node NODE=<name>   test + lint one node like CI (uv/ruff/pytest, or cargo)"
	@echo "  make lint                    ruff (repo Python) + rustfmt --check + clippy"
	@echo "  make fmt                     cargo fmt + ruff format (repo Python)"
	@echo "  make test-rust               cargo test for the Rust workspace"
	@echo "  make index-ci                node-index gates: schema + append-only + namespace + identity"
	@echo "  make typos                   spell-check (crate-ci/typos)"
	@echo "  make qa                      pre-commit static gates: lint + index-ci + typos"

.PHONY: test-node
test-node:
	@test -n "$(NODE)" || { echo "usage: make test-node NODE=<name>" >&2; exit 2; }
	scripts/test-node.sh $(NODE)

.PHONY: lint
lint:
	ruff check .
	cargo fmt --all -- --check
	cargo clippy --all $(RUST_EXCLUDES)

.PHONY: fmt
fmt:
	cargo fmt --all
	ruff format .

.PHONY: test-rust
test-rust:
	cargo test --all $(RUST_EXCLUDES)

.PHONY: index-ci
index-ci:
	@if [ -d index-ci ]; then \
	  cargo test -p dora-index-ci && \
	  cargo run -q -p dora-index-ci -- validate && \
	  cargo run -q -p dora-index-ci -- append-only --base origin/main && \
	  cargo run -q -p dora-index-ci -- namespace --base origin/main && \
	  cargo run -q -p dora-index-ci -- identity --author "$(AUTHOR)" --base origin/main; \
	else echo "index-ci: no node-index/ catalog in this tree — skipping"; fi

.PHONY: typos
typos:
	@if command -v typos >/dev/null; then typos; \
	else echo "typos: not installed (\`cargo install typos-cli\`) — skipping"; fi

.PHONY: qa
qa: lint index-ci typos
	@echo "qa: static gates passed"
