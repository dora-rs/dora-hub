# node-index bootstrap — maintainer activation checklist

The catalog machinery (Hub spec §7 / roadmap P2.3) is **built and tested** in
this repo:

| Piece | Where |
|-------|-------|
| Catalog layout (`node-index/<ns>/<name>/{package.yml,<ver>.yml}`) | this directory |
| Schema + pin + path validation, append-only, namespace governance | `index-ci` crate (`validate` / `append-only` / `namespace` / `decide`) |
| Path-scoped PR CI | `.github/workflows/node-index-ci.yml` |
| Auto-merge bot | `.github/workflows/index-auto-merge.yml` |
| Weekly reachability + integrity audits (P3.4) | `.github/workflows/node-index-audit.yml` |

What was missing was a **non-empty catalog** and the **repo settings** that let
the bot actually merge. The PR that adds this file also adds the first real
entry (`dora-rs/dora-echo`), which validates, fetches, and audits clean — proof
the pipeline runs end-to-end. The remaining steps are maintainer-only repo
settings.

## 1. Enable the auto-merge bot (repo settings)

The bot (`index-auto-merge.yml`) runs the trusted gate and, on a routine
owner publish, calls `gh pr merge --squash --auto`. That requires:

- [ ] **Allow auto-merge** — Settings → General → Pull Requests →
  check *“Allow auto-merge.”* Without it the bot logs
  *“could not enable auto-merge”* and does nothing.
- [ ] **Branch protection on `main`** — Settings → Branches → add a rule for
  `main` that **requires the `index-checks` status check** (the job in
  `node-index CI`) to pass before merging. This is what makes `--auto` wait
  for green instead of merging blind. Do **not** require the per-node
  `node-hub` matrix or the scheduled audit here — only `index-checks`.
- [ ] **Workflow write permission** — the workflow already declares
  `permissions: { contents: write, pull-requests: write }` and uses the
  built-in `GITHUB_TOKEN` via `pull_request_target`, so **no PAT is needed**.
  Just confirm Settings → Actions → General → Workflow permissions is not set
  to *read-only* (the explicit block overrides it, but confirm org policy
  doesn’t block it).

The bot uses `pull_request_target` and builds the gate from the **base** ref,
checking out the PR head as **data only** (its YAML is parsed, never built or
executed) — so a forked publish PR cannot run code in the privileged context.

## 2. Confirm namespace OWNERS

- [ ] Review `node-index/dora-rs/*/package.yml` — the seed proposes
  `haixuanTao` and `phil-opp` as `dora-rs` owners. Adjust to the real owner
  set. `decide` reads OWNERS from the **base tip**, so only owners listed on
  `main` can have routine publishes auto-merged.
- [ ] `dora-rs` is a **reserved** namespace, so its first claim needs an index
  admin (a human). The seed PR *is* that first claim — review and merge it by
  hand. The bot will `HOLD` it automatically (reserved + new namespace).

## 3. Activation sequence

1. **Merge the seed PR by hand** (the bot holds it — reserved/new namespace).
   → catalog has its first resolvable entry.
2. **Flip the two settings in §1.**
3. **Merge #85** (the remaining migrated entries) — the first real exercise of
   the now-live bot. New namespaces / OWNERS lines in it still route to a human.
4. The weekly `node-index-audit.yml` then re-checks every pinned source.

## How the gate decides (reference)

A `node-index/**` PR auto-merges only when **all three** pass:

- `validate` — schema, full-hash pins, confined `subdir`, well-formed git URL,
  sibling `package.yml` with ≥1 valid owner, no symlinks.
- `append-only` — a published `<ver>.yml` is immutable; the only allowed edits
  are flipping `yanked` (+reason) and *adding* late `source.binary` platforms.
- `decide` — the PR touches only version files in an **existing** namespace,
  every change is an add, no new namespace is claimed, and every entry is
  authored by an **owner** of its namespace.

Anything else (new namespace, OWNERS change, yank by a non-owner, edits to the
index CI/bot itself) is **HOLD** — left for a human. Correctness is still gated
by `node-index CI` regardless.
