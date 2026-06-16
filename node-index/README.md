# `node-index/` — the Dora Hub catalog

This directory is the **Dora Hub index**: a git-backed catalog mapping
`[<namespace>/]<name>@<version>` to a node's manifest snapshot and a pointer to
where its source lives. The `dora` CLI resolves `hub:` references against it.

Full design: [`docs/plan-node-hub.md` §7](https://github.com/dora-rs/dora/blob/main/docs/plan-node-hub.md)
in `dora-rs/dora`.

## Layout

```
node-index/
  <namespace>/
    <name>/
      package.yml        # description, repo, and the OWNERS list (who may publish)
      <version>.yml       # one file per published version: manifest + source pointer
```

A version file is the node's `dora-node.yml` (verbatim) plus a `source` stanza:

```yaml
# node-index/dora-rs/dora-yolo/0.5.2.yml
manifest: { ...dora-node.yml contents... }
source:
  git: https://github.com/dora-rs/dora-hub   # or the author's own repo
  rev: 9f4c1ae...                            # a full commit hash (never a branch/tag)
  subdir: node-hub/dora-yolo                 # where the node lives in the repo
published: "2026-07-01T12:00:00Z"
yanked: false
```

The index **never hosts the bytes** — it points at them. The source repo need
not be `dora-hub`; third-party namespaces point at their own repos.

## Publishing

Use the CLI rather than hand-editing:

```bash
dora hub publish            # validates the manifest, resolves the commit pin,
                            # and prints/opens the index entry to add
```

Then open a PR adding the `node-index/<ns>/<name>/<version>.yml` file. CI
validates it and the auto-merge bot merges a routine publish without a review
queue — publishing latency is CI latency.

## Rules (machine-enforced by the `dora-index-ci` Rust gate)

All enforcement lives in the [`index-ci/`](../index-ci) crate, which reuses the
same envelope types as the `dora hub` CLI (`dora-hub-client`) so the validator
and the publisher can't drift.

- **Append-only.** A PR may *add* version files. The only permitted edit to an
  existing version file is flipping `yanked` (with a non-empty `yank_reason`) or
  *appending* `source.binary` entries for platforms not already pinned. Any
  other change to a published version fails CI — published versions are immutable.
- **Schema + pins.** Entries deserialize into the typed envelope (a stray key
  fails); `source.rev` must be a full 40-/64-hex commit, `source.git` a
  scheme-checked remote, `subdir` relative with no `..`, and the directory must
  match `manifest.namespace` / `manifest.name`.
- **Namespaces.** A new namespace claim is screened against the reserved list
  ([`index-ci/reserved_namespaces.txt`](../index-ci/reserved_namespaces.txt) →
  needs an index admin) and a confusable/edit-distance check (→ human review),
  and always gets a human reviewer — see [`POLICY.md`](POLICY.md).
- **Owner identity.** The auto-merge bot merges a version only when it is
  published by an **owner** of its namespace (the `package.yml` OWNERS list,
  read from the base branch — a PR can't add itself as an owner and self-approve).

These checks run **only** on `node-index/**` — they never gate the
human-reviewed `node-hub/` source path.

## Trust boundary

The auto-merge bot (`.github/workflows/index-auto-merge.yml`) builds the
`dora-index-ci` binary from the **base branch** and runs it against a PR's
catalog *data* (never the PR's own sources), so a PR can't neuter its own gate.
Combined with the owner-identity check, a merged `node-index/` entry has
verified provenance: published by a namespace owner, schema-valid, pinning an
immutable commit, never overwriting an existing version.

Two repo settings make this airtight and should be enabled by an admin:
**branch protection** requiring the `node-index CI` check + code-owner review on
`index-ci/**` and `.github/workflows/**`; and **"Allow auto-merge"** so the bot
can queue a merge behind those required checks.
