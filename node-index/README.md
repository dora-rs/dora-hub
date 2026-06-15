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
validates it and (once the auto-merge bot lands) merges it without a review
queue — publishing latency is CI latency.

## Rules (machine-enforced by `node-index CI`)

- **Append-only.** A PR may *add* version files. The only permitted edit to an
  existing version file is flipping `yanked` (with a `yank_reason`) or
  *appending* late `source.binary` platform entries. Any other change to a
  published version fails CI — published versions are immutable.
- **Schema + pins.** Entries validate against
  [`schemas/node-index-entry.schema.json`](../schemas/node-index-entry.schema.json);
  `source.rev` must be a full 40-/64-hex commit, and the directory must match
  `manifest.namespace` / `manifest.name`.
- **Namespaces.** A new namespace claim (`package.yml`) for a
  [reserved name](../scripts/index-ci/reserved_namespaces.txt) needs an index
  admin. Owner-identity and confusable-name checks land with the governance
  workstream (P2.3 PR 2).

These checks run **only** on `node-index/**` — they never gate the
human-reviewed `node-hub/` source path.
