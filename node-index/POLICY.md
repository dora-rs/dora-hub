# Namespace policy

How namespaces in the Dora Hub catalog (`node-index/`) are claimed, governed,
transferred, and disputed. Written before the first dispute, deliberately short.
Implements [`docs/plan-node-hub.md` §7.4](https://github.com/dora-rs/dora/blob/main/docs/plan-node-hub.md)
(`dora-rs/dora`).

## Claiming a namespace

A namespace is the top directory under `node-index/<namespace>/`. You claim one
by opening a PR that adds `node-index/<namespace>/<name>/package.yml` with an
`owners` list. Index CI screens every **new** namespace structurally:

- **Identity** — the namespace must equal the claiming PR author's GitHub login,
  or an org they publicly belong to (enforced by the auto-merge bot, which has
  the trusted PR actor; routine publishing *within* a namespace you own is not
  re-checked).
- **Reserved** — names belonging to the project or type system
  ([`index-ci/reserved_namespaces.txt`](../index-ci/reserved_namespaces.txt):
  `dora`, `dora-rs`, `dora-hub`, `std`, `official`, `hub`, …) need an index
  admin, not auto-merge.
- **Confusable** — a new namespace within edit-distance 1 of, or a homoglyph
  skeleton of, an existing or reserved namespace (`d0ra-rs`, `acrne`) is flagged
  for **mandatory human review**. Lookalikes are the dependency-confusion
  vector, so this is a hard gate, not a warning.

New-namespace PRs always get a human reviewer. Once the namespace exists,
publishing new versions into it auto-merges (see §7.5).

## Ownership

`package.yml` `owners` lists the GitHub accounts (or orgs) that may publish into
the namespace. Adding or removing an owner is an `owners`-list change and always
requires human review — it is never auto-merged.

## Transfer

Ownership transfers by a PR editing the `owners` list, approved by:

- a current owner, **or**
- an index admin, when an owner is **unresponsive for 6 months** (no reply on a
  transfer-request issue that @-mentions every current owner). The 6-month clock
  starts when that issue is opened.

The catalog is append-only: a transfer never deletes published versions, and
existing lockfile pins keep resolving regardless of who owns the namespace.

## Disputes

- **Trademark / name conflict.** A rights-holder may request a namespace by
  opening an issue with evidence. Index admins mediate; the default remedy is
  transfer (above), not deletion — downstream pins must keep working.
- **Squatting / abuse.** A namespace claimed but unused, or used to typosquat an
  existing one (the confusable gate exists to catch this at PR time), may be
  reclaimed by index admins after the 6-month unresponsive window.
- **Yank over an unresponsive owner.** Index admins may yank a broken or
  malicious version even without the owner, by an admin-merged yank PR (a yank
  sets `yanked: true` + a reason; it never deletes bytes).

## Index admins

Index admins are the `dora-rs` maintainers. Admin actions (reserved-namespace
grants, transfers over unresponsive owners, disputed reclaims, yank-by-non-owner)
are performed by an admin-merged PR so the decision stays in the git history.
