<!--
Publishing a node to the Dora Hub index. Open this with:
  ?template=publish-node.md
or just edit the checklist below. CI validates everything automatically.
-->

## Publish: `<namespace>/<name>@<version>`

Generated with `dora hub publish` (please don't hand-edit version entries).

### Checklist
- [ ] Adds **only new** `node-index/<ns>/<name>/<version>.yml` file(s) (published versions are immutable — append-only).
- [ ] `source.rev` is a full commit hash the source repo actually contains.
- [ ] `manifest.namespace` / `manifest.name` match the directory path.
- [ ] I am listed in `node-index/<ns>/<name>/package.yml` `owners` (or this PR also adds the package + my namespace claim).

### Notes
<!-- Anything reviewers/automation should know. New namespaces and OWNERS changes get a human review. -->
