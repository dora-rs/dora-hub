# terminal-print

Print received inputs to the terminal — a debug sink that pretty-prints UTF-8
strings and debug-dumps any other Apache Arrow type. Handy for inspecting what a
node emits while building or debugging a dataflow.

## Behavior

`terminal-print` connects as a dora node and prints every input event it
receives as `Received id: <input-id>, data: <value>`:

- **UTF-8 string** inputs are printed as plain text.
- **Any other Arrow type** is printed with a `{:#?}` debug dump of the array.

If the node isn't connected yet it waits and retries, so wiring order doesn't
matter. It produces no outputs.

## Inputs

Accepts **any** input, under **any** name — it prints whatever you wire to it.
There is no fixed or typed input contract (so `dora-node.yml` declares none).

## Outputs

None — it is a sink.

## Environment variables

None.

## Usage

Wire any node's output into `terminal-print`:

```yaml
nodes:
  - id: terminal-print
    hub: terminal-print@^0.5
    inputs:
      text: some-node/output
```

## Build

```bash
cargo build --release --target-dir target
```

`--target-dir target` keeps the binary package-local at
`target/release/terminal-print`, which is the `entrypoint` Hub spawns (see
[`dora-node.yml`](dora-node.yml)).
