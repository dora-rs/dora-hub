# dora-echo

Echoes every input it receives straight back out under the same id.

## Behavior

`dora-echo` connects as a dora node (default node name `echo`, overridable with
`--name` for use as a dynamic node) and loops over its incoming events. For each
event of type `INPUT`, it re-sends the data unchanged with `send_output`, reusing
the input's own id, value, and metadata. The effect is a passthrough/loopback:
whatever you wire in comes back out under the same name. It is useful for testing
wiring, latency, and dynamic-node connections.

## Inputs

- `data`: the value to echo. Re-sent unchanged on the `data` output.

The node re-sends **any** input id it receives back under that same id, but a
`hub:` contract must declare every wired input/output (undeclared ones fail the
build), so it declares one generic `data` in/out. Wire to `data` for Hub usage,
or run it directly via `path:` to echo arbitrary input names.

## Outputs

- `data`: the `data` input, re-sent unchanged (same value and metadata).

## Environment variables

None. (The code references the `CI` variable to set a module-level test guard,
but it does not affect node behavior.)

## Usage

```yaml
nodes:
  - id: echo
    hub: dora-echo@^0.5
    inputs:
      data: some-node/output
    outputs:
      - data
```

The echoed value is available as `echo/data`.

## Build

```bash
pip install .
```
