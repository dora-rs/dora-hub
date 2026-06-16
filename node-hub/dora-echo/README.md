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

Accepts **any** input, under **any** name — it echoes whatever you wire to it.
There is no fixed or typed input contract (so `dora-node.yml` declares none).

## Outputs

For each input received, emits an output **with the same id** as that input,
carrying the unchanged value and metadata. The set of output ids is therefore
determined entirely by the inputs you wire in (so `dora-node.yml` declares none).

## Environment variables

None. (The code references the `CI` variable to set a module-level test guard,
but it does not affect node behavior.)

## Usage

```yaml
nodes:
  - id: echo
    hub: dora-echo@^0.5
    inputs:
      my-input: some-node/output
```

The output of the `echo` node is available as `echo/my-input`.

## Build

```bash
pip install .
```
