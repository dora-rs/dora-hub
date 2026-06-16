# pyarrow-sender

Sends a single Apache Arrow array, parsed from the `DATA` env var, then exits.

## Behavior

`pyarrow-sender` connects as a dora node, reads its payload from the `DATA`
environment variable (or the `--data` argument when run directly), and sends it
once on the `data` output before exiting.

The payload string is parsed with `ast.literal_eval`:

- A list/tuple becomes a `pyarrow` array of its elements.
- A scalar (`str`/`int`/`float`) becomes a 1-element `pyarrow` array.
- A string that fails to parse is sent as a single-element string array.

If neither `DATA` nor `--data` is set, the node raises `ValueError` and exits.
It does not read any inputs and does not loop — it emits one event and stops.

## Inputs

None.

## Outputs

- `data`: the Arrow array built from `DATA`. Emitted once at startup.

## Environment variables

- `DATA` (string, required): the payload to send. Evaluated with
  `ast.literal_eval`; a list/tuple becomes an Arrow array of its elements, a
  scalar becomes a 1-element array, and an unparsable value is sent as a string.

## Usage

```yaml
nodes:
  - id: pyarrow-sender
    hub: pyarrow-sender@^0.5
    env:
      DATA: "[1, 2, 3]"
    outputs:
      - data
```

## Build

```bash
pip install .
```
