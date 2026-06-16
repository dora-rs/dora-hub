# pyarrow-assert

Test sink that asserts every received input equals a fixed expected pyarrow value.

## Behavior

`pyarrow-assert` connects as a dora node and, for each `INPUT` event it
receives, asserts that the event's value equals an expected pyarrow array,
raising `AssertionError` (`Expected {data}, got {value}`) on mismatch.

The expected value comes from the `DATA` environment variable (falling back to
the `--data` argument, default empty string). It is parsed with
`ast.literal_eval`; if parsing fails the raw string is used. The result is then
wrapped into a pyarrow array:

- A `str`, `int`, or `float` becomes a 1-element array (`pa.array([data])`).
- Anything else (e.g. a list) is passed directly to `pa.array(data)`.

The node produces no outputs. It is intended for use in tests.

## Inputs

- `data`: the stream to assert against the expected value.

The node asserts on **any** input id it receives, but a `hub:` contract must
declare every wired input, so it declares one generic `data` input. To assert
arbitrary or multiple input names, run it directly via `path:` (where the
contract isn't enforced).

## Outputs

None — it is a sink.

## Environment variables

- `DATA` (default empty string): the expected value to assert against. Parsed
  with `ast.literal_eval` and wrapped in a pyarrow array (see Behavior).

## Usage

Wire a node's output into `pyarrow-assert` and set the expected `DATA`:

```yaml
nodes:
  - id: source
    path: dynamic
    outputs:
      - value
  - id: pyarrow-assert
    hub: pyarrow-assert@^0.5
    inputs:
      data: source/value
    env:
      DATA: "[1, 2, 3]"
```

## Build

```bash
pip install .
```
