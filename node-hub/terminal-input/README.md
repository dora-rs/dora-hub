# terminal-input

Reads data from the terminal (or the `DATA` env var) and emits it as an Apache Arrow output.

## Behavior

`terminal-input` connects as a dora node and sends a `data` output. If the node
isn't connected yet it waits and retries, so wiring order doesn't matter.

- **Interactive mode** (no `DATA` env var and no `DORA_NODE_CONFIG`): it prompts
  `Provide the data you want to send:` on stdin in a loop. Each line is parsed
  with `ast.literal_eval` (falling back to a plain string on `ValueError` /
  `SyntaxError`), wrapped in a `pyarrow` array, and sent as the `data` output.
  After each send it drains any pending input events and prints them as
  `Received: <value>`.
- **Single-shot mode** (`DATA` env var set, or `DORA_NODE_CONFIG` present): it
  parses the value once, wraps it in a `pyarrow` array, and sends it as `data`.

## Inputs

None declared. In interactive mode it does read back any input events that are
wired to it and prints them, but there is no fixed or typed input contract (so
`dora-node.yml` declares none).

## Outputs

- `data` — the value typed at the prompt (or taken from the `DATA` env var),
  parsed via `ast.literal_eval` and wrapped in an Arrow array.

## Environment variables

- `DATA` — if set, sends this value once instead of prompting interactively on
  stdin.

## Usage

```yaml
nodes:
  - id: terminal-input
    hub: terminal-input@^0.5
    outputs:
      - data
```

## Build

```bash
pip install .
```
