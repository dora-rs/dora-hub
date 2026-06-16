# dora-parquet-recorder

A generic, zero-copy data recorder for dora. It records received inputs in
batches and writes one Apache Parquet file per input id.

## Behavior

`dora-parquet-recorder` connects as a dora node and records **every** input
event it receives. For each input id it appends a row containing the receive
`timestamp`, the input `data` (raw Arrow buffer bytes when available, otherwise
a Python-level fallback), and the input `metadata` (JSON). Rows are buffered per
input id and flushed in batches of `BATCH_SIZE` to `LOG_DIR/<input-id>.parquet`
using a background writer thread. On `STOP` it flushes any remaining buffered
rows and closes all open files.

The node itself records **any** input id it receives, but a `hub:` contract must
declare every wired input (an undeclared input fails the build), so it declares
one generic `data` input. To record arbitrary or multiple input names, run it
directly via `path:` (where the contract isn't enforced) or use one instance per
stream.

## Inputs

- `data`: the stream to record. Any Arrow value. Each input id is written to its
  own `<id>.parquet` file with `timestamp`, `data`, and `metadata` columns.

## Outputs

- `status`: a single `"READY"` value emitted on startup as a handshake signal.

## Environment variables

- `BATCH_SIZE` (default `30`): number of tables buffered per input id before a
  batch is flushed to disk.
- `LOG_DIR` (default `data_logs`): directory where per-input Parquet files are
  written. Created if it does not exist.

## Usage

Wire a node's output into `dora-parquet-recorder`:

```yaml
nodes:
  - id: dora-parquet-recorder
    hub: dora-parquet-recorder@^0.5
    inputs:
      data: some-node/output
    env:
      BATCH_SIZE: "30"
      LOG_DIR: "data_logs"
```

## Build

```bash
pip install .
```
