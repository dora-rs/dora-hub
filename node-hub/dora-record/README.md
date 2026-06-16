# dora-record

Record dataflow inputs to disk. For every input it receives, `dora-record`
appends a row to a Brotli-compressed Apache Parquet file, capturing the value
together with UHLC and UTC timestamps and the OpenTelemetry trace/span ids.

## Behavior

`dora-record` connects as a dora node and, for each input event, writes to
`out/<dataflow-id>/<input-id>.parquet`. The first time it sees a given input id
it creates the file, derives the schema from that input's Arrow data type, and
spawns a dedicated writer task; subsequent events for the same id stream into
that file. Each row contains:

- `trace_id`, `span_id` — parsed from the input's OpenTelemetry `traceparent`
  (empty when absent).
- `timestamp_uhlc` — the UHLC hybrid-logical-clock time as a `UInt64`.
- `timestamp_utc` — the same instant as a millisecond UTC timestamp.
- `<input-id>` — the input value wrapped in a single-element Arrow list.

On `InputClosed` the corresponding writer is dropped (flushing and closing its
file); remaining writers are closed when the dataflow ends.

## Inputs

- `data`: the stream to record. Any Arrow value — written verbatim to
  `data.parquet` alongside timestamps and trace ids.

The node itself records **any** input id it receives (one `<id>.parquet` file
per id), but a `hub:` contract must declare every wired input (an undeclared
input fails the build), so it declares one generic `data` input. To record
arbitrary or multiple input names, run it directly via `path:` (where the
contract isn't enforced) or use one instance per stream.

## Outputs

None — it is a sink.

## Environment variables

None.

## Usage

Wire a node's output into `dora-record`:

```yaml
nodes:
  - id: dora-record
    hub: dora-record@^0.5
    inputs:
      data: some-node/output
```

Recorded files land under `out/<dataflow-id>/data.parquet`.

## Build

```bash
cargo build --release --target-dir target
```
