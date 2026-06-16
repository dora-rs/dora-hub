# dora-dav1d

AV1 video decoder using the [dav1d](https://crates.io/crates/dav1d) decoder.
It takes encoded AV1 byte streams in and emits raw decoded image frames out.

## Behavior

`dora-dav1d` connects as a dora node and feeds every input event into a dav1d
decoder. For each input it:

- Accepts only UInt8 Arrow arrays; other data types are ignored with a warning.
- Requires the input's `encoding` parameter to be `av1` (defaults to `av1` when
  absent); any other encoding is skipped with a warning.
- Sends the bytes to the decoder and, when a picture is available, converts it
  based on its pixel layout:
  - **I420 (color)** — controlled by the `ENCODING` environment variable:
    `bgr8` produces interleaved BGR; `yuv420` produces planar Y/U/V concatenated
    plus `width` and `height` parameters.
  - **I400 (mono)** — emits `mono8` for 8-bit pictures or `mono16` for 10-/12-bit
    pictures.
- Unsupported pixel layouts, bit depths, or output encodings are skipped with a
  warning.

The decoded frame is emitted under the **same id** as the received input.

## Inputs

- `tick`: encoded AV1 data as a UInt8 Arrow array, with an `encoding=av1`
  parameter. The node echoes the output under this same id, so the input id
  determines the output id.

## Outputs

- `tick`: the decoded image frame as an Arrow array, emitted under the same id
  as the input. Metadata carries `encoding` (`bgr8`, `yuv420`, `mono8`, or
  `mono16`), `primitive: image`, and `width`/`height` for `yuv420`.

## Environment variables

- `ENCODING` (default `bgr8`): output pixel encoding for I420 color pictures.
  `bgr8` converts to interleaved BGR; `yuv420` emits planar Y/U/V with `width`
  and `height` parameters. Mono pictures ignore this setting.

## Usage

```yaml
nodes:
  - id: dora-dav1d
    build: cargo build --release --target-dir target
    path: target/release/dora-dav1d
    inputs:
      tick: encoder/encoded
    outputs:
      - tick
    env:
      ENCODING: bgr8
```

## Build

```bash
cargo build --release --target-dir target
```
