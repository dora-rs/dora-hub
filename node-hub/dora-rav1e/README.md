# dora-rav1e

AV1 video encoder built on [rav1e](https://crates.io/crates/rav1e). Encodes raw
image frames into AV1 or AVIF byte streams.

## Behavior

`dora-rav1e` connects as a dora node and encodes every image frame it receives.
For each input it:

- Reads `width`/`height` from the input metadata (falling back to the
  `IMAGE_WIDTH`/`IMAGE_HEIGHT` env vars) and the pixel format from the metadata
  `encoding` parameter (default `bgr8`).
- Converts the frame to YUV (or mono for `mono16`/`z16`) and encodes it with
  rav1e using the `RAV1E_SPEED` preset and low-latency settings.
- Wraps the result as AVIF when `ENCODING=avif`, otherwise emits raw AV1
  packets.

Supported input pixel formats: `bgr8`, `rgb8`, `yuv420`, `mono16`, `z16`. For
`mono16`/`z16`, zero pixels are filled toward the row center unless
`FILL_ZEROS=false`. Any other format triggers an `unimplemented!` panic.

The encoded frame is sent back out under the **same input id** it arrived on.

## Inputs

- `image`: raw image frame to encode. Arrow uint8 (`bgr8`, `rgb8`, `yuv420`) or
  uint16 (`mono16`, `z16`). Pixel format, width, and height are taken from the
  input metadata when present.

The node re-emits under whatever id it received, so when launched via `path:` the
input may be wired under any id; `image` is the conventional one. Under a `hub:`
contract, wire the input on the declared `image` id — the hub permits only
declared input ids.

## Outputs

- `image`: encoded frame as Arrow uint8 bytes. Output metadata carries
  `encoding` (`av1` or `avif`), `primitive: image`, and for AV1 also `width` and
  `height`.

## Environment variables

- `IMAGE_HEIGHT` (int, default `480`): default frame height when no `height`
  input metadata is present.
- `IMAGE_WIDTH` (int, default `640`): default frame width when no `width` input
  metadata is present.
- `RAV1E_SPEED` (int, default `10`): rav1e speed preset (0 = slowest/best
  quality, 10 = fastest).
- `ENCODING` (string, default `av1`): output codec/container — `av1` for raw AV1
  packets, `avif` for AVIF-wrapped output.
- `FILL_ZEROS` (bool, default `true`): for `mono16`/`z16`, fill zero pixels
  toward the row center before encoding. Set to `false` to disable.

## Usage

```yaml
nodes:
  - id: dora-rav1e
    hub: dora-rav1e@^0.3
    inputs:
      image: camera/image
    env:
      ENCODING: avif
      IMAGE_WIDTH: 640
      IMAGE_HEIGHT: 480
```

## Build

```bash
cargo build --release --target-dir target
```
