# dora-vggt

Runs the [VGGT](https://github.com/facebookresearch/vggt) (Visual Geometry
Grounded Transformer) `facebook/VGGT-1B` model on incoming images to predict
depth maps and camera intrinsics.

## Behavior

On each image input, the node decodes the frame (`bgr8`, `rgb8`, or encoded
`jpeg`/`jpg`/`jpe`/`bmp`/`webp`/`png`), buffers up to `VGGT_NUM_IMAGES` recent
frames, and runs them through VGGT. It predicts camera poses (extrinsic +
intrinsic, OpenCV convention) and a depth map. Low-confidence depth pixels
(confidence < 1.0) are zeroed.

It then emits two outputs: a depth map (scaled by `SCALE_FACTOR`, encoded per
`DEPTH_ENCODING`) and the preprocessed RGB image. Both carry the predicted focal
length and principal point in metadata. If CUDA is available the model runs on
GPU, otherwise CPU. The `facebook/VGGT-1B` weights download automatically on
first run.

The node keys off the substring `image` in the input id: the depth output id is
that id with `image` replaced by `depth`, and the RGB output reuses the input id
unchanged.

## Inputs

- `image`: an image whose id contains `image`. Value is a flat pixel buffer (or
  encoded bytes); metadata must provide `encoding`, `width`, and `height`.

## Outputs

- `depth`: predicted depth map (input id with `image` -> `depth`). Flat array;
  metadata has `width`, `height`, `encoding` (`float64` raw or `mono16` scaled
  ×1000), `focal` `[fx, fy]`, `resolution` `[cx, cy]` (principal point).
- `image`: preprocessed RGB image echoed under the original input id. Flat
  `uint8` buffer; metadata has `encoding: rgb8`, `width`, `height`, `focal`,
  `resolution`.

## Environment variables

- `SCALE_FACTOR` (float, default `1.0`): multiplier applied to the depth map.
- `VGGT_NUM_IMAGES` (int, default `2`): number of recent images buffered and fed
  to the model together.
- `DEPTH_ENCODING` (string, default `float64`): `float64` emits raw depth;
  `mono16` multiplies by 1000 and casts to `uint16`.

## Usage

```yaml
nodes:
  - id: dora-vggt
    hub: dora-vggt@^0.5
    inputs:
      image: camera/image
    outputs:
      - depth
      - image
    env:
      VGGT_NUM_IMAGES: 2
      DEPTH_ENCODING: float64
```

## Build

```bash
pip install .
```
