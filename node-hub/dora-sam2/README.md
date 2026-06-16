# dora-sam2

SAM2 image segmentation node. Given an image plus a box or point prompt, it runs
the `facebook/sam2-hiera-large` predictor and emits the resulting boolean
segmentation masks.

> [!WARNING]
> SAM2 requires an Nvidia GPU to run — inference is wrapped in
> `torch.autocast("cuda", ...)`.

## Behavior

The node connects as a dora node and processes input events:

- **`image`** inputs are decoded according to `metadata["encoding"]` (raw
  `bgr8`/`rgb8`, or compressed `jpeg`/`jpg`/`jpe`/`bmp`/`webp`/`png`) using the
  `width` and `height` metadata, converted to RGB, and cached as a PIL image
  keyed by the input id. An unsupported encoding raises a `RuntimeError`. Image
  events do not themselves produce output (the SAM2 tracking path is currently
  disabled by an early `continue`).
- **`boxes2d`** inputs run box-prompted prediction. The value is either a plain
  xyxy Arrow array or a `StructArray` with `bbox` and `labels`; metadata must
  carry `encoding: xyxy` and an `image_id` referencing a previously received
  image input id. An empty value emits an empty `masks` array; otherwise the
  predicted masks are emitted on `masks`.
- **`points`** inputs run point-prompted prediction against the image named by
  `metadata["image_id"]` (defaulting to the first cached image), then emit
  `masks`. If no image has been received yet, the event is skipped.

Input ids are matched by substring (`"image" in id`, `"boxes2d" in id`,
`"points" in id`). When run via `hub:`, wire each input on its declared id
(`image`/`boxes2d`/`points`): the hub contract only permits declared input ids,
even though the code would also accept any id containing those tokens (e.g.
`camera_image`) when run via `path:`.

## Inputs

- `image`: image to segment. Metadata must carry `encoding`, `width`, `height`.
- `boxes2d`: xyxy box prompts. Metadata must carry `encoding: xyxy` and
  `image_id`. Triggers a `masks` output.
- `points`: x,y point prompts. Metadata may carry `image_id`. Triggers a `masks`
  output.

## Outputs

- `masks`: boolean segmentation masks flattened into an Arrow array. Metadata
  carries `image_id`, `width`, `height` (and `primitive: masks` for masks
  derived from a `boxes2d` request; the `points` path omits `primitive`).

## Environment variables

None. The model (`facebook/sam2-hiera-large`) is hardcoded.

## Usage

```yaml
nodes:
  - id: dora-sam2
    hub: dora-sam2@^0.5
    inputs:
      image: camera/image
      boxes2d: object-detector/boxes2d
      points: object-detector/points
    outputs:
      - masks
```

## Build

```bash
pip install .
```
