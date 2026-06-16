# dora-object-to-pose

Convert detected objects (2D bounding boxes or segmentation masks) plus a depth
frame into 3D poses. Masked/boxed pixels are projected into camera space using
the depth frame, rotated by the configured camera pitch, and reduced to one
`xyzrpy` pose per object.

## Behavior

`dora-object-to-pose` connects as a dora node and handles four input ids:

- **`depth`**: a UInt16 depth frame (millimetres). It is cached for later
  projection. Metadata parameters `height`, `width`, `focal`, and `resolution`,
  if present, override the projection defaults.
- **`image`**: a UInt8 image buffer. It is cached on receipt but is not used to
  compute poses.
- **`masks`**: segmentation masks (Float32 where `>0` means set, or Boolean),
  flattened into chunks of `height*width` pixels — one chunk per object. For
  each mask, the set pixels are projected into 3D using the cached `depth`
  frame and reduced to a pose. Emits `pose`.
- **`boxes2d`**: 2D bounding boxes as Int64 in chunks of 4
  `[x_min, y_min, x_max, y_max]`. For each box, the depth pixels inside the box
  are projected, filtered to the nearer half (by a mean/min-z threshold), and
  reduced to a pose. Emits `pose`.

Projection uses the depth value at each pixel divided by 1000, skipping empty or
too-far points, then rotates by `CAMERA_PITCH`. Each pose is computed from the
projected points' bounding extents and their x/y correlation (used as yaw). A
`pose` output is only produced when a `depth` frame has already been received;
otherwise the node logs that no depth frame was found. Any other input id is
logged and ignored.

## Inputs

- `depth` (required): UInt16 depth frame (mm), row-major `width*height`.
  Optional metadata parameters: `height`, `width`, `focal` (ListInt[2]),
  `resolution` (ListInt[2]).
- `masks`: Float32 (`>0` = set) or Boolean masks, flattened in `height*width`
  chunks (one per object). Produces `pose`.
- `boxes2d`: Int64 boxes in chunks of 4 `[x_min, y_min, x_max, y_max]`. Produces
  `pose`.
- `image`: UInt8 image buffer; cached but not used to compute poses.

## Outputs

- `pose`: flattened Float32 poses, 6 values per object
  `[x, y, z, roll(0), pitch(0), yaw]`. Carries an `encoding: xyzrpy` metadata
  parameter.

## Environment variables

- `CAMERA_PITCH` (float, default `2.47`): camera pitch in radians, used to
  rotate projected points into the world frame.

## Usage

Wire a depth frame plus a detection stream (`boxes2d` or `masks`) into the node:

```yaml
nodes:
  - id: dora-object-to-pose
    hub: dora-object-to-pose@^0.5
    inputs:
      depth: camera/depth
      boxes2d: detector/boxes2d
```

## Build

```bash
cargo build --release --target-dir target
```
