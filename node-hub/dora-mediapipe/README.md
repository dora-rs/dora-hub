# dora-mediapipe

MediaPipe pose estimation — turns RGB image frames into 2D (and optional depth-derived 3D) body landmarks.

## Behavior

`dora-mediapipe` connects as a dora node and runs Google MediaPipe Pose on
incoming image frames. For each input whose id contains `image`, it decodes the
frame (`bgr8`, `rgb8`, or encoded `jpeg`/`jpg`/`jpe`/`bmp`/`webp`/`png`), runs
pose estimation, and emits the detected landmarks as flattened `(x, y)` pixel
coordinates on `points2d`. If no landmarks are found it prints
`No pose landmarks detected.`

If an input whose id contains `depth` has been received, its frame and
`focal_length`/`resolution` metadata are cached and used to additionally lift
each landmark into a 3D coordinate, emitted on `points3d`. Without a prior depth
input, only `points2d` is produced.

## Inputs

- **image** (required) — RGB/BGR or encoded image frame; matched by substring,
  so any input id containing `image` qualifies. Requires `encoding`, `width`,
  and `height` metadata.
- **depth** (optional) — depth frame; matched by substring (id containing
  `depth`). Requires `encoding`, `width`, `height`, `focal_length`, and
  `resolution` metadata.

## Outputs

- **points2d** — flattened `(x, y)` pixel coordinates of the detected pose
  landmarks.
- **points3d** — flattened `(x, y, z)` 3D coordinates of the landmarks, emitted
  only after a depth input has been received.

## Environment variables

None.

## Usage

```yaml
nodes:
  - id: dora-mediapipe
    hub: dora-mediapipe@^0.5
    inputs:
      image: camera/image
      depth: camera/depth      # required for the points3d output
    outputs:
      - points2d
      - points3d
```

## Build

```bash
pip install .
```
