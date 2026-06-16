# dora-cotracker

Real-time point tracking using Facebook's CoTracker model. Consumes an RGB image
stream plus query points (raw points or 2D bounding boxes) and emits the tracked
point positions and an annotated visualization frame.

## Behavior

On startup the node loads the `cotracker3_online` model via `torch.hub` (using
CUDA if available, otherwise CPU) and runs an online tracker over a sliding
window of frames.

- Each `image` frame is appended to the tracking window and reshaped to
  `(height, width, 3)` from its `width`/`height` metadata.
- Once the window is full, the model tracks the current query points and emits
  the visible points (visibility > 0.5) on `points`, plus an annotated frame on
  `tracked_image`.
- Query points come from `points` (a flattened x,y array) or `boxes2d` (bounding
  boxes, from which sample points are derived). New query points are accepted
  only after the previous batch has been consumed; an empty `boxes2d` clears the
  current points.
- When `INTERACTIVE_MODE` is true, an OpenCV window is shown and left-clicking
  adds tracking points interactively.

## Inputs

- `image` (required): RGB video frame as a flattened uint8 Arrow array; metadata
  must include `width` and `height`.
- `points`: query points to track, a flattened float array of x,y coordinates
  (reshaped to `(-1, 2)`).
- `boxes2d`: 2D bounding boxes (a `StructArray` with `bbox`/`labels`, or a flat
  array reshaped to `(-1, 4)`); sample points are derived from each box. An empty
  value clears the current query points.

## Outputs

- `points`: visible tracked point positions as a flattened float32 Arrow array.
  Metadata: `num_points`, `dtype`, `shape` `(N, 2)`, `width`, `height`.
- `tracked_image`: the input frame annotated with tracked points, carrying the
  input image metadata.

## Environment variables

- `INTERACTIVE_MODE` (bool, default `false`): when true, opens an OpenCV window
  showing the feed and lets you left-click to add tracking points. Requires a
  display; leave false for headless runs.

## Usage

```yaml
nodes:
  - id: camera
    path: opencv-video-capture
    inputs:
      tick: dora/timer/millis/100
    outputs:
      - image
    env:
      CAPTURE_PATH: "0"
      ENCODING: "rgb8"
      IMAGE_WIDTH: "640"
      IMAGE_HEIGHT: "480"
  - id: tracker
    hub: dora-cotracker@^0.5
    inputs:
      image: camera/image
    outputs:
      - points
      - tracked_image
```

## Build

```bash
pip install .
```
