# lerobot-dashboard

Pygame dashboard for LeRobot dataset recording — displays two images side by side
plus a status text, and turns keyboard events into episode-recording control
outputs.

## Behavior

`lerobot-dashboard` opens a Pygame window and connects as a dora node. On each
`tick` it redraws the window (left image, right image, and a status text at the
bottom center) and processes pending keyboard events:

- **Space**: toggle recording. When starting, it emits `episode` with the current
  episode index; when stopping, it emits `episode` with `-1` and advances the
  episode index.
- **Return**: mark an episode as failed. While recording, it emits `failed` with
  the current episode index, advances the index, and emits `episode` with `-1`.
  When idle (and at least one episode exists), it emits `failed` with the previous
  episode index.
- **Window close (QUIT)**: ends the loop.

The status text is rendered locally (e.g. "Recording episode N"); it is not
emitted as an output. Each `tick` is echoed back on the `tick` output. When the
window closes, an empty `end` output is sent.

## Inputs

- `image_left`: image for the left half of the window (dora Arrow image struct
  with `width`/`height`/`channels`/`data`, interpreted as BGR).
- `image_right`: image for the right half of the window (same format).
- `tick`: refresh trigger; drives redraw and keyboard handling.

## Outputs

- `tick`: echoed once per received tick (empty array), forwarding the tick's
  metadata.
- `episode`: single-element array — the episode index when recording starts, or
  `-1` when recording stops.
- `failed`: single-element array — the index of the episode marked as failed.
- `end`: single empty-element array, emitted when the window is closed.

## Environment variables

- `WINDOW_WIDTH` (int, default `640`): window width in pixels.
- `WINDOW_HEIGHT` (int, default `480`): window height in pixels.

## Usage

```yaml
nodes:
  - id: lerobot-dashboard
    hub: lerobot-dashboard@^0.5
    inputs:
      tick: dora/timer/millis/16
      image_left: camera-left/image
      image_right: camera-right/image
    outputs:
      - tick
      - episode
      - failed
      - end
    env:
      WINDOW_WIDTH: 1280
      WINDOW_HEIGHT: 1080
```

## Build

```bash
pip install .
```
