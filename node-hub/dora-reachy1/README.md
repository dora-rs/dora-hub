# dora-reachy1-vision

Reachy 1 camera node. On any input tick it grabs the latest frame from the
robot's selected head camera over the Reachy SDK and emits it as a bgr8 image.

This package's `dora-reachy1-vision` entrypoint is the node described here. (The
package also ships a `dora-reachy1` arm/head control entrypoint, not covered by
this contract.)

## Behavior

On every input event the node reads `reachy.<right|left>_camera.last_frame` from
the Reachy SDK (selected by the `CAMERA` env var) and sends it on the `image`
output. The frame is flattened to a 1-D array; its shape is reported via
`width`/`height` metadata and the encoding is fixed to `bgr8`. Any input id
triggers a capture, so the input is treated as a generic `tick`.

## Inputs

- `tick`: capture trigger. Any input event grabs the latest camera frame and
  emits it; a timer is the typical source.

## Outputs

- `image`: latest camera frame as a flattened bgr8 array. Metadata carries
  `width`, `height`, and `encoding` (`"bgr8"`).

## Environment variables

- `ROBOT_IP` (default `10.42.0.24`): IP address of the Reachy robot.
- `CAMERA` (default `right`): which head camera to read; `right` selects the
  right camera, anything else selects the left.

## Usage

```yaml
nodes:
  - id: dora-reachy1-vision
    hub: dora-reachy1-vision@^0.5
    inputs:
      tick: dora/timer/millis/100
    outputs:
      - image
    env:
      ROBOT_IP: "10.42.0.24"
      CAMERA: right
```

## Build

```bash
pip install .
```
