# dora-reachy2

Camera node for the Reachy 2 robot package — streams stereo teleop (left/right)
and depth frames from a Reachy 2 over the network. This Hub manifest covers the
**camera** node of the `dora-reachy2` package.

## Behavior

On startup the node connects to the Reachy 2 robot at `ROBOT_IP`, retrying up to
ten times until the teleop and depth cameras are reachable, and reads the depth
camera intrinsics (focal length and resolution). On each `tick` it grabs a fresh
set of frames and emits them:

- the left teleop frame (`image_left`),
- the right teleop frame (`image_right`),
- the depth camera's color frame (`image_depth`), and
- the depth frame in meters (`depth`), with camera intrinsics in its metadata.

## Inputs

- `tick`: trigger to capture and emit one set of camera frames (wire a timer).

## Outputs

- `image_left`: left teleop camera frame (bgr8), flattened Arrow array; metadata
  carries width, height, encoding, primitive.
- `image_right`: right teleop camera frame (bgr8), flattened Arrow array;
  metadata carries width, height, encoding, primitive.
- `image_depth`: depth camera color frame (bgr8), flattened Arrow array;
  metadata carries width, height, encoding, primitive.
- `depth`: depth frame in meters (float64), flattened Arrow array; metadata
  carries width, height, focal, resolution, primitive.

## Environment variables

- `ROBOT_IP` (string, default `10.42.0.80`): IP address of the Reachy 2 robot to
  connect to.

## Usage

```yaml
nodes:
  - id: dora-reachy2-camera
    hub: dora-reachy2@^0.5
    inputs:
      tick: dora/timer/millis/33
    outputs:
      - image_left
      - image_right
      - image_depth
      - depth
    env:
      ROBOT_IP: 10.42.0.80
```

## Build

```bash
pip install .
```

> Note: the `dora-reachy2` package also ships arm, head, and mobile-base nodes
> as separate console scripts (`dora-reachy2-left-arm`, `dora-reachy2-right-arm`,
> `dora-reachy2-head`, `dora-reachy2-mobile-base`). Those nodes are **not**
> covered by this single Hub manifest, which targets only the camera node
> (`dora-reachy2-camera`).
