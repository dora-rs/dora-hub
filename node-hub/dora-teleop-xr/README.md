# dora-teleop-xr

XR teleoperation node. It serves a WebXR/VR teleop session (via
[teleop-xr](https://github.com/qrafty-ai/teleop_xr)), runs inverse kinematics on
incoming controller poses, and on each tick emits the latest head/hand poses,
the robot joint state, and the raw XR state.

## Behavior

On startup the node loads the robot class named by `ROBOT_CLASS`, builds an IK
solver and controller for it, and starts a teleop HTTPS server (host and port
come from the teleop-xr settings defaults). A background worker thread solves IK
whenever new XR state arrives and publishes joint updates back to the connected
WebXR client.

The node loop is driven by the `tick` input. On each tick, if an XR state has
been received, it extracts the head device pose and the left/right grip poses,
publishes any that are present, publishes the current robot joint configuration
(reordered into the robot's rerun URDF joint order when available), and
publishes the complete raw XR state as JSON. Ticks received before any XR state
has arrived are ignored.

## Inputs

- `tick` (required): poll trigger. Each tick reads the latest received XR state
  and publishes outputs. Wire it to a timer such as `dora/timer/millis/10`.

## Outputs

- `head_pose`: headset pose, `float32[7]` `[x, y, z, qx, qy, qz, qw]`. Emitted
  only when a head pose is present. Metadata: `primitive=pose`,
  `encoding=xyzquat`.
- `left_hand_pose`: left controller/hand grip pose, `float32[7]`. Emitted only
  when present. Metadata: `primitive=pose`, `encoding=xyzquat`.
- `right_hand_pose`: right controller/hand grip pose, `float32[7]`. Emitted only
  when present. Metadata: `primitive=pose`, `encoding=xyzquat`.
- `joint_state`: current robot joint configuration, `float32` array, reordered
  into the robot's rerun URDF joint order when available. Metadata:
  `primitive=jointstate`, `encoding=jointstate`.
- `raw_xr_state`: the complete raw XR state as a single JSON string.

## Environment variables

- `ROBOT_CLASS` (default `teleop_xr.ik.robots.so101:SO101Robot`): Python class
  path (`module:ClassName`) of the robot used to build the IK solver and report
  joint names.

## Usage

```yaml
nodes:
  - id: teleop-xr
    hub: dora-teleop-xr@^0.5
    inputs:
      tick: dora/timer/millis/10
    outputs:
      - head_pose
      - left_hand_pose
      - right_hand_pose
      - joint_state
      - raw_xr_state
    env:
      ROBOT_CLASS: "teleop_xr.ik.robots.so101:SO101Robot"
```

## Build

```bash
pip install .
```
