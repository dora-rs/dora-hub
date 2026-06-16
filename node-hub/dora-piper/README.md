# dora-piper

Dora node (experimental) to drive an AgileX Piper robot arm via the `piper_sdk` CAN interface.

## Behavior

On startup the node opens the CAN port (`C_PiperInterface(CAN_BUS).ConnectPort()`).
Unless `TEACH_MODE` is set, it enables the arm (`EnableArm`), waits up to 5s for all
six motors to report enabled, zeroes the joints and gripper, then enters the event loop.

For each input event:

- **`joint_action`** — interprets the value as 7 floats (6 joint positions in radians +
  gripper) and applies them via `JointCtrl` / `GripperCtrl`. Skipped in teach mode and
  rate-limited to ~20Hz.
- **`eef_action`** — interprets the value as 7 floats (end-effector X/Y/Z, RX/RY/RZ +
  gripper) and applies them via `EndPoseCtrl` / `GripperCtrl`. Skipped in teach mode and
  rate-limited to ~20Hz.
- **any other input id** — treated as a state-poll trigger: reads the arm joint, end-pose,
  and gripper messages from the SDK and emits the `jointstate`, `pose`, and `gripper`
  outputs.

On `STOP` (unless teach mode) it zeroes the joints and gripper, waits 5s, and exits.

Make sure to follow the setup and installation steps from
https://github.com/agilexrobotics/piper_sdk, that the CAN bus is activated, and that no
leader arm is connected.

## Inputs

- `joint_action` — target joint positions (7 floats: 6 joints + gripper). Ignored in teach mode.
- `eef_action` — target end-effector pose (7 floats: X/Y/Z/RX/RY/RZ + gripper). Ignored in teach mode.
- any other input id triggers a one-shot read of the current arm state (see Outputs).

## Outputs

- `jointstate` — current arm joint state as 7 float32 values (6 joints in radians + gripper opening).
- `pose` — current end-effector pose as 6 float32 values (X/Y/Z in meters, RX/RY/RZ in radians).
- `gripper` — current gripper opening as a single float32.

All three are emitted together when a state-poll input (any id other than
`joint_action` / `eef_action`) is received.

## Environment variables

- `CAN_BUS` — CAN bus interface name passed to `C_PiperInterface` (default: empty string, uses the SDK default).
- `TEACH_MODE` — when `True`/`true`, the arm is not enabled and command inputs are ignored; used to read state while hand-guiding the arm (default: `False`).

## Usage

```yaml
nodes:
  - id: dora-piper
    hub: dora-piper@^0.5
    inputs:
      joint_action: some-controller/action
    outputs:
      - jointstate
      - pose
      - gripper
```

## Build

```bash
pip install .
```
