# dora-pytorch-kinematics

Forward and inverse kinematics for a serial robot chain, built on
[`pytorch_kinematics`](https://github.com/UM-ARM-Lab/pytorch_kinematics).
It converts between joint states and end-effector poses, and can compute the
manipulator Jacobian.

## Behavior

On startup the node loads a robot from a URDF (`URDF_PATH`, or a
`robot_descriptions` module named by `MODEL_NAME`) and builds a serial chain to
`END_EFFECTOR_LINK`. It keeps a `last_known_state` of joint angles (initialized
to zeros) that is updated on every IK/FK request.

What it computes depends on the input:

- **`cmd_vel`** input: runs FK on the last known joint state, adds the supplied
  end-effector velocity, then solves IK back to joint angles. Emits the solved
  joint angles on the `cmd_vel` id with `encoding = jointstate`.
- **Any other input**: dispatches on the input's `encoding` metadata field:
  - `xyzquat` — target pose as position + wxyz quaternion -> IK -> joint angles
    (`encoding = jointstate`).
  - `xyzrpy` — target pose as position + roll/pitch/yaw -> IK -> joint angles,
    rejecting solutions whose per-joint delta exceeds ±pi (`encoding = jointstate`).
  - `jointstate` — joint angles -> FK -> end-effector pose as xyz + rpy
    (`encoding = xyzrpy`).
  - `jacobian` — joint angles -> flattened 6xN Jacobian (`encoding =
    jacobian_result`, with a `jacobian_shape` metadata field).

The output is always sent on the **same id** as the input that triggered it.
IK requests that fail to converge are skipped (no output for that frame).

## Inputs

- `cmd_vel`: end-effector velocity command (float32). Drives the FK-then-IK
  velocity path described above.
- `pose`: a generic kinematics request. The operation is chosen by the input's
  `encoding` metadata (`xyzquat`, `xyzrpy`, `jointstate`, or `jacobian`), not by
  the id. Wire these requests on the `pose` id: the code accepts any non-`cmd_vel`
  id and echoes its result on the same id, but the hub contract only permits the
  declared `cmd_vel` and `pose` inputs.

## Outputs

- `cmd_vel`: IK solution joint angles produced from a `cmd_vel` input
  (`encoding = jointstate`).
- `pose`: the result of the request, emitted on the same id as the input.
  Encoding is `xyzrpy` (FK), `jointstate` (IK), or `jacobian_result` (Jacobian).

## Environment variables

- `URDF_PATH` — path to the robot URDF file. **Required** unless `MODEL_NAME`
  is set; if unset or the path does not exist, the node falls back to
  `MODEL_NAME`.
- `MODEL_NAME` — `robot_descriptions` module name (e.g. `iiwa7_description`)
  used to fetch a URDF/MJCF when `URDF_PATH` is unavailable. **Required** if
  `URDF_PATH` is not provided.
- `END_EFFECTOR_LINK` — name of the end-effector link in the URDF. **Required.**
- `TRANSFORM` — base transform as a space-separated `x y z qw qx qy qz` string
  (position + wxyz quaternion). Default `"0. 0. 0. 1. 0. 0. 0."`.
- `POSITION_TOLERANCE` — IK position convergence tolerance in meters. Default
  `0.005`.
- `ROTATION_TOLERANCE` — IK rotation convergence tolerance in radians. Default
  `0.05`.

## Usage

```yaml
nodes:
  - id: dora-pytorch-kinematics
    build: pip install dora-pytorch-kinematics
    path: dora-pytorch-kinematics
    inputs:
      pose: robot-controller/target_pose
    outputs:
      - pose
    env:
      MODEL_NAME: iiwa7_description
      END_EFFECTOR_LINK: iiwa_link_ee
```

Downstream nodes set the request type via the `encoding` metadata of the value
they send (`xyzquat`, `xyzrpy`, `jointstate`, or `jacobian`).

## Build

```bash
pip install .
```
