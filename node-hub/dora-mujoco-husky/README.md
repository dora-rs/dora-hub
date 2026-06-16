# dora-mujoco-husky

A MuJoCo-based Clearpath Husky mobile-robot simulation node. It consumes
velocity commands, steps a physics-accurate Husky model, and emits the robot's
position and velocity feedback.

## Behavior

On startup the node downloads the required Husky meshes, loads the
`husky/husky.xml` MuJoCo model, and opens a passive viewer window.

For each `cmd_vel` input it:

- reads linear velocity from index 0 (clamped to [-2.0, 2.0] m/s) and angular
  velocity from index 5 (clamped to [-3.0, 3.0] rad/s),
- converts them to left/right wheel velocities (wheel radius 0.17775 m, track
  width 0.59 m) and applies them to the four wheel actuators,
- advances the simulation one step and syncs the viewer,
- emits the base `position` (qpos[:3]) and `velocity` (qvel[:3]).

Inputs other than `cmd_vel` are ignored.

## Inputs

- `cmd_vel`: velocity command array. Index 0 = linear velocity (m/s, clamped to
  [-2.0, 2.0]); index 5 = angular velocity (rad/s, clamped to [-3.0, 3.0]).

## Outputs

- `position`: base position `[x, y, z]` from MuJoCo `qpos`, float64.
- `velocity`: base velocity `[vx, vy, vz]` from MuJoCo `qvel`, float64.

## Environment variables

None.

## Usage

Drive the sim from a gamepad and inspect its output:

```yaml
nodes:
  - id: gamepad
    hub: gamepad@^0.5
    inputs:
      tick: dora/timer/millis/10
    outputs:
      - cmd_vel

  - id: mujoco_husky
    hub: dora-mujoco-husky@^0.5
    inputs:
      cmd_vel: gamepad/cmd_vel
    outputs:
      - position
      - velocity
```

## Build

```bash
pip install .
```
