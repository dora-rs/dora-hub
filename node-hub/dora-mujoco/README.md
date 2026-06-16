# dora-mujoco

A MuJoCo physics simulation node for the Dora dataflow framework. It loads a
robot model, steps the physics once per received event, applies actuator
control inputs, and emits joint and sensor state.

## Behavior

On startup the node loads a model named by the `MODEL_NAME` environment
variable. If `MODEL_NAME` is a path to an existing `.xml` file it is loaded
directly; otherwise it is treated as a `robot_descriptions` name and loaded with
its `scene` variant. The model is reset to its first keyframe (or to zero if it
has none), and a passive MuJoCo viewer is launched for live visualization.

The node then runs the Dora event loop. For every event it receives it steps the
physics simulation once and syncs the viewer. When the event is an input, after
stepping it emits the current state (`joint_positions`, `joint_velocities`,
`actuator_controls`, and `sensor_data` when the model has sensors). A
`control_input` event additionally applies its values to the model's actuators
before the step — each value is clamped to that actuator's control range, and
values beyond the actuator count are ignored.

## Inputs

- `tick` (required): drives the simulation. Each received event steps the
  physics once and triggers the state outputs. Wire to a timer such as
  `dora/timer/millis/2` for ~500 Hz.
- `control_input` (optional): actuator commands as a float64 Arrow array,
  applied in actuator order and clamped to each actuator's control range.

## Outputs

- `joint_positions`: joint positions (MuJoCo `qpos`), float64 Arrow array.
  Metadata: `timestamp`, `encoding: jointstate`.
- `joint_velocities`: joint velocities (MuJoCo `qvel`), float64 Arrow array.
  Metadata: `timestamp`.
- `actuator_controls`: current actuator commands (MuJoCo `ctrl`), float64 Arrow
  array. Metadata: `timestamp`.
- `sensor_data`: sensor readings (MuJoCo `sensordata`), float64 Arrow array.
  Emitted only when the model defines sensors. Metadata: `timestamp`.

## Environment variables

- `MODEL_NAME` (default `go2_mj_description`): robot to load — a
  `robot_descriptions` name (loaded as its `scene` variant) or a path to a
  MuJoCo `.xml` model file.

## Usage

Drive the simulation with a timer and (optionally) wire a controller into
`control_input`:

```yaml
nodes:
  - id: mujoco-sim
    hub: dora-mujoco@^0.5
    inputs:
      tick: dora/timer/millis/2
      control_input: controller/output
    outputs:
      - joint_positions
      - joint_velocities
      - actuator_controls
      - sensor_data
    env:
      MODEL_NAME: go2_mj_description
```

## Build

```bash
pip install .
```
