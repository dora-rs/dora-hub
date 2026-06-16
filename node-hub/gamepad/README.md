# gamepad

Read a connected gamepad/joystick (via pygame) and emit robot velocity commands
plus the raw controller state. Defaults are tuned for the Logitech F710 but any
standard controller works after adjusting the axis/button mapping.

## Behavior

On startup the node initializes pygame, opens joystick index `0`, and asserts a
controller is present (it exits with an error if none is found). It then runs as
a dora node: on **every** received event it pumps pygame, reads all axes,
buttons, and the first D-pad hat, and builds a velocity command:

- D-pad vertical → linear X, D-pad horizontal → linear Y
- Right stick Y → linear Z, right stick X → angular Z
- Left stick Y → angular X, left stick X → angular Y

Stick axes below `JOYSTICK_DEADZONE` (absolute value) are zeroed. Linear
components are scaled by `MAX_LINEAR_SPEED`, angular components by
`MAX_ANGULAR_SPEED`. When any of the six components is non-zero it sends both
`cmd_vel` and `raw_control`. The node does not inspect the input id, so it must
be driven by a trigger (wire a timer to `tick`). On shutdown it sends a single
zero `cmd_vel`.

## Inputs

- `tick`: poll trigger. Each event reads the current gamepad state and emits
  outputs. Wire a timer (e.g. `dora/timer/millis/10`).

## Outputs

- `cmd_vel`: 6-element `float64` array `[linear_x, linear_y, linear_z,
  angular_x, angular_y, angular_z]`. Sent only when any component is non-zero; a
  zero vector is sent once on shutdown.
- `raw_control`: full controller state as a single JSON string — `axes`,
  `buttons`, `hats`, and the axis/button name `mapping`. Sent alongside each
  non-zero `cmd_vel`.

## Environment variables

- `MAX_LINEAR_SPEED` (default `0.05`): scale for linear velocity components (m/s).
- `MAX_ANGULAR_SPEED` (default `0.8`): scale for angular velocity components (rad/s).
- `JOYSTICK_DEADZONE` (default `0.2`): absolute axis magnitude below which stick
  input is treated as zero.

## Usage

```yaml
nodes:
  - id: gamepad
    hub: gamepad@^0.5
    inputs:
      tick: dora/timer/millis/10
```

## Build

```bash
pip install .
```
