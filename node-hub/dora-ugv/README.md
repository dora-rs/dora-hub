# dora-ugv

Dora node for controlling an Agilex UGV (Hunter) over CAN — sends linear/angular
motion commands and emits the robot's measured velocity.

## Behavior

On startup the node connects to a Hunter robot (protocol `AGX_V2`) on the CAN
interface named by `CAN_BUS`, prints the protocol and robot versions, and enables
commanded mode.

For each input event:

- An **`action`** input is read as a 2-element array `[linear_velocity,
  angular_velocity]` and forwarded to the robot via `set_motion_command`.
- **Any other input id** triggers a read of the current robot state and publishes
  the measured `[linear_velocity, angular_velocity]` on the `velocity` output.

Requires the `ugv_sdk_py` package (see Build); the node exits at import time if it
is missing.

## Inputs

- `action`: motion command, a 2-element array `[linear_velocity,
  angular_velocity]`.
- `tick`: any non-`action` input; used to poll the robot and emit `velocity`
  (e.g. wire a timer here).

## Outputs

- `velocity`: measured robot velocity, a 2-element array `[linear_velocity,
  angular_velocity]`.

## Environment variables

- `CAN_BUS` (default `can0`): CAN bus interface the robot is connected on.

## Usage

```yaml
nodes:
  - id: dora-ugv
    hub: dora-ugv@^0.5
    inputs:
      action: command-source/action
      tick: dora/timer/millis/100
    outputs:
      - velocity
    env:
      CAN_BUS: can0
```

## Build

```bash
pip install .
```

Also requires `ugv_sdk_py`, installed separately by following the
[ugv_sdk build instructions](https://github.com/westonrobot/ugv_sdk/tree/main?tab=readme-ov-file#build-the-package-as-a-python-package).
