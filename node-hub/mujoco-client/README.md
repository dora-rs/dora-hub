# mujoco-client

Dora node client for a MuJoCo simulation. It steps the physics loop, publishes
the simulated robot's joint state, and applies incoming actuator commands — so a
MuJoCo model can stand in for a real robot in a dataflow.

## Behavior

The node loads a MuJoCo model from the `SCENE` XML file and opens a passive
viewer. It then reacts to input events:

- **`tick`** — echoes a `tick` output first (so downstream nodes don't stall
  during the step), advances the physics one `mj_step`, syncs the viewer, and
  publishes the joint state as `joint_state`. If the viewer window has been
  closed, the node exits.
- **`action`** — writes the incoming flat array into `data.ctrl`. Its length
  must equal the model's actuator count (`m.nu`); the values are applied in the
  XML `<actuator>` declaration order. A mismatched length raises an error.
- **`render`** — renders every camera declared in the model off-screen and
  publishes one JPEG per camera (output id = camera name). This branch only runs
  when off-screen rendering is enabled via the `--cameras` flag; it is not
  reachable through Hub environment configuration alone.

The node raises at startup if the model has no actuators (`m.nu == 0`).

## Inputs

- `tick`: step trigger — advances the simulation and publishes joint state.
- `action`: flat actuator command array, length `m.nu`, in XML actuator order.

> Off-screen camera rendering is **not part of the Hub contract**: the `render`
> input and its per-camera JPEG outputs are only available when the node is
> launched via `path:` with the `--cameras` flag, which a `hub:` spawn cannot
> pass. The `render` input is therefore omitted from `dora-node.yml`.

## Outputs

- `tick`: echo of the `tick` input, sent before stepping.
- `joint_state`: full generalised coordinates (`data.qpos`) as a float32 array.

Camera JPEG outputs (one per model camera, named after the camera) are produced
only when off-screen rendering is enabled via a `path:` launch (see note above).

## Environment variables

- `SCENE` (required): path to the MuJoCo scene XML file.
- `IMAGE_WIDTH` (default `960`): rendered camera image width in pixels — only
  used when camera rendering is enabled.
- `IMAGE_HEIGHT` (default `600`): rendered camera image height in pixels — only
  used when camera rendering is enabled.
- `JPEG_QUALITY` (default `90`): JPEG encoding quality 0-100 — only used when
  camera rendering is enabled.

## Usage

```yaml
nodes:
  - id: mujoco-client
    hub: mujoco-client@^0.5
    inputs:
      tick: dora/timer/millis/10
      action: controller/action
    outputs:
      - tick
      - joint_state
    env:
      SCENE: scene.xml
```

## Build

```bash
pip install .
```
