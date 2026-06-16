# dora-policy-inference

Load a pre-trained LeRobot policy and run it in real time to control a robot
from camera observations and the current robot joint state.

## Behavior

The node loads a LeRobot policy from `MODEL_PATH` on startup (device — CUDA / MPS
/ CPU — is selected automatically by LeRobot) and buffers every input it
receives. The camera input ids are taken from `CAMERA_NAMES` (default
`laptop,front`); the **first** camera in that list is the lead camera.

When the lead camera input arrives — and at most once per `1 / INFERENCE_FPS`
seconds — the node assembles an observation from the buffered camera images
(`observation.images.<camera>`) and the robot state (`observation.state`), runs
the policy, converts the predicted action from degrees to radians, and emits it
on `robot_action`. Camera images are reshaped from the input's `height`/`width`
metadata and color-converted per the `encoding` metadata (`bgr8`, `yuv420`).
Robot state is converted from radians to degrees and float32 before inference.

## Inputs

- `laptop`: lead camera image (Arrow uint8). Its arrival triggers inference.
  Must match the first entry of `CAMERA_NAMES`.
- `front`: secondary camera image (Arrow uint8). Must match an entry of
  `CAMERA_NAMES`. Add one input per camera you list.
- `robot_state`: current robot joint state in radians (Arrow numeric).

Image inputs rely on `height`, `width`, and optional `encoding` metadata.

## Outputs

- `robot_action`: predicted robot action, converted to radians.
- `status`: status / log messages (e.g. "Policy loaded successfully").

## Environment variables

| Variable           | Type   | Default        | Description                                                              |
| ------------------ | ------ | -------------- | ------------------------------------------------------------------------ |
| `MODEL_PATH`       | string | (required)     | Trained LeRobot policy model directory.                                  |
| `CAMERA_NAMES`     | string | `laptop,front` | Comma-separated camera input ids; first is the lead (inference trigger). |
| `TASK_DESCRIPTION` | string | `""`           | Task string for task-conditioned policies.                               |
| `INFERENCE_FPS`    | int    | `30`           | Max inference frequency (Hz).                                            |

## Usage

```yaml
nodes:
  - id: dora-policy-inference
    hub: dora-policy-inference@^0.5
    inputs:
      laptop: laptop_cam/image
      front: front_cam/image
      robot_state: robot/pose
    outputs:
      - robot_action
      - status
    env:
      MODEL_PATH: /path/to/your/lerobot/model
      CAMERA_NAMES: "laptop,front"
      TASK_DESCRIPTION: "pick up the cup"
      INFERENCE_FPS: 30

  - id: robot_controller
    path: your-robot-controller
    inputs:
      action: dora-policy-inference/robot_action
    outputs:
      - pose
```

## Build

```bash
pip install .
```
