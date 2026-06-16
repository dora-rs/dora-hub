# dora-dataset-record

Record synchronized camera feeds and robot state/action streams into a
HuggingFace [LeRobot](https://github.com/huggingface/lerobot) dataset, for
imitation learning and robot training.

## Behavior

The node connects as a dora node and a background timer adds one frame to the
dataset every `1/FPS` seconds while an episode is active. Each frame is built
from the latest values it has received per input:

- `robot_action` is stored as the `action` feature.
- `robot_state` is stored as the `observation.state` feature.
- Any input whose metadata carries `height` and `width` is treated as a camera
  frame and stored as `observation.images.<input-id>` (the input id must be one
  of `CAMERA_NAMES`). `bgr8` and `yuv420` encodings are converted to RGB.
- When `SAVE_AVIF_FRAMES=true`, inputs named `rav1e_<camera>` are written
  verbatim as `.avif` files alongside the dataset.

A frame is only added once every feature declared by `ROBOT_JOINTS` and
`CAMERA_NAMES` is present. Recording runs for `TOTAL_EPISODES` episodes of
`EPISODE_DURATION_S` each, separated by `RESET_DURATION_S` reset phases (input
during a reset phase is dropped), then the node finalizes the dataset
(optionally pushing it to the Hub) and stops. Status messages are emitted on the
`text` output.

## Inputs

- `robot_state`: robot observation pose, a float array with one value per
  `ROBOT_JOINTS` entry (radians). Stored as `observation.state`.
- `robot_action`: robot action pose, a float array with one value per
  `ROBOT_JOINTS` entry (radians). Stored as `action`.
- `<camera-name>`: a camera frame carrying `height`/`width` metadata, stored as
  `observation.images.<camera-name>`. Wire one input per entry in
  `CAMERA_NAMES` (e.g. `laptop`, `front`).
- `rav1e_<camera-name>`: optional AVIF-encoded frame for a camera, used only
  when `SAVE_AVIF_FRAMES=true`.

A `hub:` contract must declare every wired input, so the contract declares
`robot_state`, `robot_action`, and example `laptop` / `rav1e_laptop` camera
inputs. Camera input ids are dynamic (driven by `CAMERA_NAMES`); rename or add
camera inputs to match your cameras.

## Outputs

- `text`: status messages about the recording lifecycle (episode start, reset
  phase, episode saved, completion).

## Environment variables

Required:

- `REPO_ID`: HuggingFace dataset repo id (e.g. `username/dataset_name`).
- `ROBOT_JOINTS`: comma-separated joint names defining the `action` /
  `observation.state` shape.

Optional:

- `SINGLE_TASK` (default `Your task`): task description attached to each frame.
- `CAMERA_NAMES` (default unset): comma-separated camera names. Each name needs
  a matching input and a `CAMERA_<NAME>_RESOLUTION` variable. No cameras are
  recorded if unset.
- `CAMERA_<NAME>_RESOLUTION`: per-camera resolution as `height,width,channels`
  (e.g. `CAMERA_LAPTOP_RESOLUTION: 480,640,3`). One per camera name; not set if
  the camera is omitted.
- `FPS` (default `30`): recording frame rate.
- `TOTAL_EPISODES` (default `10`): number of episodes to record.
- `EPISODE_DURATION_S` (default `60`): episode length in seconds.
- `RESET_DURATION_S` (default `15`): reset break between episodes in seconds.
- `USE_VIDEOS` (default `true`): encode frames as MP4 video, else store images.
- `SAVE_AVIF_FRAMES` (default `false`): also save AVIF frames from
  `rav1e_<camera>` inputs.
- `ROBOT_TYPE` (default `your_robot_type`): robot type in dataset metadata.
- `ROOT_PATH` (default unset): local dataset directory; LeRobot cache if unset.
- `IMAGE_WRITER_PROCESSES` (default `0`): LeRobot image-writer subprocesses.
- `IMAGE_WRITER_THREADS` (default `4`): LeRobot image-writer threads per camera.
- `PUSH_TO_HUB` (default `false`): upload the dataset when finished.
- `PRIVATE` (default `false`): make the pushed dataset private.
- `TAGS` (default unset): comma-separated tags attached when pushing to Hub.

## Usage

```yaml
nodes:
  - id: dataset-record
    hub: dora-dataset-record@^0.5
    inputs:
      laptop: laptop-cam/image
      front: front-cam/image
      robot_state: robot-follower/pose
      robot_action: leader-interface/pose
    outputs:
      - text
    env:
      REPO_ID: your_username/your_dataset_name
      SINGLE_TASK: Pick up the cube and place it in the box
      ROBOT_JOINTS: joint1,joint2,joint3,joint4,joint5,gripper
      CAMERA_NAMES: laptop,front
      CAMERA_LAPTOP_RESOLUTION: 480,640,3
      CAMERA_FRONT_RESOLUTION: 480,640,3
      FPS: 30
      TOTAL_EPISODES: 50
```

## Build

```bash
pip install .
```
