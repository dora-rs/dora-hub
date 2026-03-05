# OpenArm Grasp Example

Collision-aware grasp and pick-and-place planning for OpenArm. Two ways to run:

1. **Standalone scripts** — detect an object, plan a trajectory, save JSON + GIF
2. **Dora dataflow** — end-to-end pipeline with VLM + SAM3 + motion planning + CAN playback

## Quick Start

### 1. Plan a pick-and-place trajectory

```bash
ssh champagne@172.18.128.207
source ~/miniconda3/etc/profile.d/conda.sh && conda activate base
cd ~/dora-hub/examples/openarm-grasp

# Pick the yellow cube and place it in the green box
python grasp_trajectory.py \
    --config openarm-config.json \
    --targets '{"p1": [316, 382], "p2": [359, 382], "place": [609, 528]}' \
    --camera champagne-realsense --device cuda
```

- `p1` / `p2` are the two gripper jaw contact points on the yellow cube (0-1000 normalized pixels)
- `place` is the center of the green box where the cube should be dropped
- Outputs: `output/target_0_trajectory.json`, `output/target_0_trajectory.gif`, `output/target_0_final.png`

### 2. Grasp-only (no place)

```bash
# Omit "place" for a simple grasp trajectory (home → pre-grasp → grasp)
python grasp_trajectory.py \
    --config openarm-config.json \
    --targets '{"p1": [316, 382], "p2": [359, 382]}' \
    --camera champagne-realsense --device cuda
```

`grasp_trajectory.py` will:
- Capture depth from the RealSense
- Try both arms, pick the one with lowest collision count
- Build a point cloud and detect the table plane
- Optimize a collision-free trajectory:
  - **Grasp-only** (no `place`): home → pre-grasp → grasp (2-phase)
  - **Pick-and-place** (with `place`): 6-phase trajectory with gripper open/close and return-to-home
- Save `output/<label>_trajectory.json`, `output/<label>_trajectory.gif`, `output/<label>_final.png`

### 3. Copy results locally (optional)

```bash
scp champagne@172.18.128.207:~/dora-hub/examples/openarm-grasp/output/target_0_* examples/openarm-grasp/output/
```

### 4. Visualize an existing trajectory (optional)

```bash
cd ~/dora-hub/node-hub/dora-motion-planner
python tests/viz_trajectory.py path/to/trajectory.json
```

### 5. Play back over CAN

The `pick-and-place.yml` dataflow handles playback automatically (`PLAYBACK: "true"`).
For manual playback from a saved trajectory JSON:

```python
from dora_motion_planner.trajectory_json import load
import base64, can, time

traj, meta = load("output/target_0_trajectory.json")
dt = meta["dt"]

bus = can.interface.Bus(channel="can2", interface="socketcan", fd=False)

# Enable motors
for mid in range(0x01, 0x08):
    bus.send(can.Message(arbitration_id=mid, data=b'\xff' * 6 + b'\xff\xfc'))
    bus.recv(timeout=0.5)

# Play CAN frames from JSON
import json
with open("output/target_0_trajectory.json") as f:
    doc = json.load(f)

t_start = time.monotonic()
for i, cmd in enumerate(doc["commands"]):
    for frame in cmd["frames"]:
        bus.send(can.Message(
            arbitration_id=int(frame["id"], 16),
            data=base64.b64decode(frame["data"]),
        ))
    for _ in range(7):
        bus.recv(timeout=0.05)
    sleep = t_start + (i + 1) * dt - time.monotonic()
    if sleep > 0:
        time.sleep(sleep)

# Disable motors
for mid in range(0x01, 0x08):
    bus.send(can.Message(arbitration_id=mid, data=b'\xff' * 6 + b'\xff\xfd'))
bus.shutdown()
```

## Target JSON Format

Multi-target:
```json
{"targets": [
  {"label": "yellow_cube", "p1": [316, 382], "p2": [359, 382], "place": [609, 528]}
]}
```

Single-target (also accepted):
```json
{"p1": [316, 382], "p2": [359, 382], "place": [609, 528]}
```

Grasp-only (no place):
```json
{"p1": [316, 382], "p2": [359, 382]}
```

Coordinates are 0-1000 normalized (same as `grasp_selector.py` output).

## Dora Dataflows

### Full pick-and-place pipeline

`pick-and-place.yml` is the main production dataflow:

```
camera → SAM3 (text-prompted segmentation)
       → grasp_selector (candidate ranking via VLM)
       → motion_planner (collision-aware trajectory + built-in playback + gripper)
       → arm driver (CAN)
```

The motion planner has built-in playback (`PLAYBACK: "true"`) and sends `joint_command` + `gripper_command` directly to the arm — no separate trajectory player node needed.

Set `TARGET_OBJECT` in the selector to change what to grasp.
Set `PLACE_CONTAINER` in the selector to enable pick-and-place (the grasp result will include a `place` pixel and the motion planner builds a 6-phase trajectory with gripper open/close).

```bash
# On champagne (requires RealSense camera + CAN arm)
dora up
dora build pick-and-place.yml
dora start pick-and-place.yml
# ... pick-and-place happens automatically when the target is detected ...
dora stop --name <id>
dora destroy
```

### Other dataflows

| Dataflow | Description |
|---|---|
| `pick-and-place.yml` | Full pipeline: camera + SAM3 + VLM + motion planner (built-in playback) + arm (champagne) |
| `openarm-grasp-motion.yml` | Older full pipeline using separate trajectory_player node |
| `test-pick-place.yml` | Full pipeline with RealSense + fake CAN server (no physical arm) |
| `test-pick-place-oneshot.yml` | Vision-only (static image, no motion planner) for baguette |
| `test-grasp-motion.yml` | Vision + motion planner + fake CAN (single-shot, champagne) |
| `test-motion-only.yml` | Motion planner only with hardcoded grasp trigger (no VLM) |
| `test_vlm_critic_sam3_static.yml` | Vision-only with static image (no motion planner, champagne) |
| `test_vlm_critic_sam3.yml` | Vision-only with live camera (no motion planner, champagne) |
| `openarm-grasp.yml` | Original dataflow with separate IK node + grasp_planner state machine |

## Configuration

Camera transforms and RealSense relay paths are stored in `openarm-config.json`. The standalone scripts read this file via `--config`. The dora dataflows have the values inline in their YAML env vars.

## Files

| File | Purpose |
|---|---|
| `grasp_trajectory.py` | Standalone trajectory planner: targets JSON → collision-free trajectory + GIF |
| `white_handle_grasp.py` | HSV-based white handle detector → targets JSON (feeds into `grasp_trajectory.py`) |
| `grasp_selector.py` | Dora node: SAM3 mask + VLM rating → grasp point selection (with optional place detection) |
| `grasp_planner.py` | Dora node: original state-machine grasp planner (used by `openarm-grasp.yml`) |
| `grasp_trigger.py` | Dora node: sends a hardcoded grasp_result after warmup frames (for testing motion planner) |
| `test_static_image_sender.py` | Dora node: sends a saved PNG image on tick (for testing without a camera) |
| `test_visualize_grasp.py` | Dora node: draws grasp points on images and saves annotated output |
| `openarm-config.json` | Camera transforms and relay paths (for standalone scripts) |
| `output/` | Generated trajectories, GIFs, and annotated images |
