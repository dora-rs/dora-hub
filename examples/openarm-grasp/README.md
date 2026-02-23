# OpenArm Grasp Example

Collision-aware grasp planning for OpenArm. Two ways to run:

1. **Standalone scripts** — detect an object, plan a trajectory, save JSON + GIF
2. **Dora dataflow** — end-to-end pipeline with VLM + SAM3 + motion planning + CAN playback

## Quick Start

### 1. Detect a grasp target

```bash
cd ~/dora-hub/examples/openarm-grasp

# Detect a white handle via HSV filtering
python white_handle_grasp.py --config openarm-config.json --camera champagne-realsense
```

This captures a color + depth frame, detects the white handle, and writes
`output/white_handle_targets.json` with normalized jaw coordinates.

### 2. Plan a collision-free trajectory

```bash
# Plan from detected targets (outputs trajectory JSON + animated GIF)
python grasp_trajectory.py \
    --config openarm-config.json \
    --targets output/white_handle_targets.json \
    --camera champagne-realsense

# Or plan from inline coordinates (skip detection)
python grasp_trajectory.py \
    --config openarm-config.json \
    --targets '{"p1": [626, 492], "p2": [695, 447]}'

# Detect + plan in one shot
python white_handle_grasp.py --config openarm-config.json --camera champagne-realsense --plan
```

`grasp_trajectory.py` will:
- Capture depth from the RealSense
- Try both arms, pick the one with lowest IK error
- Build a point cloud and detect the table plane (camera frustum projection)
- Optimize a two-phase collision-free trajectory (home → pre-grasp → grasp)
- Save `output/<label>_trajectory.json`, `output/<label>_trajectory.gif`, `output/<label>_final.png`

### 3. Visualize an existing trajectory (optional)

```bash
cd ~/dora-hub/node-hub/dora-motion-planner
python tests/viz_trajectory.py path/to/trajectory.json
```

### 4. Play back over CAN

Use the trajectory player node in a dora dataflow, or send the CAN frames directly:

```python
from dora_motion_planner.trajectory_json import load
import base64, can, time

traj, meta = load("output/white_handle_trajectory.json")
dt = meta["dt"]

bus = can.interface.Bus(channel="can2", interface="socketcan", fd=False)

# Enable motors
for mid in range(0x01, 0x08):
    bus.send(can.Message(arbitration_id=mid, data=b'\xff' * 6 + b'\xff\xfc'))
    bus.recv(timeout=0.5)

# Play CAN frames from JSON
import json
with open("output/white_handle_trajectory.json") as f:
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
  {"label": "white handle", "p1": [430, 410], "p2": [480, 410]},
  {"label": "yellow cube", "p1": [469, 486], "p2": [547, 486]}
]}
```

Single-target (also accepted):
```json
{"p1": [430, 410], "p2": [480, 410]}
```

Coordinates are 0-1000 normalized (same as `grasp_selector.py` output).

## Full Dora Dataflow

The `openarm-grasp-motion.yml` dataflow runs the complete pipeline:

```
camera → SAM3 (text-prompted segmentation)
       → grasp_selector (candidate ranking via VLM)
       → motion_planner (collision-aware trajectory)
       → trajectory_player → arm driver (CAN)
```

```bash
# On champagne
dora up
dora build openarm-grasp-motion.yml
dora start openarm-grasp-motion.yml
# ... grasp happens automatically when the target object is detected ...
dora stop --name <id>
dora destroy
```

Set `TARGET_OBJECT` in the selector's env to change what to grasp (default: `"the white handle"`).

Set `EXPORT_PATH` on the motion planner node to auto-save every planned trajectory as JSON.

## Configuration

Camera transforms and RealSense relay paths are stored in `openarm-config.json`. The standalone scripts read this file via `--config`. The dora dataflows have the values inline in their YAML env vars.

## Files

| File | Purpose |
|---|---|
| `grasp_trajectory.py` | General-purpose trajectory planner: targets JSON → collision-free trajectory + GIF |
| `white_handle_grasp.py` | HSV-based white handle detector → targets JSON (feeds into `grasp_trajectory.py`) |
| `grasp_selector.py` | Dora node: SAM3 mask + VLM rating → grasp point selection |
| `grasp_planner.py` | Dora node: grasp planning utilities |
| `grasp_trigger.py` | Dora node: trigger grasp on command |
| `openarm-grasp-motion.yml` | Full dora dataflow (camera + SAM3 + VLM + planner + arm) |
| `openarm-config.json` | Camera transforms and relay paths |
| `output/` | Generated trajectories, GIFs, and annotated images |
