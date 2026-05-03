# GOSIM 2026 Hackathon — OpenArm Screw Demo

End-to-end voice/chat → VLM → SAM3 → motion planner → CAN playback,
with a new dual-arm **screw** action (one arm holds the base, the other
ratchet-rotates the cap). Hardware: two OpenArm v10 arms wired to a
PEAK PCAN-USB Pro FD adapter on baguette, RealSense D435i streaming
over a MoQ relay.

**Live demo & 3D viewer**: <https://huggingface.co/spaces/haixuantao/openarm>
The Hugging Face Space hosts the OpenArm web client — it subscribes
to the same MoQ relay (`cdn.1ms.ai`) that this dataflow publishes to,
so once `can-server` and `realsense-server` are up on baguette the
page renders the live arm state and camera feed. Chat commands sent
through the page's `@robot` channel are what trigger the VLM →
selector → motion-planner pipeline below.

## What's new on this branch

### 1. Screw action — dual-arm coordinated motion (~600 LOC across 3 files)

Phase 1 (`plan_screw_hold`) — hold arm grasps the base (horizontal 90°
approach to avoid jaw collision with the cap), no retract.
Phase 2 (`plan_screw_rotation`) — screw arm grasps the cap with
70° approach for wrist-rotation freedom, then ratchets through
`screw_degrees` total rotation in IK-feasible cycles. Each cycle:
forward rotation → release gripper → unrotate → re-grip.
Phase 3 (`plan_screw_release`) — hold arm releases the base and
returns home via a safe-Z staging path.

Files:
- `node-hub/dora-motion-planner/dora_motion_planner/main.py`:
  added `_solve_grasp_ik` (heading sweep extracted from
  `plan_grasp_from_pixels`), `plan_screw_hold`, `plan_screw_rotation`,
  `plan_screw_release`. The `_build_pick_place_trajectory` helper
  gained a `stop_at_grasp` flag so the screw planners can reuse the
  approach/staging logic without the retract+home tail.
- `node-hub/dora-xoq-chat/dora_xoq_chat/main.py`:
  VLM parse prompt and chat acknowledgement now recognise
  `{"action": "screw"}`.
- `examples/openarm-grasp/grasp_selector.py`:
  derives base/cap jaw pixel pairs from the SAM3 mask bbox
  (top 15% = base, bottom 85% = cap, same X centre), emits
  `screw_p1`, `screw_p2`, `screw_degrees` in the `grasp_result`.

Trigger via chat: `@robot screw the black cylinder`.

### 2. New file: `node-hub/dora-motion-planner/dora_motion_planner/analyze_screw.py`

Standalone trajectory diagnostics: decodes Damiao MIT frames out of a
screw `*_trajectory.json`, runs FK on each waypoint, and reports
joint-space jumps, EE Z drift between hold and rotate, heading
selection, and the safe-return path.

### 3. Pinned `dora-rs >= 0.3.9, < 0.5` in every node's `pyproject.toml`

The dora-cli daemon on baguette is 0.4.1 (message format v0.7.0).
v0.5.0 introduced an incompatible v0.8.0 wire format that fails with
`version mismatch: message format v0.8.0 is not compatible with
expected v0.7.0` at node startup. The upper bound stops new installs
from silently breaking the daemon.

Affected: `dora-xoq-chat`, `dora-pyrealsense`, `dora-sam3`,
`dora-motion-planner`.

### 4. New file: `examples/openarm-grasp/QUICKSTART.md`

End-to-end setup guide for running the demo on baguette: uv-based venv
bootstrap, SAM3 install with `[dev,train,notebooks]` extras (so
einops / pycocotools / hydra-core etc. resolve), CAN endpoint config,
fake-can fallback for hardware-less testing, `dora run` launch via
the persistent `dora` tmux session, and a troubleshooting table that
covers every failure we hit during integration.

### 5. `pick-and-place-chat.yml` runtime tweaks

- CAN keys updated to the iroh server IDs published by the live
  `can-server` running on baguette against `can0:fd` (left,
  `d3cd5a3b…7f9b24`) and `can1:fd` (right, `d1c0840b…f79824`).
- `GRASP_CHECK` flipped to `"false"` for screw testing — the screw
  hold pose intentionally has no torque-based grasp confirmation.
  Flip to `"true"` for picks/places on real arms when you want
  torque-based "did we actually grab it" verification.

## Reproducing the demo

See `examples/openarm-grasp/QUICKSTART.md` for the full setup. TL;DR:

1. On baguette, bring up `can0` + `can1` in CAN-FD mode (`bash setup-can.sh`).
2. Start `xoq-can` server: `~/wser/target/release/can-server can0:fd can1:fd --key-dir ~/.config/xoq/keys --moq-relay https://cdn.1ms.ai`.
3. Start the RealSense relay:
   `~/wser/target/release/realsense-server --relay https://cdn.1ms.ai --path anon/7e58263812ba/realsense-243222073892 --width 1280 --height 720 --fps 15 --serial 243222073892`.
4. `dora run examples/openarm-grasp/pick-and-place-chat.yml` from the venv (in the `dora` tmux session).
5. From a chat client subscribed to `anon/openarm-chat` on `cdn.1ms.ai`,
   send `@robot screw the black cylinder`, then `ok` to confirm.

## Hardware lessons (for future hackathons)

- **Wire-format drift**: the playback (`dora-openarm-playback`) and
  `can-server` must be linked against the *same* version of the `xoq`
  crate. We had a stale Feb-19 `can-server` from `~/XoQ` running while
  the freshly-rebuilt playback was linked against `~/wser` xoq 0.3.6,
  which produced 70k "Invalid standard CAN ID" rejections per minute
  and zero motor responses. Fix: rebuild both from the same source
  tree (we use `~/wser` end-to-end now).
- **`/tmp/sam3` evaporates** after reboots — install SAM3 to a
  persistent path (e.g. `~/sam3`) or the next dora launch will
  `ModuleNotFoundError`.
- **`~/.local/bin/dora-sam3` shadow**: the system-Python console
  script has a shebang that points at `/usr/bin/python3`, so even
  inside a venv it imports against the wrong interpreter. Either
  delete the shim (`rm ~/.local/bin/dora-sam3`) or drop
  `~/.local/bin` from `PATH` before launching dora.
- **fake-can-server** without `--gravity` only publishes MoQ state
  on motor-position changes. Idle = no publish = frozen viewer. With
  `--gravity` it publishes at 100Hz from the 1kHz physics tick. Use
  gravity for visual demos, snap mode for deterministic playback.

