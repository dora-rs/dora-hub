# Pick-and-Place + Screw Demo — Quickstart

End-to-end voice/chat → VLM → SAM3 → motion planner → CAN playback pipeline,
running on baguette. Supports `pick`, `place`, `pour`, `flip`, and `screw` actions.

**Live web client**: <https://huggingface.co/spaces/haixuantao/openarm>
The Hugging Face Space is the OpenArm web UI — 3D arm viewer, camera
preview, and the `@robot` chat channel that drives this dataflow.
It subscribes to the same `cdn.1ms.ai` MoQ relay that `can-server` and
`realsense-server` publish to, so it lights up automatically once those
two services are running on baguette.

## Prerequisites

- baguette (`172.18.128.205`) reachable over SSH with key auth
- `xoq-can` server running somewhere reachable from baguette (real arms or fake)
- Both 64-char iroh hex IDs for the left/right CAN endpoints
- A RealSense relay published to `anon/<id>/realsense-<serial>` (xoq-realsense on the camera host)
- `cdn.1ms.ai` reachable for chat + debug-image streaming

## One-time setup on baguette

```bash
ssh baguette@172.18.128.205
cd ~/dora-hub

# Install uv if missing
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a project venv pinned to Python 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate

# Install all node deps (uses pyproject.toml in each node-hub package).
# dora-rs is pinned `< 0.5` in each pyproject.toml because the dora-cli on
# baguette is 0.4.1 and v0.5+ uses an incompatible message format.
uv pip install \
  -e node-hub/dora-xoq-chat \
  -e node-hub/dora-pyrealsense \
  -e node-hub/dora-sam3 \
  -e node-hub/dora-motion-planner \
  -e ~/wser/packages/realsense \
  -e ~/wser/packages/chat \
  -e ~/XoQ/packages/can \
  numpy opencv-python-headless pyarrow

# SAM3 library — install to a persistent path (NOT /tmp).
# The `[dev,train,notebooks]` extras pull in einops, pycocotools, hydra-core,
# fvcore, etc. which dora-sam3 imports transitively even for inference.
git clone https://github.com/facebookresearch/sam3 ~/sam3
uv pip install -e '~/sam3[dev,train,notebooks]'

# Remove any stale system-Python dora-sam3 shim — its shebang points at
# /usr/bin/python3 which can't see the venv, so dora picks the wrong one.
rm -f ~/.local/bin/dora-sam3

# HuggingFace token (for SAM3 model download)
mkdir -p ~/.cache/huggingface
echo "<your-hf-token>" > ~/.cache/huggingface/token

# Build Rust binaries (dora-qwen-omni, dora-openarm-playback, dora-moq-image)
# See CLAUDE.md "Building dora-qwen-omni with CUDA" — needs conda CUDA toolkit.
# After building on champagne, the binaries can also be rsync'd into ~/dora-hub/target/release/
cargo build --release \
  -p dora-qwen-omni \
  -p dora-openarm-playback \
  -p dora-moq-image
```

### Models

| Model | Path on baguette |
|---|---|
| Qwen3.5-35B-A3B GGUF | `~/models/qwen3.5-35b-a3b/Qwen3.5-35B-A3B-UD-Q4_K_M.gguf` |
| Qwen3.5 mmproj | `~/models/qwen3.5-35b-a3b/mmproj-F16.gguf` |
| SAM3 weights | downloaded automatically to HF cache on first run |

If they're missing, copy from champagne:
```bash
rsync -az champagne@172.18.128.207:~/models/qwen3.5-35b-a3b/ ~/models/qwen3.5-35b-a3b/
```

## Configure CAN endpoints

Edit `examples/openarm-grasp/pick-and-place-chat.yml` — set both `CAN_INTERFACE`
keys to the current 64-char iroh IDs of your `xoq-can` server (left + right).
The keys at the top of the file as comments are stale references; replace the
live values under `left_arm` and `right_arm`.

For real arms, also set `GRASP_CHECK: "true"` to enable torque-based grasp
verification (currently `false` for screw demo testing).

### Fake CAN servers (no physical arms)

For end-to-end testing without real hardware, run two `fake-can-server`
instances (one per arm) in their own dirs so each gets a stable iroh key:

```bash
ssh baguette@172.18.128.205
mkdir -p ~/fake_can_left ~/fake_can_right

tmux new-session -d -s fake_can_left  \
  "cd ~/fake_can_left  && ~/XoQ/target/release/fake-can-server \
     --moq-relay https://cdn.1ms.ai --moq-path anon/xoq-can-left  --arm left"

tmux new-session -d -s fake_can_right \
  "cd ~/fake_can_right && ~/XoQ/target/release/fake-can-server \
     --moq-relay https://cdn.1ms.ai --moq-path anon/xoq-can-right --arm right"

# Grab the "Server ID:" line from each tmux pane and paste those 64-char hex
# IDs into pick-and-place-chat.yml as CAN_INTERFACE for left_arm / right_arm.
tmux capture-pane -t fake_can_left  -p -S -200 | grep 'Server ID'
tmux capture-pane -t fake_can_right -p -S -200 | grep 'Server ID'
```

The keys are persisted in `.xoq_fake_can_server_key_anon_xoq-can-{left,right}`
inside each dir, so they stay stable across restarts.

## Run the demo

Dora runs inside a tmux session named `dora` on baguette so logs persist
across SSH disconnects.

```bash
# 1. From your laptop — sync local code to baguette
rsync -az --exclude='.git' --exclude='target' --exclude='__pycache__' \
  --exclude='.xoq_fake_can_server_key' \
  ./ baguette@172.18.128.205:~/dora-hub/

# 2. Make sure the dora tmux session exists
ssh baguette@172.18.128.205 "tmux has-session -t dora 2>/dev/null || tmux new-session -d -s dora"

# 3. Stop any previous run
ssh baguette@172.18.128.205 "tmux send-keys -t dora C-c"

# 4. Start the RealSense relay (xoq camera bridge) in its own tmux session
#    Path + serial must match RELAY_PATH in pick-and-place-chat.yml.
ssh baguette@172.18.128.205 "tmux has-session -t realsense 2>/dev/null || \
  tmux new-session -d -s realsense '~/wser/target/release/realsense-server \
    --relay https://cdn.1ms.ai \
    --path anon/7e58263812ba/realsense-243222073892 \
    --width 1280 --height 720 --fps 15 --serial 243222073892'"

# 5. Start the dataflow.
#    NOTE: do NOT prepend ~/.local/bin to PATH — it shadows the venv binaries
#    (system dora-sam3 / dora-pyrealsense have shebangs pointing at /usr/bin/python3).
ssh baguette@172.18.128.205 "tmux send-keys -t dora '\
  source ~/dora-hub/.venv/bin/activate && \
  export PATH=\$HOME/.cargo/bin:\$PATH && \
  export LD_LIBRARY_PATH=~/dora-hub/.venv/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:\$LD_LIBRARY_PATH && \
  cd ~/dora-hub/examples/openarm-grasp && \
  dora run pick-and-place-chat.yml' Enter"
```

> **Why `LD_LIBRARY_PATH`?** baguette's system NVRTC (12.0) can't compile for
> RTX 5090 (sm_120). PyTorch's bundled NVRTC 12.8 must come first on the path.

### Watch logs

```bash
# Last 80 lines
ssh baguette@172.18.128.205 "tmux capture-pane -t dora -p -S -80"

# Follow live (attach to tmux)
ssh -t baguette@172.18.128.205 "tmux attach -t dora"   # Ctrl+B D to detach
```

If a node crashes, dora dumps the run UUID — the per-node logs are at:
`~/dora-hub/examples/openarm-grasp/out/<uuid>/log_<node>.txt`

## Trigger actions via chat

The `chat` node listens to MoQ chat channel `anon/openarm-chat`. Send a message
prefixed `@robot` from any client connected to `cdn.1ms.ai`:

| Command | Action |
|---|---|
| `@robot pick the yellow cube` | grasp + return home |
| `@robot pick the cube and place it in the green box` | full pick-and-place |
| `@robot flip the sausage in the pan` | grasp + 90° wrist roll |
| `@robot pour the cup into the pan` | pick + tilt-pour + return |
| `@robot screw the black cylinder` | dual-arm: hold base, ratchet-screw cap |

The VLM parses the command into JSON (`{"pick": ..., "place": ..., "action": ...}`),
then the selector finds the object via SAM3 + bbox prompts and emits a
`grasp_result` with jaw pixel coordinates. The motion planner solves IK,
checks collisions, and plays the trajectory over CAN.

## Troubleshooting

| Symptom | Cause / Fix |
|---|---|
| `ModuleNotFoundError: No module named 'sam3'` | SAM3 source dir missing. Reinstall to a persistent path (see one-time setup), don't use `/tmp`. |
| `ModuleNotFoundError: No module named 'einops'` / `'pycocotools'` / etc. | SAM3's runtime imports come from optional extras. Reinstall with `uv pip install -e '~/sam3[dev,train,notebooks]'`. |
| `version mismatch: message format v0.8.0 is not compatible with expected v0.7.0` | venv has dora-rs ≥ 0.5 but the dora-cli daemon is 0.4.1. Each `pyproject.toml` pins `dora-rs < 0.5`; run `uv pip install -e ...` again to resolve. |
| `Could not initiate node from environment variable` after fixing the version mismatch | A stale system shim (e.g. `~/.local/bin/dora-sam3`) still on PATH ahead of the venv. Delete the offending file or remove `~/.local/bin` from `PATH`. |
| `Failed to connect to CAN server` / DNS TXT errors | iroh CAN endpoint not reachable. Check `xoq-can` (or `fake-can-server`) is running and the 64-char keys in the YAML match. |
| RealSense node times out (`Timed out waiting for broadcast announcement (10s)`) | The `realsense-server` relay isn't running, or `RELAY_PATH` doesn't match `--path` of the running server. |
| Chat replays old messages on startup | Known issue — selector ignores commands that arrive before the first camera frame. Wait for "ready" status. |
| `/dev/shm` full / `OSError: No space left on device` | Force-killed dora left semaphores behind. `rm -f /dev/shm/shmem_* /dev/shm/psm_*` |
| `dora stop` hangs | Kill processes directly, then `pkill -9 dora-coordinator dora-daemon`, then clean `/dev/shm`. |
| Motion planner: "all headings unreachable" | Object is outside reachable workspace; reposition the workspace or check camera transform `CAMERA_TRANSFORM` env. |
| VLM stuck loading | First run downloads/loads ~19GB to GPU (~30-60s). Watch with `nvidia-smi`. |

## Files in this directory

| File | Purpose |
|---|---|
| `pick-and-place-chat.yml` | **Main dataflow** for the demo (chat-driven, dual-arm, screw-capable) |
| `pick-and-place.yml` | Older single-target dataflow (no chat) |
| `grasp_selector.py` | Dora node: SAM3 + VLM → grasp pixel selection (handles screw target geometry) |
| `grasp_trajectory.py` | Standalone planner (no dora) for offline trajectory testing |
| `openarm-config.json` | Camera transforms + RealSense relay paths for standalone scripts |
| `output/` | GIFs, JSON trajectories, annotated PNGs from `grasp_trajectory.py` |
