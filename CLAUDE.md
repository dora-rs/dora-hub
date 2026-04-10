# Node Hub

## Remote Machine: champagne@172.18.128.207

- **Hardware**: RTX 5090 (32GB VRAM), Intel Core Ultra 9 285K
- **OS**: Ubuntu Linux, CUDA 12.8 driver
- **Toolchain**: Rust 1.93.1, miniconda3 (conda base env)

### Building dora-qwen-omni with CUDA

Full build command (all env vars needed for conda-based CUDA toolkit):

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate base
export CONDA_PREFIX=$HOME/miniconda3

RUSTFLAGS='-L native=$CONDA_PREFIX/lib -L native=$CONDA_PREFIX/targets/x86_64-linux/lib' \
LIBCLANG_PATH=$CONDA_PREFIX/lib \
BINDGEN_EXTRA_CLANG_ARGS='-I$CONDA_PREFIX/lib/clang/21/include' \
OPENSSL_DIR=$CONDA_PREFIX \
CUDA_PATH=$CONDA_PREFIX \
cargo build --release -p dora-qwen-omni --features cuda
```

Required conda packages: `cmake`, `cuda-toolkit` (nvidia channel), `cuda-libraries-static` (nvidia channel), `clangdev` (conda-forge)

### Running dora dataflows

Use `dora run` to build and run a dataflow in one step:

```bash
export PATH=$HOME/.cargo/bin:$HOME/miniconda3/bin:$PATH
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate base
export LD_LIBRARY_PATH=$HOME/miniconda3/lib:$LD_LIBRARY_PATH
dora run <dataflow.yml>
```

### Stop / sync / restart workflow (baguette)

Dora runs inside a **tmux session** named `dora` on baguette. All dora commands must be sent through tmux.

To redeploy code changes on baguette:

```bash
# 1. Stop the running dataflow (Ctrl+C via tmux)
ssh baguette@172.18.128.205 "tmux send-keys -t dora C-c"

# 2. Sync code
rsync -az --exclude='.git' --exclude='target' --exclude='__pycache__' --exclude='.xoq_fake_can_server_key' ./ baguette@172.18.128.205:~/dora-hub/

# 3. Run (via tmux)
ssh baguette@172.18.128.205 "tmux send-keys -t dora 'cd ~/dora-hub/examples/openarm-grasp && dora run pick-and-place-chat.yml' Enter"
```

To check logs or output, capture the tmux pane:
```bash
ssh baguette@172.18.128.205 "tmux capture-pane -t dora -p -S -50"  # last 50 lines
```

### Stop / sync / restart workflow (baguette)

The dora coordinator runs inside a **tmux session** named `dora` on baguette. All dora commands must be sent through tmux since the coordinator socket is only accessible from that session's environment.

To redeploy code changes to a running dataflow on baguette:

```bash
# 1. Find the running dataflow name (via tmux)
ssh baguette@172.18.128.205 "tmux send-keys -t dora 'dora list' Enter" && sleep 2 && \
ssh baguette@172.18.128.205 "tmux capture-pane -t dora -p | grep -E 'Running|Stopped'"

# 2. Stop it (via tmux, use the name from step 1)
ssh baguette@172.18.128.205 "tmux send-keys -t dora 'dora stop --name <name>' Enter"
# Wait for stop to complete (check tmux output)
sleep 5 && ssh baguette@172.18.128.205 "tmux capture-pane -t dora -p | tail -5"

# 3. Sync code
rsync -az --exclude='.git' --exclude='target' --exclude='__pycache__' --exclude='.xoq_fake_can_server_key' ./ baguette@172.18.128.205:~/dora-hub/

# 4. Start (via tmux)
ssh baguette@172.18.128.205 "tmux send-keys -t dora 'cd ~/dora-hub/examples/openarm-grasp && dora start pick-and-place-chat.yml' Enter"
```

To check logs or output, capture the tmux pane:
```bash
ssh baguette@172.18.128.205 "tmux capture-pane -t dora -p -S -50"  # last 50 lines
```

If `dora stop` times out, send `dora destroy` via tmux, then `dora up` to restart the coordinator.

### Qwen3.5-35B-A3B Model

- Model path (champagne): `/home/champagne/models/qwen3.5-35b-a3b/Qwen3.5-35B-A3B-UD-Q4_K_M.gguf`
- Model path (baguette): `/home/baguette/models/qwen3.5-35b-a3b/Qwen3.5-35B-A3B-UD-Q4_K_M.gguf`
- mmproj path: `mmproj-F16.gguf` (same dir)
- 35B total params, 3B active (MoE, 256 experts), Q4_K_M quantization (~19GB VRAM)
- Uses llama-cpp-rs v0.1.137 (llama.cpp with Qwen3.5 support via PR #19468)
- Architecture: `qwen35moe`, projector: `qwen3vl_merger`

### Generating grasp/pick-and-place trajectories

Run `grasp_trajectory.py` standalone on champagne (no dora/CAN needed — reads depth from xoq-realsense relay):

```bash
ssh champagne@172.18.128.207
source ~/miniconda3/etc/profile.d/conda.sh && conda activate base
cd ~/dora-hub/examples/openarm-grasp

# Pick-and-place: pick yellow cube, place in green box
python grasp_trajectory.py \
    --config openarm-config.json \
    --targets '{"p1": [316, 382], "p2": [359, 382], "place": [609, 528]}' \
    --camera champagne-realsense --device cuda

# Grasp-only (2-phase: home → pre-grasp → grasp)
python grasp_trajectory.py \
    --config openarm-config.json \
    --targets '{"p1": [316, 382], "p2": [359, 382]}' \
    --camera champagne-realsense --device cuda
```

- Coordinates are 0-1000 normalized pixel coords (same as `grasp_selector.py` output)
- `--camera` selects the RealSense by label from `openarm-config.json` (reads via xoq relay)
- Outputs: `output/<label>_trajectory.json`, `output/<label>_trajectory.gif`, `output/<label>_final.png`
- Tries both arms, picks the one with lowest collision count / cost
- Copy results locally: `scp champagne@172.18.128.207:~/dora-hub/examples/openarm-grasp/output/* examples/openarm-grasp/output/`

## dora-qwen-omni Input Format

The node expects a string array with special markers:
- `<|user|>\n<|vision_start|>\n` + base64 data URI (e.g., `data:image/png;base64,...`)
- `<|user|>\n<|im_start|>\n` + text prompt
