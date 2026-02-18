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

**Use `dora up` / `dora start` / `dora stop` instead of `dora run`** for proper lifecycle management:
- `dora run` + Ctrl+C can leave orphaned GPU processes
- `dora stop` sends proper Stop events and nodes exit gracefully

The coordinator must be started with the correct environment so spawned nodes inherit it:

```bash
export PATH=$HOME/.cargo/bin:$HOME/miniconda3/bin:$PATH
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate base
export LD_LIBRARY_PATH=$HOME/miniconda3/lib:$LD_LIBRARY_PATH
dora up
dora build <dataflow.yml>
dora start <dataflow.yml>
# ... later ...
dora stop <dataflow-id>
dora destroy
```

### Qwen3-VL-A3B Model

- Model path: `/home/champagne/models/qwen3-vl-a3b/Qwen3VL-30B-A3B-Instruct-Q4_K_M.gguf`
- mmproj path: `/home/champagne/models/qwen3-vl-a3b/mmproj-Qwen3VL-30B-A3B-Instruct-F16.gguf`
- 30B total params, 3B active (MoE), Q4_K_M quantization (~18GB VRAM)
- Performance on RTX 5090: ~1.2s per frame (bottleneck is vision encoder prefill: 300 batches × 3ms)
- Constraining output (bbox vs sentences) doesn't meaningfully affect speed

## dora-qwen-omni Input Format

The node expects a string array with special markers:
- `<|user|>\n<|vision_start|>\n` + base64 data URI (e.g., `data:image/png;base64,...`)
- `<|user|>\n<|im_start|>\n` + text prompt
