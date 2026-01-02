# Quick example on using a VLM with dora-rs

This example demonstrates how to use vision language models (VLM) with dora-rs using the Qwen2.5-VL model.

## System Requirements

| Platform | Support | Notes |
|----------|---------|-------|
| Linux (NVIDIA GPU) | Full | Best performance with CUDA |
| macOS (Apple Silicon) | Supported | Uses MPS backend, see notes below |
| macOS (Intel) | Limited | CPU-only, very slow |
| Windows (NVIDIA GPU) | Full | Requires CUDA |

### Apple Silicon (M1/M2/M3/M4) Requirements

- **Memory**: 16GB RAM minimum (32GB recommended for smooth operation)
- **macOS**: 12.3 or later (for MPS support)
- **Model**: Uses Qwen2.5-VL-3B-Instruct (~6GB in memory)

**Performance Notes**:
- First run downloads the model (~6GB)
- Inference is slower than NVIDIA GPUs but usable for real-time applications
- Flash attention is not available on macOS (automatically falls back to standard attention)

## Prerequisites

Make sure to have the following installed:
- [dora-rs](https://dora-rs.ai/)
- [uv](https://github.com/astral-sh/uv)
- [cargo](https://rustup.rs/) (Rust toolchain)

## Quick Start

### From cloned repository

```bash
cd examples/vlm
uv venv -p 3.11 --seed
uv pip install -e ../../apis/python/node --reinstall
dora build qwen2-5-vl-vision-only-dev.yml --uv
dora run qwen2-5-vl-vision-only-dev.yml --uv
```

### Without cloning the repository

```bash
uv venv -p 3.11 --seed
dora build https://raw.githubusercontent.com/dora-rs/node-hub/main/examples/vlm/qwenvl.yml --uv
dora run https://raw.githubusercontent.com/dora-rs/node-hub/main/examples/vlm/qwenvl.yml --uv
```

## Apple Silicon Specific Setup

For MacBooks with Apple Silicon, PyTorch automatically uses the MPS (Metal Performance Shaders) backend. No additional configuration is needed.

To verify MPS is available:

```bash
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### Troubleshooting on macOS

1. **Out of memory errors**: Close other applications to free up RAM, or use a smaller model by setting:
   ```bash
   export MODEL_NAME_OR_PATH="Qwen/Qwen2.5-VL-2B-Instruct"
   ```

2. **Camera permission**: Grant terminal/IDE camera access in System Settings > Privacy & Security > Camera

3. **Slow first inference**: The first inference is slower due to MPS shader compilation. Subsequent inferences are faster.

## Available Configurations

| Config File | Description |
|-------------|-------------|
| `qwen2-5-vl-vision-only-dev.yml` | Vision-only pipeline (camera + VLM + visualization) |
| `qwen2-5-vl-speech-to-speech-dev.yml` | Full speech-to-speech pipeline with VLM |
| `qwenvl.yml` | Production configuration |
| `qwenvl-dev.yml` | Development configuration |
