# Dora Node for Generic ONNX Inference

This node loads an exported `.onnx` model, accepts image tensors from the dataflow, performs inference using ONNX Runtime, and outputs the raw result tensors. It automatically tries GPU acceleration (CUDA) and falls back to CPU.

## YAML

```yaml
- id: onnx-inference
  build: pip install ../../node-hub/dora-onnx
  path: dora-onnx
  inputs:
    image: webcam/image
  outputs:
    - tensor
  env:
    MODEL: model.onnx
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--name` | `str` | `onnx-inference` | The name of the node in the dataflow. |
| `--model` | `str` | `model.onnx` | Path to the `.onnx` model file. |
| `MODEL` (env) | `str` | — | Environment variable override for the model path (takes priority over `--model`). |

## Inputs

- **`image`**: Arrow `UInt8Array` containing the raw image pixels.

The following metadata fields are required:

| Key | Type | Description |
|-----|------|-------------|
| `width` | `int` | Image width in pixels. |
| `height` | `int` | Image height in pixels. |
| `encoding` | `str` | Pixel format: `rgb8` or `bgr8`. |

```python
import pyarrow as pa

node.send_output(
    "image",
    pa.array(img.ravel()),
    {"width": 640, "height": 480, "encoding": "rgb8"},
)
```

### Preprocessing

The node automatically applies the following preprocessing steps:

1. Reshape to `(H, W, 3)` and convert BGR→RGB if needed.
2. Transpose to NCHW layout `(1, 3, H, W)`.
3. Cast to `float32` and normalize to `[0, 1]`.

## Outputs

- **`tensor`**: Arrow array containing the flattened raw output tensor from the model.

Output metadata includes the original input metadata plus:

| Key | Type | Description |
|-----|------|-------------|
| `shape` | `list[int]` | Original shape of the output tensor (e.g. `[1, 1000]`). |
| `type` | `str` | Always `"onnx_tensor"`. |

```python
# Decoding the output tensor
storage = event["value"]
metadata = event["metadata"]
shape = metadata["shape"]

output_tensor = storage.to_numpy().reshape(shape)
```

## License

This project is licensed under Apache-2.0. Check out [NOTICE.md](../../NOTICE.md) for more information.
