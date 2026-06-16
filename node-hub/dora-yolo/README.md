# dora-yolo

Object detection with Ultralytics YOLO — image in, bounding boxes out.

## Behavior

`dora-yolo` connects as a dora node and runs an Ultralytics YOLO model on each
incoming `image` input. It reads the frame from a UInt8 Arrow array, reshapes it
using the `width`/`height`/`encoding` metadata (converting `bgr8` to `rgb8` as
needed), runs detection (including NMS), and sends the detected boxes,
confidence scores, and class names on the `bbox` output.

The model defaults to `yolov8n.pt` and can be overridden via the `MODEL`
environment variable. The bounding-box coordinate format defaults to `xyxy` and
can be switched to `xywh` via `FORMAT`.

## Inputs

- `image`: a UInt8 Arrow array holding the raw image, with metadata fields
  `width`, `height`, and `encoding` (`bgr8` or `rgb8`). Any other encoding raises
  an error.

```python
## Image data
image_data: UInt8Array  # Example: pa.array(img.ravel())
metadata = {
  "width": 640,
  "height": 480,
  "encoding": str,  # bgr8, rgb8
}
```

## Outputs

- `bbox`: an Arrow struct array containing the detected objects. Output metadata
  carries `format` (the bbox format) and `primitive` set to `boxes2d`.

```python
bbox: {
    "bbox": np.array,    # flattened array of bounding boxes
    "conf": np.array,    # flat array of confidence scores
    "labels": np.array,  # flat array of class names
}

encoded_bbox = pa.array([bbox], {"format": "xyxy"})

decoded_bbox = {
    "bbox": encoded_bbox[0]["bbox"].values.to_numpy().reshape(-1, 4),
    "conf": encoded_bbox[0]["conf"].values.to_numpy(),
    "labels": encoded_bbox[0]["labels"].values.to_numpy(zero_copy_only=False),
}
```

## Environment variables

- `MODEL` (string, default `yolov8n.pt`): Ultralytics YOLO model file to load.
- `FORMAT` (string, default `xyxy`): bounding-box coordinate format, `xyxy` or
  `xywh`.

## Usage

```yaml
nodes:
  - id: object-detection
    hub: dora-yolo@^0.5
    inputs:
      image: camera/image
    outputs:
      - bbox
```

## Build

```bash
pip install .
```
