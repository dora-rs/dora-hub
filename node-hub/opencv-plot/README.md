# opencv-plot

Overlays bounding boxes, labels, and text on an image input and displays it in
an OpenCV window. Ideal for visualizing object-detection output.

## Behavior

`opencv-plot` connects as a dora node and maintains a current frame, bbox set,
and text string. Each input updates its slice of that state:

- An **`image`** event decodes the frame and immediately redraws: it draws every
  stored bbox (green rectangle + `label, conf` text), draws the stored text at
  the top-left, optionally resizes to `PLOT_WIDTH`x`PLOT_HEIGHT`, and shows it in
  a window titled `Dora Node: opencv-plot`. Press `q` in the window to quit.
- A **`bbox`** or **`text`** event only updates stored state; it is drawn on the
  next `image` redraw.

Display is suppressed when the environment variable `CI` is `true` (the node
still decodes inputs but opens no window).

It produces no outputs.

## Inputs

- `image`: base image to draw on. UInt8 Arrow array plus metadata
  `{width, height, encoding}`. Supported encodings: `bgr8`, `rgb8`,
  `jpeg`/`jpg`/`jpe`/`bmp`/`webp`/`png` (encoded byte stream, decoded with
  `cv2.imdecode`), `yuv420`, `avif`. An unsupported encoding raises an error.

  ```python
  image_data: UInt8Array            # e.g. pa.array(img.ravel())
  metadata = {"width": 640, "height": 480, "encoding": "bgr8"}
  node.send_output("image", image_data, metadata)
  ```

- `bbox`: bounding boxes, confidence scores, and labels. The value is an Arrow
  struct array whose element `[0]` has fields `bbox` (flattened, reshaped to
  `-1, 4`), `conf`, and `labels`, with metadata `{format}`. Supported formats:
  `xyxy` and `xywh` (converted to `xyxy`). An unsupported format raises an error.

  ```python
  bbox = {
      "bbox": np.array(...),    # flattened bounding boxes
      "conf": np.array(...),    # confidence scores
      "labels": np.array(...),  # class names
  }
  node.send_output("bbox", pa.array([bbox]), {"format": "xyxy"})
  ```

- `text`: text to overlay at the top-left. Arrow array of size 1; element `[0]`
  is read with `.as_py()`.

  ```python
  node.send_output("text", pa.array(["hello"]))
  ```

## Outputs

None — it is a display sink.

## Environment variables

- `PLOT_WIDTH` (optional): resize the displayed frame to this width in pixels.
  Defaults to the image input width (no resize).
- `PLOT_HEIGHT` (optional): resize the displayed frame to this height in pixels.
  Defaults to the image input height (no resize).

(`CI=true` suppresses the display window; it is not a configuration knob.)

## Usage

```yaml
nodes:
  - id: opencv-plot
    hub: opencv-plot@^0.5
    inputs:
      image: camera/image
      bbox: detector/bbox
      text: detector/text
    env:
      PLOT_WIDTH: "640"
      PLOT_HEIGHT: "480"
```

## Build

```bash
pip install .
```
