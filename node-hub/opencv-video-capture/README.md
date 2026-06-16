# opencv-video-capture

Capture video frames from a camera or video device using OpenCV.

## Behavior

On each `tick` input the node reads one frame from an OpenCV `VideoCapture`
device, optionally flips and resizes it, encodes it to the configured format,
and sends it on the `image` output as a flat UInt8 Arrow array. The frame's
`width`, `height`, `encoding`, and `primitive` (`"image"`) are attached as
metadata.

The capture device is selected either by `CAMERA_ID` (a stable per-platform
unique identifier, resolved via macOS `system_profiler`, Linux
`/dev/v4l/by-id/`, or Windows `Get-PnpDevice`) or, if that is unset, by
`CAPTURE_PATH` (a device path or integer index). `CAMERA_ID` takes precedence.

If a frame cannot be read the node raises an error (so a `restart_policy:
on-failure` can recover the device), unless `CI=true`, in which case it emits a
black placeholder frame and exits after 10 seconds. The node stops when it
receives `STOP` or when the `tick` input closes.

## Inputs

- `tick`: trigger to read and emit the next frame (typically a dora timer).

## Outputs

- `image`: the captured frame as a flat UInt8 Arrow array
  (`pa.array(frame.ravel())`). Metadata: `width`, `height`, `encoding`,
  `primitive: "image"`.

```python
storage = event["value"]
metadata = event["metadata"]
encoding = metadata["encoding"]   # bgr8, rgb8, ...
width = metadata["width"]
height = metadata["height"]

if encoding == "bgr8":
    frame = storage.to_numpy().astype(np.uint8).reshape((height, width, 3))
```

## Environment variables

- `CAMERA_ID` (string, default unset): unique camera identifier for stable
  selection across platforms. Takes precedence over `CAPTURE_PATH` when set.
- `CAPTURE_PATH` (string/int, default `0`): device path or integer index of the
  camera (e.g. `/dev/video1` or `0`). Numeric strings are treated as an index.
  Used only when `CAMERA_ID` is unset.
- `IMAGE_WIDTH` (int, default = camera native width): requested capture width;
  also resizes output frames.
- `IMAGE_HEIGHT` (int, default = camera native height): requested capture
  height; also resizes output frames.
- `ENCODING` (string, default `bgr8`): output encoding — one of `bgr8`, `rgb8`,
  `yuv420`, `jpeg`/`jpg`/`jpe`, `bmp`, `webp`, `png`.
- `JPEG_QUALITY` (int, default `95`): JPEG quality (0-100), used only when
  `ENCODING` is `jpeg`, `jpg`, or `jpe`.
- `FLIP` (string, default unset): flip the frame — `VERTICAL`, `HORIZONTAL`, or
  `BOTH`. Empty/unset means no flip.
- `CI` (bool, default unset): when `true`, runs for only 10 seconds and emits a
  black error frame instead of raising when a frame cannot be read.

## Usage

```yaml
nodes:
  - id: camera
    hub: opencv-video-capture@^0.5
    inputs:
      tick: dora/timer/millis/20
    outputs:
      - image
    env:
      CAPTURE_PATH: 0
      IMAGE_WIDTH: 640
      IMAGE_HEIGHT: 480
```

## Build

```bash
pip install .
```
