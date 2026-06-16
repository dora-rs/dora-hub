# dora-pyrealsense

Captures aligned RGB color and depth frames from an Intel RealSense camera.

## Behavior

On startup the node queries for connected RealSense devices and raises a
`ConnectionError` if none are found. It opens the device selected by
`DEVICE_SERIAL` (or the default device) and enables a color stream
(`rgb8`) and a depth stream (`z16`) at `IMAGE_WIDTH` x `IMAGE_HEIGHT`, 30 FPS.
Depth frames are aligned to the color stream via `rs.align`.

On each `tick` input it waits for a frame set, aligns it, and:

- Optionally flips the color frame (`FLIP` = `VERTICAL` / `HORIZONTAL` / `BOTH`).
- Encodes the color frame per `ENCODING` (`rgb8` passthrough, `bgr8` color
  conversion, or OpenCV image encoding for `jpeg`/`jpg`/`jpe`/`bmp`/`webp`/`png`).
- Sends the color frame on `image` with metadata (`encoding`, `width`,
  `height`, `resolution` = RGB principal point, `focal_length`, `timestamp`).
- Zeros depth values above 5000, then sends the depth frame on `depth` as
  `mono16`, reusing the color frame metadata.

When the `CI` environment variable is `true`, the node runs for only 10 seconds.

## Inputs

- `tick`: trigger to capture one aligned color + depth frame.

## Outputs

- `image`: color frame as a flattened Arrow array (`pa.array(frame.ravel())`).
  Metadata: `encoding`, `width`, `height`, `resolution`, `focal_length`,
  `timestamp`.
- `depth`: aligned depth frame (`mono16`, values > 5000 zeroed) as a flattened
  Arrow array, sharing the color frame metadata.

Decoding the `image` output:

```Python
storage = event["value"]
metadata = event["metadata"]
encoding = metadata["encoding"]
width = metadata["width"]
height = metadata["height"]

if encoding in ["rgb8", "bgr8"]:
    channels = 3
    frame = (
        storage.to_numpy()
        .astype(np.uint8)
        .reshape((height, width, channels))
    )
```

## Environment variables

- `FLIP`: flip the color frame; one of `VERTICAL`, `HORIZONTAL`, `BOTH` (empty disables). Default empty.
- `DEVICE_SERIAL`: serial number of the RealSense device to open (empty selects the default device). Default empty.
- `IMAGE_HEIGHT`: requested stream height in pixels. Default `480`.
- `IMAGE_WIDTH`: requested stream width in pixels. Default `640`.
- `ENCODING`: color output encoding; `rgb8`, `bgr8`, or an OpenCV-encoded format (`jpeg`, `jpg`, `jpe`, `bmp`, `webp`, `png`). Default `rgb8`.

Make sure to install the RealSense udev rules:
<https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md>

## Usage

```yaml
nodes:
  - id: dora-pyrealsense
    hub: dora-pyrealsense@^0.5
    inputs:
      tick: dora/timer/millis/33
    outputs:
      - image
      - depth
```

## Build

```bash
pip install .
```
