# Dora Node for capturing video with OpenCV

This node is used to capture video from a camera using OpenCV.

# YAML

```yaml
- id: opencv-video-capture
  build: pip install ../../opencv-video-capture
  path: opencv-video-capture
  inputs:
    tick: dora/timer/millis/16 # how often to read a frame (~60Hz)
  outputs:
    - image: # the captured image

  env:
    PATH: 0 # optional, default is 0

    IMAGE_WIDTH: 640 # optional, default is video capture width
    IMAGE_HEIGHT: 480 # optional, default is video capture height

    # optional, default is the camera default (often 30 fps).
    # The tick input only controls how often frames are READ; the actual
    # capture rate is negotiated with the camera. To capture at 60 fps you
    # must both set CAPTURE_FPS: 60 and use a tick of at most ~16ms.
    CAPTURE_FPS: 60

    # optional, default is the camera default.
    # On Linux UVC cameras, MJPG is often required to reach high frame
    # rates (e.g. 60 fps at 720p or above) due to USB bandwidth limits.
    CAPTURE_FOURCC: MJPG

    # optional, default is 95.
    # This is used only when encoding is one of "jpeg", "jpg" or "jpe".
    JPEG_QUALITY: 95
```

## Frame rate notes

- A warning is printed at startup if the camera negotiates a frame rate
  different from `CAPTURE_FPS`.
- In low light, cameras with auto-exposure may halve the frame rate.

# Inputs

- `tick`: empty Arrow array to trigger the capture

# Outputs

- `image`: an arrow array containing the captured image

```Python
## Image data
image_data: UInt8Array # Example: pa.array(img.ravel())
metadata = {
  "width": 640,
  "height": 480,
  "encoding": str, # bgr8, rgb8
}

## Example
node.send_output(
  image_data, {"width": 640, "height": 480, "encoding": "bgr8"}
  )

## Decoding
storage = event["value"]

metadata = event["metadata"]
encoding = metadata["encoding"]
width = metadata["width"]
height = metadata["height"]

if encoding == "bgr8":
    channels = 3
    storage_type = np.uint8

frame = (
    storage.to_numpy()
    .astype(storage_type)
    .reshape((height, width, channels))
)
```

## Examples

Check example at [examples/python-dataflow](examples/python-dataflow)

## License

This project is licensed under Apache-2.0. Check out [NOTICE.md](../../NOTICE.md) for more information.
