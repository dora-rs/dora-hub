# dora-ios-lidar

Streams RGB and LiDAR depth frames from an iOS device running the
[Record3D](https://record3d.app/) app over USB, emitting the color frame on
`image` and the depth frame on `depth` together with camera intrinsics.

## Behavior

On startup the node searches for connected Record3D-capable iOS devices and
connects to device index 0, raising an error if none is found. It then runs as a
dora node and, on each input event, waits for the next frame from the device and:

- Copies the newly arrived depth and RGB frames and the camera intrinsic matrix.
- If the depth and RGB frames differ in size, resizes RGB to match depth.
- If both `IMAGE_WIDTH` and `IMAGE_HEIGHT` are set, optionally rotates the frames
  (when `ROTATE` = `ROTATE_90_CLOCKWISE`), resizes RGB and depth to the requested
  resolution, and rescales the focal length and principal point accordingly.
  Otherwise the native intrinsics are used.

> Note: intrinsics rescaling is only correct when `ROTATE` is unset. With
> `ROTATE_90_CLOCKWISE` the focal length and principal point are scaled but not
> transformed for the rotation (a 90° rotation should swap the x/y axes and remap
> the principal point), so the published `focal`/`resolution` metadata is
> approximate in that mode.
- Sends the RGB frame on `image` (encoding `rgb8`).
- Converts depth to millimeters (`* 1000`), clips to `[0, 4095]` (uint12 range)
  as `uint16`, and sends it on `depth` (encoding `mono16`) with the focal length
  and principal point in the metadata.

You must install the Record3D app on the iOS device and use a USB connection to
stream.

## Inputs

- `tick`: trigger to capture and emit one RGBD frame.

## Outputs

- `image`: color frame as a flattened Arrow array (`rgb.ravel()`). Metadata:
  `encoding` = `rgb8`, `width`, `height`.
- `depth`: depth frame in millimeters as a flattened `uint16` Arrow array,
  clipped to `[0, 4095]`. Metadata: `encoding` = `mono16`, `width`, `height`,
  `focal` = `[fx, fy]`, `resolution` = `[cx, cy]` (principal point).

Decoding the `image` output:

```Python
storage = event["value"]
metadata = event["metadata"]
width = metadata["width"]
height = metadata["height"]

frame = (
    storage.to_numpy()
    .astype(np.uint8)
    .reshape((height, width, 3))
)
```

## Environment variables

- `IMAGE_WIDTH`: target output width in pixels. When set together with
  `IMAGE_HEIGHT`, frames are resized and intrinsics rescaled. Unset keeps the
  native resolution.
- `IMAGE_HEIGHT`: target output height in pixels. When set together with
  `IMAGE_WIDTH`, frames are resized and intrinsics rescaled. Unset keeps the
  native resolution.
- `ROTATE`: set to `ROTATE_90_CLOCKWISE` to rotate frames 90 degrees clockwise
  before resizing. Only applied when both `IMAGE_WIDTH` and `IMAGE_HEIGHT` are
  set.

## Usage

```yaml
nodes:
  - id: dora-ios-lidar
    hub: dora-ios-lidar@^0.5
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
