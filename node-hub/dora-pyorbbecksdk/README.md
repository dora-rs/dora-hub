# dora-pyorbbecksdk

Captures RGB color and depth frames from an Orbbec depth camera via the
[pyorbbecsdk](https://github.com/orbbec/pyorbbecsdk).

## Behavior

On startup the node enumerates connected Orbbec devices and opens the one at
`DEVICE_INDEX`. It tries to enable a color stream at 640x480 `RGB` 30 FPS and a
depth stream at 640x480 `Y16` 30 FPS, falling back to each sensor's default
profile if the requested profile is unavailable.

On each `tick` input it waits up to 100 ms for a frame set and, if both a color
and a depth frame are present:

- Converts the color frame to a BGR image (handling RGB, BGR, YUYV, MJPG, I420,
  NV12, NV21, UYVY source formats), JPEG-encodes it, and sends it on `image`
  with metadata `encoding=jpeg`.
- Converts the depth frame to meters (`raw * depth_scale * 0.001`), zeros values
  outside 0.01-15.0 m, applies a temporal filter (alpha 0.5), and sends the
  flattened float32 array on `depth`.
- Colorizes the depth map (`COLORMAP_JET`), JPEG-encodes it, and sends it on
  `image_depth` with metadata `encoding=jpeg`.

If no color image conversion is possible for the source format, that frame is
skipped.

## Inputs

- `tick`: trigger to capture one color + depth frame set. The node dispatches on
  any input event, so a single periodic `tick` drives capture.

## Outputs

- `image`: color frame, JPEG-encoded into an Arrow array. Metadata: `encoding`
  (`jpeg`). Decode with `cv2.imdecode(storage.to_numpy(), cv2.IMREAD_COLOR)`.
- `depth`: depth frame in meters as a flattened float32 Arrow array (values
  outside 0.01-15.0 m zeroed, temporally filtered). Reshape to the depth stream
  height x width.
- `image_depth`: colorized depth visualization (`COLORMAP_JET`), JPEG-encoded
  into an Arrow array. Metadata: `encoding` (`jpeg`).

## Environment variables

- `DEVICE_INDEX`: index of the Orbbec device to open from the enumerated device
  list. Default `0`.

## Usage

```yaml
nodes:
  - id: dora-pyorbbecksdk
    hub: dora-pyorbbecksdk@^0.5
    inputs:
      tick: dora/timer/millis/33
    outputs:
      - image
      - depth
      - image_depth
```

Follow the pyorbbecsdk installation instructions:
<https://github.com/orbbec/pyorbbecsdk>

## Hardware products supported by the pyorbbecsdk

| Products list    | Firmware version            |
| ---------------- | --------------------------- |
| Gemini 335       | 1.2.20                      |
| Gemini 335L      | 1.2.20                      |
| Gemini 336       | 1.2.20                      |
| Gemini 336L      | 1.2.20                      |
| Femto Bolt       | 1.0.6/1.0.9                 |
| Femto Mega       | 1.1.7/1.2.7                 |
| Gemini 2 XL      | Obox: V1.2.5 VL:1.4.54      |
| Astra 2          | 2.8.20                      |
| Gemini 2 L       | 1.4.32                      |
| Gemini 2         | 1.4.60 /1.4.76              |
| Astra+           | 1.0.22/1.0.21/1.0.20/1.0.19 |
| Femto            | 1.6.7                       |
| Femto W          | 1.1.8                       |
| DaBai            | 2436                        |
| DaBai DCW        | 2460                        |
| DaBai DW         | 2606                        |
| Astra Mini Pro   | 1007                        |
| Gemini E         | 3460                        |
| Gemini E Lite    | 3606                        |
| Gemini           | 3.0.18                      |
| Astra Mini S Pro | 1.0.05                      |

## Build

```bash
pip install .
```
