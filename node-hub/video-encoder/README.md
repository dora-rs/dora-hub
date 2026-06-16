# video-encoder

Records incoming image frames per episode and encodes them into an H.264 MP4
video (LeRobot compatible).

## Behavior

`video-encoder` connects as a dora node (its node name defaults to
`video_encoder`, overridable with `--name`). On startup it requires the
`VIDEO_NAME` and `FPS` environment variables and raises a `ValueError` if either
is unset.

It tracks two pieces of state: whether it is currently recording and the current
episode index. Recording is toggled by the `episode_index` input:

- An `episode_index` value other than `-1` selects that episode and starts
  recording.
- An `episode_index` of `-1` stops recording and encodes the buffered frames.

While recording, each `image` input frame is reshaped to
`(height, width, channels)` and written to disk as a PNG under
`out/<dataflow_id>/videos/<VIDEO_NAME>_episode_<index>/frame_<n>.png`, and an
`image` output referencing the target video is emitted.

When recording stops (`episode_index == -1`), the buffered PNG frames are encoded
with ffmpeg (`libx264`, `pix_fmt yuv444p`, GOP size 2, rate `FPS`) into
`out/<dataflow_id>/videos/<VIDEO_NAME>_episode_<index>.mp4`, and the frame
counter resets.

An initial `image` output is also sent once at startup before any input arrives.

## Inputs

- `image` — an image frame as an Arrow struct with `width`, `height`,
  `channels`, and `data`. Buffered to disk as a PNG only while recording.
- `episode_index` — episode index integer. A value other than `-1` starts
  recording that episode; `-1` ends the episode and triggers MP4 encoding.

## Outputs

- `image` — a reference to the encoded video file as an Arrow struct
  `{path, timestamp}` (path relative to the videos directory; timestamp =
  `frame_count / FPS`). Emitted once at startup and once per recorded frame.

## Environment variables

- `VIDEO_NAME` (required) — base name for the per-episode output directory and
  the resulting `.mp4` file.
- `FPS` (required, integer) — frames per second; used to compute frame
  timestamps and as the ffmpeg input/encode rate.

## Usage

```yaml
nodes:
  - id: video-encoder
    hub: video-encoder@^0.5
    inputs:
      image: camera/image
      episode_index: recorder/episode_index
    outputs:
      - image

    env:
      VIDEO_NAME: cam_up
      FPS: 30
```

## Build

```bash
pip install .
```
