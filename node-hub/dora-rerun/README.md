# dora-rerun

Log images, depth, boxes, points, series, text, and URDF robot state to a
[Rerun.io](https://rerun.io) viewer.

## Behavior

`dora-rerun` connects as a dora node and forwards every input event to a Rerun
recording stream. On startup it chooses how to deliver data based on
`OPERATING_MODE`:

- **SPAWN** (default): launches a local Rerun viewer.
- **CONNECT**: connects over gRPC to a viewer at `RERUN_SERVER_ADDR`.
- **SAVE**: writes a `.rerun` archive under `out/<dataflow-id>/`.

It also loads any URDF robots declared via `*_urdf` / `*_URDF` environment
variables (optionally positioned with a matching `*_transform` variable), and
logs the value of the `README` variable as a viewer text document if set.

For each input it picks a visualization primitive: it first reads a `primitive`
metadata parameter, and if absent infers one from a keyword contained in the
input id (e.g. an id containing `image`). An id containing more than one keyword,
or an unknown primitive, is an error. The supported primitives are: `image`,
`depth`, `text`, `boxes2d`, `boxes3d`, `masks`, `jointstate`, `pose`, `series`,
`points3d`, `points2d`, and `lines3d`.

Some primitives read extra per-message metadata parameters — for example
`width`/`height`/`encoding` for `image` and `depth`, `format` for `boxes2d`/
`boxes3d`, `color`/`radii`/`radius` for points and lines.

## Inputs

The node selects behavior by primitive (from metadata or the input id). Declared
input ids:

- `image`: RGB/BGR or encoded (jpeg/png/avif) image bytes.
- `depth`: depth map; rendered as a 3D DepthImage when camera metadata is present.
- `text`: UTF-8 string array (Chinese characters are transliterated to pinyin).
- `boxes2d`: 2D bounding boxes.
- `boxes3d`: 3D bounding boxes.
- `masks`: float or boolean mask array (cached).
- `jointstate`: joint positions applied to a loaded URDF chain.
- `pose`: joint positions applied to a loaded URDF chain.
- `series`: float series logged as scalars.
- `points3d`: flat array of xyz triples.
- `points2d`: flat array of xy pairs.
- `lines3d`: flat array of xyz triples forming a line strip.

## Outputs

None — it is a visualization sink.

## Environment variables

- `OPERATING_MODE`: `SPAWN` (default), `CONNECT`, or `SAVE`. Unknown values fall
  back to `SPAWN`.
- `RERUN_SERVER_ADDR`: gRPC address used in `CONNECT` mode. Default
  `127.0.0.1:9876`.
- `RERUN_MEMORY_LIMIT`: viewer memory limit (e.g. `25%` or `4GB`). Default `25%`.
- `README`: optional Markdown text logged as a README document in the viewer.
- `*_urdf` / `*_URDF`: each such variable points to a URDF file (or a
  `*_description` robot-descriptions name) to load. An optional matching
  `*_transform` variable (space-separated `x y z` or `x y z qx qy qz qw`)
  positions it.

## Usage

```yaml
nodes:
  - id: camera
    path: opencv-video-capture
    outputs:
      - image
  - id: dora-rerun
    hub: dora-rerun@^0.5
    inputs:
      image: camera/image
    env:
      OPERATING_MODE: SPAWN
```

## Build

```bash
cargo build --release --target-dir target
```
