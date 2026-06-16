# replay-client

Replays a previously recorded episode into a dataflow. It reads a parquet
dataset of recorded `action` and `joints` columns, filters to a single episode,
and emits one recorded frame each time it is ticked.

## Behavior

On startup the node opens `dataset.parquet` inside the configured recording
directory and keeps only the rows whose `episode_index` matches the configured
episode. It then steps through those rows one at a time:

- Each `pull_position` tick advances the internal frame counter by one and emits
  the corresponding recorded position on `position`.
- When every frame has been consumed, the node emits `end` and exits.
- Receiving `end` also stops the node (it emits a final `end` and exits).
- A dataflow `ERROR` event aborts the node with an error.

## Inputs

- `pull_position`: tick that advances the replay by one frame and emits the next
  recorded position.
- `end`: stop signal; the node emits `end` and exits.

## Outputs

- `position`: the next recorded frame as a struct array with fields `joints`
  (the joint names) and `values` (the float32 action positions).
- `end`: an empty array, emitted once when the replay finishes or is stopped.

## Environment variables

- `DATASET_PATH` (required): path to the recording directory containing
  `dataset.parquet`.
- `EPISODE` (required, integer): episode index to replay (`episode_index`).

## Usage

```yaml
nodes:
  - id: replay-client
    hub: replay-client@^0.5
    inputs:
      pull_position: dora/timer/millis/10
    outputs:
      - position
      - end
    env:
      DATASET_PATH: /path/to/record
      EPISODE: "1"
```

## Build

```bash
pip install .
```
