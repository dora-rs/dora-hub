# lebai-client

Dora node client for Lebai robotic arms. It connects to the arm, then drives it
from incoming Cartesian/joint move commands, claw control, and teach mode, and
keeps a JSON-backed library of named poses and recordings.

## Behavior

On startup the node calls `lebai_sdk.init()`, connects to the arm at `LEBAI_IP`,
starts the arm (`start_sys`), and reads the current TCP pose and joint position.
It loads a pose library from `pose_library.json` (created empty if absent) and
re-saves it after every input event.

It then processes input events by id:

- `claw` — sets the claw to the given position.
- `movec` — relative Cartesian move `[dx, dy, dz, drx, dry, drz, t]`; target is
  solved with inverse kinematics and executed with `move_pvat`. Skipped while
  teaching.
- `movej` — relative joint move; the first 6 values are added to the current
  joint position and executed. Skipped while teaching.
- `stop` — stops motion and resyncs cached TCP/joint state from the arm.
- `save` — stores the current joint pose under the given name.
- `go_to` — moves to a saved pose by name (waits for the move to finish).
  Skipped while teaching.
- `record` — starts a named recording; later `movej`/`movec`/`go_to` events are
  appended to it.
- `cut` — stops the active recording.
- `teach` — toggles teach (free-drive) mode on; a second `teach` toggles it off.
- `end_teach` — exits teach mode.
- `play` — replays a named recording, issuing each stored move in turn; a `stop`
  input received during playback aborts it.

Any other input id is ignored. The node produces no outputs.

## Inputs

- `claw`: single value for the claw position.
- `movec`: `[dx, dy, dz, drx, dry, drz, t]` relative Cartesian move.
- `movej`: relative joint move (first 6 values used).
- `stop`: stop current motion.
- `save`: string name to save the current pose under.
- `go_to`: string name of a saved pose to move to.
- `record`: string name for a new recording.
- `cut`: stop the active recording.
- `teach`: toggle teach mode.
- `end_teach`: exit teach mode.
- `play`: string name of a recording to replay.

## Outputs

None — this node is a sink that commands the arm.

## Environment variables

- `LEBAI_IP` (default `10.42.0.253`): IP address of the Lebai arm to connect to.

## Usage

```yaml
nodes:
  - id: lebai-client
    hub: lebai-client@^0.5
    env:
      LEBAI_IP: "10.42.0.253"
    inputs:
      movej: commander/movej
      movec: commander/movec
      claw: commander/claw
      stop: commander/stop
```

## Build

```bash
pip install .
```
