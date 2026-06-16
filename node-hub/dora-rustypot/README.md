# dora-rustypot

Servo/motor bus control over a serial port using the [rustypot](https://crates.io/crates/rustypot) library. Drives Feetech STS3215 servos (protocol v1): writes goal positions from a `pose` input and reads back present positions on `tick`.

## Behavior

On startup the node opens the serial port named by `PORT` at `BAUDRATE`, builds an STS3215 controller (protocol v1), and configures torque for the servos listed in `IDS`:

- If `TORQUE` is set, torque is enabled on every servo; if its value parses as an integer it is also applied as a torque limit.
- If `TORQUE` is unset, torque is disabled.

It then processes input events:

- On a `tick` input, it reads the present position of every configured servo and emits them as the `pose` output (with `encoding=jointstate` metadata).
- On a `pose` input, it casts the data to a vector of `f64` (one value per id) and writes them as goal positions.
- Any other input id is logged to stderr and ignored.

## Inputs

- `tick`: on any value, triggers a present-position read that is emitted as `pose`.
- `pose`: goal positions to write, a vector of `f64` (one per id, in `IDS` order).

## Outputs

- `pose`: present joint positions read from the servos, carrying an `encoding=jointstate` metadata parameter.

## Environment variables

- `PORT` (required): serial port device name, e.g. `/dev/ttyUSB0`.
- `BAUDRATE` (default `1000000`): serial baud rate.
- `IDS` (default `1,2,3,4,5,6`): comma/space-separated servo ids.
- `TORQUE` (optional): if set, enables torque on all servos; an integer value is also applied as a torque limit. Unset disables torque.

## Usage

```yaml
nodes:
  - id: dora-rustypot
    hub: dora-rustypot@^0.1
    inputs:
      tick: dora/timer/millis/20
      pose: planner/goal_position
    outputs:
      - pose
    env:
      PORT: /dev/ttyUSB0
      BAUDRATE: "1000000"
      IDS: "1,2,3,4,5,6"
```

## Build

```bash
cargo build --release --target-dir target
```
