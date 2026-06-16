# dora-kit-car

Differential-drive ("kit car") mobile-robot controller. Receives velocity
commands and drives the robot forward/backward and turns it left/right by
writing chassis speed frames to a serial port.

## Behavior

`dora-kit-car` connects as a dora node and opens the serial port named by the
`SERIAL_PORT` environment variable at 115200 baud (8N1, no flow control). For
every input event it receives, it interprets the payload as an Apache Arrow
`Float64Array`. When the array has exactly 6 elements it reads them as a ROS
`geometry_msgs/Twist` vector `[x, y, z, rx, ry, rz]`, takes the linear `x`
(index 0) and angular `rz` (index 5), encodes them into a chassis command frame,
and writes that frame to the serial port. Arrays of any other length are
ignored.

The node listens on **any** input id (it does not match on a specific name), so
the wired input can be called anything; the Hub contract declares it as `tick`.
It produces no outputs.

## Inputs

- `tick`: velocity command as a `Float64Array` of length 6 in
  `geometry_msgs/Twist` order `[x, y, z, rx, ry, rz]`. Only `x` (forward/back)
  and `rz` (turn) are used.

## Outputs

None — it is an actuator sink.

## Environment variables

- `SERIAL_PORT` (string, default `/dev/ttyUSB0`): serial port device the car
  chassis is connected to.

## Usage

```yaml
nodes:
  - id: keyboard-listener
    build: pip install dora-keyboard
    path: dora-keyboard
    inputs:
      tick: dora/timer/millis/10
    outputs:
      - twist # e.g. [2.0, 0.0, 0.0, 0.0, 0.0, 1.0]

  - id: dora-kit-car
    hub: dora-kit-car@^0.3
    inputs:
      tick: keyboard-listener/twist
    env:
      SERIAL_PORT: /dev/ttyUSB0
```

## Build

```bash
cargo build --release --target-dir target
```
