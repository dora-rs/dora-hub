# dora-keyboard

Keyboard listener source — emits the character of each printable key as it is
pressed.

## Behavior

`dora-keyboard` connects as a dora node and listens to the system keyboard via
`pynput`. On every key **press** event, if the key has a printable character it
sends that character on the `char` output as a single-element UTF-8 string array
(metadata `{"primitive": "text"}`). Non-printable keys (modifiers, arrows, etc.)
produce no output.

It uses `node.next()` only as a liveness check: if it detects that the dora
input stream has closed, the listen loop exits. It does not consume or react to
any specific input id.

## Inputs

None — it is a source. (The node polls the dora event stream only to detect
shutdown, not to read named inputs.)

## Outputs

- `char`: the character of a printable key on press, as a single-element UTF-8
  string array.

## Environment variables

None.

## Usage

```yaml
nodes:
  - id: dora-keyboard
    hub: dora-keyboard@^0.5
    outputs:
      - char
```

## Build

```bash
pip install .
```
