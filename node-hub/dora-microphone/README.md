# dora-microphone

Captures audio from the system microphone and emits raw PCM chunks.

## Behavior

`dora-microphone` opens the default input device with Python
[`sounddevice`](https://python-sounddevice.readthedocs.io/) as a single-channel
(mono) `int16` stream at `SAMPLE_RATE`. Incoming samples are buffered, and once
`MAX_DURATION` seconds have elapsed the buffer is flushed: the samples are
converted to `float32` and normalized to the `[-1, 1]` range (divided by
32768.0), then sent on the `audio` output.

The node polls its own inputs while recording; when an input event returns
`None` (the dataflow is finishing), it stops the stream and exits.

## Inputs

- `tick` — any event. Used only to detect when the dataflow is finished so the
  node can stop recording. Not required.

## Outputs

- `audio` — mono microphone audio sent in chunks as `float32` samples
  normalized to `[-1, 1]`, captured at `SAMPLE_RATE`.

## Environment variables

- `MAX_DURATION` (float, default `0.1`) — maximum buffering duration in seconds
  before a chunk is flushed as an `audio` output.
- `SAMPLE_RATE` (int, default `16000`) — microphone capture sample rate in Hz.

## Usage

```yaml
nodes:
  - id: dora-microphone
    hub: dora-microphone@^0.5
    outputs:
      - audio
```

## Build

```bash
pip install .
```
