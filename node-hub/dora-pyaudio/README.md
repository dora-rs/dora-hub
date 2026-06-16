# dora-pyaudio

Plays received audio samples to the system speakers using PyAudio.

## Behavior

`dora-pyaudio` is an audio playback sink. It connects as a dora node and, on
each `audio` input event, plays the samples to the default output device via
PyAudio (`stream.write`). Floating-point arrays are scaled by 70000 and cast to
int16; int16 arrays are played as-is. Playback uses 1 channel (mono) at the
`paInt16` format. The sample rate is taken from the input message's
`sample_rate` metadata, falling back to the `SAMPLE_RATE` environment variable.

It produces no outputs.

Requires `portaudio` to be installed on the system: `brew install portaudio`
(macOS) or `sudo apt-get install portaudio19-dev python-all-dev` (Linux).

## Inputs

- `audio` — audio samples to play, as an Arrow array of int16 or floating-point
  values. Optional `sample_rate` metadata selects the playback rate.

## Outputs

None — it is a sink.

## Environment variables

- `SAMPLE_RATE` (int, default `16000`) — default playback sample rate in Hz,
  used when an input message omits `sample_rate` metadata.

## Usage

```yaml
nodes:
  - id: dora-pyaudio
    hub: dora-pyaudio@^0.5
    inputs:
      audio: some-node/audio
```

## Build

```bash
pip install .
```
