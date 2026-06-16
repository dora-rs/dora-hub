# dora-vad

Voice activity detection using Silero VAD — emits speech segments from an audio
stream.

## Behavior

`dora-vad` accumulates incoming audio and runs the Silero VAD model over the
buffer to detect voice activity. Two cases:

- **Speech still ongoing** — if the detected speech runs to the end of the
  current buffer, the node emits a `timestamp_start` marker (the start index of
  that speech) and keeps accumulating, waiting for the speech to end.
- **Segment finalized** — once the speech ends before the end of the buffer (or
  after a few buffers with no further speech), the node emits the buffered
  `audio` and a `timestamp_end`, then resets the buffer.

So `timestamp_start` is a "speech is ongoing" signal, not a per-segment start —
a short utterance that completes within a single buffer emits `audio` +
`timestamp_end` with no preceding `timestamp_start`.

## Inputs

- `audio`: audio samples at 8kHz or 16kHz. The `sample_rate` metadata key is read
  if present (defaults to 16000).

## Outputs

- `audio`: the accumulated buffer for a finalized utterance, carrying a
  `sample_rate` metadata key. (The whole buffer is sent, not just the speech
  span.)
- `timestamp_end`: end sample index of the speech, emitted alongside `audio`
  when an utterance finalizes.
- `timestamp_start`: start sample index of speech that is still ongoing (emitted
  only while speech reaches the end of the current buffer). Not emitted for an
  utterance that finalizes within one buffer.

## Environment variables

- `MIN_SILENCE_DURATION_MS` (int, default `200`): minimum silence duration (ms)
  that separates speech segments.
- `MIN_SPEECH_DURATION_MS` (int, default `300`): minimum speech duration (ms) for
  a segment to be reported.
- `THRESHOLD` (float, default `0.4`): Silero VAD speech-probability threshold.

## Usage

```yaml
nodes:
  - id: dora-vad
    hub: dora-vad@^0.5
    inputs:
      audio: dora-microphone/audio
    outputs:
      - audio
      - timestamp_start
      - timestamp_end
```

## Build

```bash
pip install .
```
