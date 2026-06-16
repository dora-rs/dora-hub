# dora-vad

Voice activity detection using Silero VAD — emits speech segments from an audio
stream.

## Behavior

`dora-vad` buffers incoming audio and runs the Silero VAD model to detect the
beginning and ending of voice activity. When a complete speech segment is
detected it emits the buffered audio together with the start/end sample indices
of the speech. While speech is still ongoing at the end of the current buffer it
keeps accumulating audio and emits only a `timestamp_start` marker until the
segment closes. A maximum buffered duration bounds how long it waits.

## Inputs

- `audio`: audio samples at 8kHz or 16kHz. The `sample_rate` metadata key is read
  if present (defaults to 16000).

## Outputs

- `audio`: the buffered audio for the detected speech segment, carrying a
  `sample_rate` metadata key.
- `timestamp_start`: start sample index of the detected speech segment.
- `timestamp_end`: end sample index of the detected speech segment.

## Environment variables

- `MIN_SILENCE_DURATION_MS` (int, default `200`): minimum silence duration (ms)
  that separates speech segments.
- `MIN_SPEECH_DURATION_MS` (int, default `300`): minimum speech duration (ms) for
  a segment to be reported.
- `THRESHOLD` (float, default `0.4`): Silero VAD speech-probability threshold.
- `MAX_AUDIO_DURATION_S` (float, default `75`): maximum buffered audio duration
  (s) before flushing.
- `MIN_AUDIO_SAMPLING_DURATION_MS` (int, default `500`): minimum sampling
  duration (ms) accumulated before running detection.

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
