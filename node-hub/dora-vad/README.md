# Speech Activity Detection(VAD)

This is using Silero VAD.

It detects beginning and ending of voice activity within a stream of audio and returns the parts that contains activity.

There's a maximum amount of voice duration, to avoid having no input for too long.

## Input/Output Specification

- inputs:
  - audio: 8kHz or 16kHz sample rate.
- outputs:
  - audio: Same as input but truncated

## YAML Specification

```yaml
- id: dora-vad
  description: Voice activity detection. See; <a href='https://github.com/snakers4/silero-vad'>sidero</a>
  build: pip install dora-vad
  path: dora-vad
  inputs:
    audio: dora-microphone/audio
  outputs:
    - audio
```

## Reference documentation

- dora-vad
  - github: https://github.com/dora-rs/dora-hub/blob/main/node-hub/dora-vad
- Silero VAD
  - github: https://github.com/snakers4/silero-vad

## Examples

- Speech to Text
  - github: https://github.com/dora-rs/dora-hub/blob/main/examples/speech-to-text

## License

The code and model weights are released under the MIT License.
