# Dora Whisper Node for transforming speech to text

## YAML Specification

This node is supposed to be used as follows:

```yaml
- id: dora-distil-whisper
  build: pip install dora-distil-whisper
  path: dora-distil-whisper
  inputs:
    input: dora-vad/audio
  outputs:
    - text
  env:
    TARGET_LANGUAGE: english
```

## Examples

- speech to text
  - github: https://github.com/dora-rs/dora-hub/blob/main/examples/speech-to-text
- vision language model
  - github: https://github.com/dora-rs/dora-hub/blob/main/examples/vlm

## License

Dora-whisper's code and model weights are released under the MIT License
