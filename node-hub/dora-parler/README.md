# dora-parler

Text-to-speech node using Parler-TTS. Receives a text string and synthesizes
speech, playing the audio on the local speaker.

## Behavior

On startup the node loads a Parler-TTS model (selected by `MODEL_NAME_OR_PATH`)
and plays a short "Ready !" prompt. It then waits for `text` inputs. Each `text`
input is synthesized and streamed to the local audio output device via PyAudio.

While synthesizing, a `stop` input interrupts the current playback, and a new
`text` input interrupts the current playback and starts synthesizing the new
text.

The node plays audio directly through the system speaker (PyAudio) and does not
emit any dora output.

## Inputs

- `text`: UTF-8 string to synthesize and play as speech (required).
- `stop`: signal to interrupt the current synthesis/playback (optional).

## Outputs

None — audio is played directly on the local speaker; the node produces no dora
outputs.

## Environment variables

- `MODEL_NAME_OR_PATH` (default `ylacombe/parler-tts-mini-jenny-30H`): Parler-TTS
  model id or local path to load.
- `USE_MODELSCOPE_HUB` (default `false`): when `true`, download the model from
  ModelScope Hub instead of Hugging Face.

## Usage

```yaml
nodes:
  - id: text-source
    path: some-text-node
    outputs:
      - text
  - id: dora-parler
    hub: dora-parler@^0.5
    inputs:
      text: text-source/text
    env:
      MODEL_NAME_OR_PATH: ylacombe/parler-tts-mini-jenny-30H
```

## Build

```bash
pip install .
```
