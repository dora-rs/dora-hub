# dora-kokoro-tts

Text-to-speech using Kokoro — converts text input into synthesized audio output.

## Behavior

On startup the node loads a Kokoro `KPipeline` (model from `REPO_ID`, language
from `LANGUAGE`) and warms it up with a short utterance using `VOICE`.

For each `text` input event it:

- Strips any `<tool_call>...</tool_call>` sections from the text (skipping the
  event entirely if nothing remains).
- Splits the remaining text on punctuation (`。 , . ， ? ! :`) into chunks.
- Per chunk, auto-detects Chinese characters and switches the pipeline
  `lang_code` to `"z"` for Chinese text (or back to `LANGUAGE` otherwise).
- Synthesizes each chunk at speed 1.2 and emits the resulting audio.

## Inputs

- `text` — UTF-8 text to synthesize into speech (first element of the Arrow
  array is read).

## Outputs

- `audio` — synthesized speech as a float Arrow array, emitted once per
  synthesized chunk. Output metadata carries `sample_rate: 24000`.

## Environment variables

- `REPO_ID` — Hugging Face repo id of the Kokoro model. Default
  `hexgrad/Kokoro-82M`.
- `LANGUAGE` — default Kokoro `lang_code` (e.g. `a` for English). The node
  auto-switches to `z` for Chinese text. Default `a`.
- `VOICE` — Kokoro voice id used for synthesis. Default `af_heart`.

## Usage

```yaml
nodes:
  - id: dora-kokoro-tts
    hub: dora-kokoro-tts@^0.5
    inputs:
      text: some-llm/text
    outputs:
      - audio
```

## Build

```bash
pip install .
```
