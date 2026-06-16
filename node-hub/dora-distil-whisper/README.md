# dora-distil-whisper

Transcribes (or translates) speech audio to text using a Whisper model — `mlx-whisper` on macOS, a HuggingFace `transformers` automatic-speech-recognition pipeline on every other platform.

## Behavior

`dora-distil-whisper` connects as a dora node and, for each input event:

- If the input id contains `text_noise`, the value is stored as noise text (parentheses/brackets stripped) and subtracted from later transcriptions to avoid echo; the noise filter expires after a few seconds.
- Otherwise the value is treated as audio (Arrow array → numpy) and transcribed. If the previous result ended with `...`, the new audio is concatenated onto the cached audio first.

After transcription it post-processes the text:

- Results matching a built-in `BAD_SENTENCES` list (hallucination fillers) are discarded.
- Repeated runs of words/characters are trimmed (`cut_repetition`).
- Stored `text_noise` words are removed.
- Empty, `.`-only, or `...`-ending results are not emitted; a `...`-ending result is cached so the next audio chunk is appended to it.

On macOS the `transformers`/torch path is skipped entirely and `mlx-whisper` (`mlx-community/whisper-large-v3-turbo`) is used; `TRANSLATE` is not applied on this path.

## Inputs

- `input` — audio samples to transcribe (Arrow array of floats). Any input id that does **not** contain `text_noise` is treated as audio.
- `text_noise` — text to subtract from subsequent transcriptions (e.g. the assistant's own spoken output). Optional.

## Outputs

- `text` — the transcribed/translated string (Arrow string array) with metadata `{language, primitive: text}`.
- `speech_started` — emitted alongside `text` carrying the same string; signals a finalized utterance.

## Environment variables

- `TARGET_LANGUAGE` (default `english`) — language for transcription, and target language when translating.
- `TRANSLATE` (default `False`) — when `True`/`true`, translate into `TARGET_LANGUAGE` instead of transcribing. Ignored on macOS.
- `MODEL_NAME_OR_PATH` (default `openai/whisper-large-v3-turbo`) — HuggingFace model id or local path for the non-macOS pipeline. (macOS always uses `mlx-community/whisper-large-v3-turbo`.)
- `USE_MODELSCOPE_HUB` (default `False`) — when `True`/`true` (non-macOS), download the model from ModelScope instead of HuggingFace.

## Usage

```yaml
nodes:
  - id: dora-distil-whisper
    hub: dora-distil-whisper@^0.5
    inputs:
      input: some-vad/audio
    outputs:
      - text
      - speech_started
```

## Build

```bash
pip install .
```
