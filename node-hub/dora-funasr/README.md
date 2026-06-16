# dora-funasr

Speech-to-text (ASR) node using FunASR's `paraformer-zh` model with `ct-punc`
punctuation.

## Behavior

On startup the node loads the FunASR `AutoModel` (`paraformer-zh` +
`ct-punc`) and warms it up with a silent buffer.

For each `INPUT` event:

- If the input id contains `text_noise`, the first element is read as a string,
  has its brackets/parentheses stripped, and is stored as noise text used to
  filter later results.
- Any other input is treated as audio: its value is converted to a numpy array
  and passed to `model.generate(...)`. The recognized text has spaces removed;
  empty or `"."`-only results are skipped. Otherwise the text is emitted on
  `text` (with metadata `language=zh`, `primitive=text`) and the same text is
  also emitted on `speech_started`.

The node prints the raw recognized text to stdout each time.

## Inputs

- `audio` (required): raw audio samples as an Arrow array. Any input id NOT
  containing `text_noise` is dispatched here.
- `text_noise` (optional): text whose words are removed from results as known
  noise. Matched by any input id containing `text_noise`.

## Outputs

- `text`: recognized text, single-element UTF-8 Arrow array
  (`language=zh`, `primitive=text`).
- `speech_started`: emitted with the same text payload whenever a non-empty
  transcription is produced.

## Environment variables

None.

## Usage

```yaml
nodes:
  - id: microphone
    path: dora-microphone
    outputs:
      - audio
  - id: dora-funasr
    hub: dora-funasr@^0.5
    inputs:
      audio: microphone/audio
    outputs:
      - text
      - speech_started
```

## Build

```bash
pip install .
```
