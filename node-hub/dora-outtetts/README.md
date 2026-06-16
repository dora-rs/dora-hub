# dora-outtetts

Text-to-speech node using OuteTTS (`OuteAI/OuteTTS-0.2-500M`). It turns text into
a synthesized waveform.

> dora-outtetts is no longer maintained in favor of dora-kokorotts.

## Behavior

On startup the node loads an OuteTTS interface — the HuggingFace model by default,
or a GGUF (llama.cpp) model when `INTERFACE` is set to anything other than `HF` —
and loads the built-in default speaker `male_1`.

For each input event:

- `text`: takes the first string element of the value, calls
  `interface.generate(...)` (temperature 0.1, repetition penalty 1.1), and emits
  the resulting waveform on `audio`.
- `tick`: an input with the exact id `tick` is logged and produces nothing.
  Inputs with any other id are ignored (there is no catch-all branch).

The CLI entrypoint also supports `--create-speaker <audio>` (builds and saves
`speaker.json`) and `--test` (runs the package tests); neither is part of the
dataflow path.

## Inputs

- `text`: UTF-8 text to synthesize, read as a single string element.
- `tick`: optional heartbeat matched on the exact id `tick`; only logged, no audio produced.

## Outputs

- `audio`: synthesized waveform as a flat float array. Metadata includes
  `sample_rate` (from the model) and `language` (`"en"`).

## Environment variables

- `INTERFACE` (default `HF`): `HF` uses the HuggingFace model; any other value
  uses the GGUF backend.
- `GGUF_MODEL_PATH` (default a cached `OuteTTS-0.2-500M-Q4_0.gguf` path): GGUF
  model file, used only when `INTERFACE` is not `HF`.
- `PATH_SPEAKER` (default `speaker.json`): path checked at startup for a speaker
  file; the default speaker `male_1` is loaded regardless.

## Usage

```yaml
nodes:
  - id: llm
    hub: some-llm-node
    outputs:
      - text
  - id: dora-outtetts
    hub: dora-outtetts@^0.5
    inputs:
      text: llm/text
    outputs:
      - audio
    env:
      INTERFACE: "HF"
```

## Build

```bash
pip install .
```
