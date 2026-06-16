# dora-transformers

Generate text with a Hugging Face transformers causal-LM (text in → text out).

## Behavior

`dora-transformers` loads a Hugging Face `AutoModelForCausalLM` and
`AutoTokenizer` for `MODEL_NAME` at startup, then runs as a dora node.

For each input event it reads the first element of the Arrow array as a user
message. If `ACTIVATION_WORDS` is set, the message is answered only when it
contains at least one of those words; otherwise every input is answered. The
message is fed through the tokenizer's chat template (optionally prefixed with
`SYSTEM_PROMPT`), generated with `repetition_penalty=1.2` and up to
`MAX_TOKENS` new tokens, decoded, and emitted on `text`.

When `HISTORY` is true the assistant reply is appended to the running
conversation so subsequent inputs see prior turns; otherwise each input is
answered independently.

The node reads the value of **any** input id it receives, but a `hub:` contract
must declare every wired input, so it declares one `text` input.

## Inputs

- `text`: prompt string. The first element of the Arrow array is read as the
  user message.

## Outputs

- `text`: generated response as a single-element UTF-8 Arrow array.

## Environment variables

- `MODEL_NAME` (string, default `Qwen/Qwen2.5-0.5B-Instruct`): Hugging Face
  model identifier or local path.
- `SYSTEM_PROMPT` (string, default empty): optional system message prepended to
  the conversation.
- `MAX_TOKENS` (int, default `512`): maximum new tokens per response.
- `DEVICE` (string, default `auto`): `device_map` passed to `from_pretrained`
  (e.g. `auto`, `cpu`, `cuda`, `mps`).
- `TORCH_DTYPE` (string, default `auto`): `torch_dtype` passed to
  `from_pretrained` (e.g. `auto`, `float16`).
- `HISTORY` (bool, default `false`): keep multi-turn history across inputs.
- `ACTIVATION_WORDS` (string, default empty): space-separated trigger words;
  when set, only inputs containing one respond.

## Usage

```yaml
nodes:
  - id: dora-transformers
    hub: dora-transformers
    inputs:
      text: source-node/text
    outputs:
      - text
    env:
      MODEL_NAME: "Qwen/Qwen2.5-0.5B-Instruct"
      SYSTEM_PROMPT: "You are a succinct AI assistant with short answers."
      MAX_TOKENS: "128"
```

## Build

```bash
pip install .
```
