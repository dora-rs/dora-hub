# dora-llama-cpp-python

LLM text generation via [llama-cpp-python](https://github.com/abetlen/llama-cpp-python).
Consumes input text and emits a generated response (text -> text) using a GGUF
model, with optional CUDA (Linux) / Metal (macOS) GPU acceleration.

## Behavior

On startup the node loads a GGUF model: if `MODEL_NAME_OR_PATH` points at an
existing local file it is loaded directly, otherwise the model is downloaded
from HuggingFace using `MODEL_NAME_OR_PATH` as the repo id and
`MODEL_FILE_PATTERN` to pick the file.

For each input event the node takes the first element of the value as the prompt
text. If `ACTIVATION_WORDS` is set, the prompt is only answered when it contains
at least one of those words; if it is empty, every input is answered. It then
calls `create_chat_completion` (optionally prefixed by a `SYSTEM_PROMPT` message)
and sends the generated content on the `text` output.

## Inputs

- `text`: the prompt to generate a response for. The first array element is read
  and used as the user message.

## Outputs

- `text`: the generated response string.

## Environment variables

- `SYSTEM_PROMPT` (string, default `""`): system prompt prepended to the chat. Empty means no system message.
- `MODEL_NAME_OR_PATH` (string, default `TheBloke/Llama-2-7B-Chat-GGUF`): local GGUF path, or HuggingFace repo id to download.
- `MODEL_FILE_PATTERN` (string, default `*Q4_K_M.gguf`): filename glob to select the GGUF file when downloading.
- `MAX_TOKENS` (int, default `512`): maximum tokens generated per response.
- `N_GPU_LAYERS` (int, default `0`): layers offloaded to GPU (e.g. `35` for full acceleration); `0` is CPU-only.
- `N_THREADS` (int, default `4`): CPU threads used for inference.
- `CONTEXT_SIZE` (int, default `4096`): maximum context window (`n_ctx`).
- `ACTIVATION_WORDS` (string, default `""`): space-separated trigger words; when set, only prompts containing one are answered. Empty answers everything.

## Usage

```yaml
nodes:
  - id: dora-llama-cpp-python
    hub: dora-rs/dora-llama-cpp-python
    inputs:
      text: source-node/text
    outputs:
      - text
    env:
      MODEL_NAME_OR_PATH: TheBloke/Llama-2-7B-Chat-GGUF
      MODEL_FILE_PATTERN: "*Q4_K_M.gguf"
      SYSTEM_PROMPT: "You're a very succinct AI assistant with short answers."
      MAX_TOKENS: 512
```

## Build

```bash
pip install .
```
