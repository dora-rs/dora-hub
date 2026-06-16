# dora-qwen

LLM text generation using Qwen2.5 — turns text prompts into generated text.

## Behavior

`dora-qwen` connects as a dora node, loads a Qwen2.5-0.5B-Instruct model, and
maintains a running conversation `history`. The backend is chosen by platform:

- **Darwin (macOS):** GGUF model via `llama-cpp` (`Llama.from_pretrained`,
  `create_chat_completion`).
- **Linux:** Hugging Face `transformers` (`AutoModelForCausalLM` /
  `AutoTokenizer`, `Qwen/Qwen2.5-0.5B-Instruct`).
- **Other:** MLX (`mlx_lm`, `mlx-community/Qwen2.5-0.5B-Instruct-8bit`).

For each input event:

- An id containing `system_prompt` appends a system message and continues
  (no generation).
- An id containing `tools` parses the value as a JSON tool list and continues
  (no generation).
- Any other id is treated as prompt text. Each string is appended to the
  history as a user turn, with optional inline role tags honored:
  `<|system|>`, `<|assistant|>`, `<|tool|>`, `<|user|>\n<|im_start|>`, and
  `<|user|>\n<|vision_start|>` (image url appended to the current user turn).

After ingesting prompt text, the node generates a response and emits it on
`text` — but only if `ACTIVATION_WORDS` is empty, or the prompt contains one of
those words. Per-message tools can be supplied via the `tools` metadata key.

## Inputs

- `text` (required) — prompt text used to drive generation. Any input id other
  than `system_prompt`/`tools` is treated as prompt text.
- `system_prompt` (optional) — any id containing `system_prompt` sets the
  system message; does not trigger generation.
- `tools` (optional) — any id containing `tools` supplies a JSON tool list;
  does not trigger generation.

## Outputs

- `text` — the generated assistant response, as a single-element UTF-8 string
  Arrow array.

## Environment variables

- `SYSTEM_PROMPT` — system prompt initializing the conversation. Default:
  `You're a very succinct AI assistant with short answers.`
- `MODEL_NAME_OR_PATH` — GGUF repo id / path (Darwin backend). Default:
  `Qwen/Qwen2.5-0.5B-Instruct-GGUF`.
- `MODEL_FILE_PATTERN` — GGUF filename pattern (Darwin backend). Default:
  `*fp16.gguf`.
- `MAX_TOKENS` (int) — parsed at startup but not currently applied; generation
  length is hardcoded per backend. Default: `512`.
- `N_GPU_LAYERS` (int) — GPU layers for the llama-cpp (Darwin) backend.
  Default: `0`.
- `N_THREADS` (int) — CPU threads for the llama-cpp (Darwin) backend.
  Default: `4`.
- `CONTEXT_SIZE` (int) — context window (`n_ctx`) for the llama-cpp (Darwin)
  backend. Default: `4096`.
- `TOOLS_JSON` — JSON tool list applied at startup as the default tools.
- `ACTIVATION_WORDS` — space-separated words; if set, generation only runs when
  the prompt contains one of them. Empty means always generate.

## Usage

```yaml
nodes:
  - id: dora-qwen
    hub: dora-qwen@^0.5
    inputs:
      text: some-node/text
    outputs:
      - text
```

## Build

```bash
pip install .
```
