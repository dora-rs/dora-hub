# dora-qwen-omni

Multimodal llama.cpp (mtmd) node. It loads a GGUF model plus a multimodal
projection file and, for each input event, builds a chat prompt from the
received strings, runs the model, and emits the generated text.

## Behavior

The node connects as a dora node (`init_from_env`) and loops over input events.
For every input event it:

- Clears the chat history and KV cache.
- Reads the input data as a UTF-8 string array and inspects each string by
  prefix marker:
  - `<|user|>\n<|im_start|>\n` — the remainder is appended to the text prompt.
  - `<|user|>\n<|vision_start|>\n` — the remainder is treated as an image. A
    `data:image/...` base64 data URI is decoded inline; otherwise, if the value
    parses as a URL it is fetched with a blocking HTTP GET. The loaded image is
    added as a media bitmap and a default media marker is inserted into the
    prompt.
  - `<|system|>\n` — ignored.
  - Anything else — appended to the text prompt.
- Evaluates the assembled user message and generates a response, then emits it.

The input id is not inspected — any wired input triggers a turn. Model,
multimodal projection, prompt, and runtime knobs are supplied as command-line
arguments (clap), not environment variables. `--model` and `--mmproj` must
point to existing files or the process exits before connecting.

## Inputs

- `tick`: any input event. Its data is read as a string array and parsed by the
  markers described above. The input id itself is ignored.

## Outputs

- `text`: the model's generated text response for the received prompt/image.

## Environment variables

None. Configuration is passed via command-line arguments:

- `--model <PATH>` (required): path to the GGUF model file.
- `--mmproj <PATH>` (required): path to the multimodal projection file.
- `--prompt <TEXT>` (required): text prompt.
- `--image <PATH>`, `--audio <PATH>`: media file paths (repeatable).
- `--n-predict <N>` (default 4096), `--threads <N>` (default 128),
  `--n-tokens <N>` (default 16384), `--chat-template <TEMPLATE>`,
  `--no-gpu`, `--no-mmproj-offload`, `--marker <TEXT>`.

## Usage

Because the required model/prompt settings are CLI arguments, run the node via
`path:` with `args:` (a `hub:` reference cannot pass command-line arguments):

```yaml
nodes:
  - id: dora-qwen-omni
    build: cargo build --release --target-dir target
    path: target/release/dora-qwen-omni
    args: >-
      --model ./model.gguf --mmproj ./mmproj.gguf
      --prompt "What is in the picture?"
    inputs:
      tick: some-node/output
    outputs:
      - text
```

## Build

```bash
cargo build --release --target-dir target
```
