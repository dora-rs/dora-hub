# dora-mistral-rs

LLM text generation via [mistral.rs](https://github.com/EricLBuehler/mistral.rs) — receives a prompt string and emits the generated completion.

## Behavior

`dora-mistral-rs` connects as a dora node and builds a text model with mistral.rs's `TextModelBuilder`, loading the model named by `MODEL_PATH_OR_NAME` (defaults to `Qwen/Qwen2.5-0.5B-Instruct`).

For each `text` input it receives, it reads the UTF-8 string, sends it to the model as a single user-role chat message, and emits the first choice's content string as a `text` output. The output carries the input event's metadata parameters.

Input events with any other id are logged to stderr and otherwise ignored.

## Inputs

- `text`: prompt string sent to the model as a user message.

## Outputs

- `text`: generated completion text returned by the model.

## Environment variables

- `MODEL_PATH_OR_NAME` (string, default `Qwen/Qwen2.5-0.5B-Instruct`): Hugging Face model id or local path loaded by mistral.rs.

## Usage

```yaml
nodes:
  - id: dora-mistral-rs
    hub: dora-mistral-rs@^0.1
    inputs:
      text: prompt-source/text
    env:
      MODEL_PATH_OR_NAME: Qwen/Qwen2.5-0.5B-Instruct
```

## Build

```bash
cargo build --release --target-dir target
```
