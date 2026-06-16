# dora-openai-proxy-server

An OpenAI-compatible HTTP proxy server that bridges a dora dataflow with
OpenAI-style chat-completion requests.

## Behavior

On startup the node binds an HTTP server on `0.0.0.0:8000` and connects as a
dora node. It exposes an OpenAI-style API:

- `POST /v1/chat/completions` — parses the incoming `ChatCompletionRequest`,
  flattens its messages into a list of prompt strings, and emits them on the
  `text` output. The HTTP request is held open and queued.
- `GET /v1/models` — returns a single static `custom model` entry owned by
  `dora`.
- `/echo` — returns `echo test`.
- Any other path — serves a static file from the `chatbot-ui` directory
  (falling back to `404.html`).

When the dataflow replies on the `text` input, the node pops the oldest queued
request, joins the received string array with newlines, wraps it in a
`chat.completion` object, and returns it as the HTTP response (supports both
streaming and non-streaming modes). Requests and replies are matched in FIFO
order.

The HTTP port (`8000`), bind address (`0.0.0.0`), and web-UI directory
(`chatbot-ui`) are hardcoded.

## Inputs

- `text`: the completion text to return to the pending HTTP chat request. A
  UTF-8 string array; its elements are joined with newlines and used as the
  assistant message of the next queued chat-completion response.

## Outputs

- `text`: the flattened prompt texts extracted from an incoming chat-completion
  request (one string per message), emitted on each `POST /v1/chat/completions`.

## Environment variables

None.

## Usage

Wire a model node so that this server's `text` output feeds the model and the
model's `text` output feeds back into this server:

```yaml
nodes:
  - id: openai-proxy-server
    build: cargo build --release --target-dir target
    path: dora-openai-proxy-server
    inputs:
      text: model-node/text
    outputs:
      - text

  - id: model-node
    path: model-node
    inputs:
      text: openai-proxy-server/text
    outputs:
      - text
```

## Build

```bash
cargo build --release --target-dir target
```
