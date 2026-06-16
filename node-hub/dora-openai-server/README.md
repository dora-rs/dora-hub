# dora-openai-server

An OpenAI-compatible HTTP server that bridges a dora dataflow with OpenAI-style
chat completion requests.

## Behavior

`dora-openai-server` runs a FastAPI/uvicorn HTTP server (bound to `0.0.0.0:8000`)
that exposes two OpenAI-style endpoints:

- `POST /v1/chat/completions`: extracts the first `user`-role message from the
  request, forwards its content into the dataflow as the `v1_chat_completions`
  output, then blocks waiting for a `v1_chat_completions` input event (10-second
  timeout) and returns its first element as the assistant message content.
- `GET /v1/models`: returns a static single-model list (`gpt-3.5-turbo`).

Before forwarding, the user message is parsed with `ast.literal_eval`; lists
become a multi-element Arrow array, while strings/ints/floats/dicts become a
single-element array (parse failures fall back to the raw string).

It connects as a dora node and shuts down when it receives a `STOP` event.

## Inputs

- `v1_chat_completions`: the dataflow's reply to a forwarded message. The first
  element is returned as the assistant content of the HTTP response.

## Outputs

- `v1_chat_completions`: the user message extracted from an incoming
  `POST /v1/chat/completions` request, forwarded into the dataflow.

## Environment variables

None. The HTTP host (`0.0.0.0`) and port (`8000`) are hardcoded.

## Usage

Wire the server's output to a node that processes the message, and feed that
node's reply back into the server:

```yaml
nodes:
  - id: dora-openai-server
    hub: dora-openai-server@^0.5
    inputs:
      v1_chat_completions: echo/data
    outputs:
      - v1_chat_completions
  - id: echo
    hub: dora-echo@^0.5
    inputs:
      data: dora-openai-server/v1_chat_completions
    outputs:
      - data
```

Then POST to `http://localhost:8000/v1/chat/completions`.

## Build

```bash
pip install .
```
