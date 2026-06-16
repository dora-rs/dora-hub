# dora-openai-websocket

Bridges a dora dataflow to the OpenAI Realtime/websocket API. It runs an HTTP +
WebSocket server that speaks the OpenAI Realtime protocol so OpenAI-compatible
clients can talk to a dora-backed model pipeline.

## Behavior

On startup the node binds an HTTP/WebSocket server (default `0.0.0.0:8123`) and
connects as a dora node. It serves four routes:

- `/realtime`, `/v1/realtime` — WebSocket upgrade speaking the OpenAI Realtime
  protocol.
- `/v1/chat/completions` — OpenAI chat-completions endpoint (non-streaming;
  streaming requests are answered with a full response).
- `/models`, `/v1/models` — lists a single static model (`gpt-5`).

When a Realtime client connects and sends a `session.update`, the node emits the
session `instructions` as `system_prompt` and the `tools` list as `tools`.
Client `input_audio_buffer.append` audio is base64-decoded, converted from PCM16
to float32, and emitted as `audio` (with `sample_rate: 16000` metadata). Client
`conversation.item.create` message items are emitted as `text`,
`function_call_output` items as `function_call_output`, and `response.create`
instructions (and chat-completion requests) as `response.create`.

In the other direction, dora inputs are streamed back to the connected client as
Realtime events: `audio` becomes PCM16 `response.audio.delta` + `response.done`,
`text` becomes `response.text.delta` (or function-call events when wrapped in
`<tool_call>`), `transcript` becomes `response.audio_transcript.delta`, and the
speech inputs become `input_audio_buffer.speech_started` /
`input_audio_buffer.speech_stopped`. Inputs are matched by substring on the id.

## Inputs

- `audio`: float32 PCM audio, sent to the client as audio deltas.
- `text`: Utf8 assistant text, sent as text deltas (or function-call events).
- `transcript`: Utf8 transcription text, sent as transcript deltas.
- `speech_started`: signals the user started speaking.
- `speech_stopped`: signals the user stopped speaking.

## Outputs

- `audio`: float32 PCM audio decoded from client audio appends
  (`sample_rate: 16000` metadata).
- `text`: Utf8 text from client conversation message items.
- `system_prompt`: Utf8 session instructions from `session.update`.
- `tools`: Utf8 JSON tool list from `session.update`.
- `function_call_output`: Utf8 output from `function_call_output` items.
- `response.create`: Utf8 prompt text from client `response.create` or a
  chat-completions request.

## Environment variables

- `HOST` (string, default `0.0.0.0`): address the server binds to.
- `PORT` (string, default `8123`): TCP port the server listens on.

## Usage

```yaml
nodes:
  - id: dora-openai-websocket
    hub: dora-openai-websocket@^0.5
    inputs:
      audio: tts/audio
      text: llm/text
      transcript: stt/transcript
    env:
      PORT: "8123"
```

## Build

```bash
cargo build --release --target-dir target
```
