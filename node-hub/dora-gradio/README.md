# dora-gradio

A Gradio web UI that bridges a browser into a dataflow, letting you stream text,
microphone audio, and camera video from `http://localhost:7860` as dora outputs.

## Behavior

On start the node launches a Gradio app on `0.0.0.0:7860` with two tabs:

- **Camera**: a WebRTC video stream. Each frame is resized to 640x480 and sent
  on `image` as flattened BGR8 bytes with `encoding`, `width`, `height`,
  `timestamp`, and `_time` metadata.
- **Audio and Text Input**: a streaming microphone input and a chat box. Mic
  audio is converted to mono float32, resampled to 16 kHz, buffered, and sent on
  `audio` (with `sample_rate`, `channels`, `timestamp` metadata) roughly every
  0.1 s. Submitting text sends it on `text` as a UTF-8 string.

A **Stop Server** button exits the process. The node reads no dataflow inputs —
it is a source driven entirely by the browser UI.

## Inputs

None — this node is a source.

## Outputs

- `text`: UTF-8 string from the chat box.
- `audio`: 16 kHz mono float32 audio chunks from the microphone, with
  `sample_rate`, `channels`, and `timestamp` (nanoseconds) metadata.
- `image`: 640x480 BGR8 camera frame (flattened uint8) from WebRTC, with
  `encoding`, `width`, `height`, `timestamp`, and `_time` metadata.

## Environment variables

None.

## Usage

```yaml
nodes:
  - id: dora-gradio
    hub: dora-gradio@^0.5
    outputs:
      - text
      - audio
      - image
  - id: terminal-print
    hub: terminal-print@^0.5
    inputs:
      data: dora-gradio/text
```

Open `http://localhost:7860` and type in the chat box to emit `text`.

## Build

```bash
pip install .
```
