# dora-qwen2-5-vl

A Qwen2.5-VL vision-language node: it takes camera/image frames plus a text
prompt and emits a generated text response.

## Behavior

The node loads a `Qwen2_5_VLForConditionalGeneration` model (id or path from
`MODEL_NAME_OR_PATH`) and connects as a dora node. Device is selected
automatically: MPS on Apple Silicon, CUDA when available, otherwise CPU. If
`flash_attn` is installed it is used; otherwise it falls back to the default
attention. An optional PEFT adapter is loaded when `ADAPTER_PATH` is set.

For each event:

- An input whose id **contains** `image` is decoded into a PIL image. Raw
  `bgr8`/`rgb8` buffers are reshaped from metadata `width`/`height`; encoded
  `jpeg`/`jpg`/`jpe`/`bmp`/`webp`/`png` payloads are decoded with OpenCV. The
  latest frame per input id is cached. Any other encoding raises an error.
- An input whose id **contains** `text` triggers generation. The text strings
  (or the cached `DEFAULT_QUESTION` when empty) are turned into chat messages
  along with the cached image(s), and the model generates up to 128 new tokens.
  When `ACTIVATION_WORDS` is set, generation is skipped unless an incoming word
  matches. Special prefixes (`<|system|>`, `<|assistant|>`, `<|tool|>`,
  `<|user|>...`) are parsed into chat roles. With `HISTORY` enabled the
  conversation is retained across turns.

The generated text is sent on the `text` output.

## Inputs

- `image`: image frames to describe. Any input id containing `image` is treated
  as an image — raw `bgr8`/`rgb8` buffer or encoded `jpeg`/`jpg`/`jpe`/`bmp`/
  `webp`/`png` byte array, with metadata `encoding`/`width`/`height`.
- `text`: text prompt(s) that trigger generation. Any input id containing `text`
  is treated as text — a UTF-8 string array. Empty values fall back to the
  cached question.

## Outputs

- `text`: the generated UTF-8 text response (a single-element string array),
  with metadata `image_id` and `primitive` = `text`.

## Environment variables

- `MODEL_NAME_OR_PATH` (string, default `Qwen/Qwen2.5-VL-3B-Instruct`): model id
  or local path to load.
- `USE_MODELSCOPE_HUB` (bool, default `false`): download from ModelScope instead
  of Hugging Face.
- `SYSTEM_PROMPT` (string, default `You're a very succinct AI assistant, that
  describes image with a very short sentence.`): seeded as the first message.
- `ACTIVATION_WORDS` (string, default empty): space-separated words; when set,
  generation only runs when an incoming text contains one of them.
- `DEFAULT_QUESTION` (string, default `Describe this image`): used when a text
  input arrives empty.
- `IMAGE_RESIZE_RATIO` (float, default `1.0`): multiplier applied to image
  width/height before inference.
- `HISTORY` (bool, default `false`): keep conversation history across turns.
- `ADAPTER_PATH` (string, default empty): optional PEFT/LoRA adapter to load.

## Usage

```yaml
nodes:
  - id: dora-qwen2-5-vl
    hub: dora-qwen2-5-vl@^0.5
    inputs:
      image:
        source: camera/image
        queue_size: 1
      text: whisper/text
    outputs:
      - text
    env:
      DEFAULT_QUESTION: Describe the image in a very short sentence.
```

## Build

```bash
pip install .
```
