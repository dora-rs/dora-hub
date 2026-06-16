# dora-internvl

InternVL vision-language model node for dora — takes an image and a text prompt
and emits a generated text response.

## Behavior

The node loads an InternVL model (default `OpenGVLab/InternVL2-1B`) on `cuda:0`
if a GPU is available, otherwise on CPU.

It keeps the most recently received `image` as the current frame. Generation is
triggered by a `text` input: the prompt is prefixed with the `<image>` token and
run against the stored frame via the model's chat interface, and the response is
emitted on the `text` output. If no image has been received yet, a `text` input
produces no output.

Images are accepted in `bgr8` or `rgb8` encoding (any other encoding raises an
error).

## Inputs

- `image`: image frame as a UInt8 Arrow array, with metadata fields `width`,
  `height`, and `encoding` (`bgr8` or `rgb8`). Stored as the latest frame; does
  not by itself trigger generation.
- `text`: prompt as a UTF-8 Arrow array (first element used). Triggers
  generation against the latest image.

## Outputs

- `text`: generated model response as a UTF-8 Arrow array. Emitted only when a
  `text` input arrives after at least one `image`.

## Environment variables

- `MODEL` (default `OpenGVLab/InternVL2-1B`): Hugging Face model path to load.

## Usage

```yaml
nodes:
  - id: dora-internvl
    hub: dora-internvl@^0.5
    inputs:
      image: camera/image
      text: prompt/text
    outputs:
      - text
```

## Build

```bash
pip install .
```
