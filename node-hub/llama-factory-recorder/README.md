# llama-factory-recorder

Records dataflow image and ground-truth inputs into [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
training data: it writes ShareGPT-format JSON samples and saves the corresponding
image frames as PNG files, ready for fine-tuning a vision-language model.

## Behavior

The node connects as a dora node and buffers inputs to assemble training samples:

- Any input whose id contains the substring `image` (e.g. `image_right`) is
  decoded from its `encoding`/`width`/`height` metadata (`bgr8`, `rgb8` or
  `jpeg`) and buffered as a frame.
- A non-empty `text` input overrides the prompt question (default
  `DEFAULT_QUESTION`) for the next sample.
- A `ground_truth` input triggers a write: the buffered frames are saved as PNGs
  and a ShareGPT message pair (`user` prompt with one `<image>` token per frame +
  `assistant` ground-truth) is appended to the dataset JSON. If no frames are
  buffered yet, the `ground_truth` is ignored.

On startup it requires `LLAMA_FACTORY_ROOT_PATH` and writes the dataset JSON,
`dataset_info.json` and image folders under `<LLAMA_FACTORY_ROOT_PATH>/data`. If
the dataset name already exists, an incremental suffix is appended so existing
data is not overwritten.

## Inputs

- `image`: an image frame to record. Any input id containing `image` is treated
  as an image; metadata must carry `encoding` (`bgr8`/`rgb8`/`jpeg`), `width` and
  `height`.
- `text`: optional prompt text; a non-empty value overrides `DEFAULT_QUESTION`.
- `ground_truth`: the assistant answer; receiving it writes a sample.

The node matches image inputs by substring, so multiple image streams can be
wired under distinct ids (each must contain `image`); the contract declares one
generic `image` input. To record arbitrary image-input names, run it directly via
`path:` (where the contract isn't enforced).

## Outputs

- `text`: echoes the recorded `ground_truth` text after a sample is written.

## Environment variables

- `DEFAULT_QUESTION` (string, default `Describe this image`): default user prompt
  used when no `text` input overrides it.
- `LLAMA_FACTORY_ROOT_PATH` (string, required): path to the LLaMA-Factory
  checkout. Output is written under `<path>/data`.
- `ENTRY_NAME` (string, default `dora_demo`): dataset name used for the JSON
  file, image subfolder and `dataset_info.json` key.

## Usage

```yaml
nodes:
  - id: llama-factory-recorder
    hub: llama-factory-recorder@^0.5
    inputs:
      image: camera/image
      text: planner/text
      ground_truth: planner/ground_truth
    outputs:
      - text
    env:
      DEFAULT_QUESTION: Describe this image
      LLAMA_FACTORY_ROOT_PATH: /path/to/LLaMA-Factory
```

After the run, the recorded dataset is in `<LLAMA_FACTORY_ROOT_PATH>/data`; point
your LLaMA-Factory training config's `dataset:` at the `ENTRY_NAME` you used.

## Build

```bash
pip install .
```
