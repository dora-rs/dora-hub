# dora-sam3

Text-prompted segmentation node using [Meta SAM 3](https://github.com/facebookresearch/sam3).

## Inputs

| Input   | Type          | Description                           |
|---------|---------------|---------------------------------------|
| `image` | Arrow array   | RGB/BGR image with encoding metadata  |
| `text`  | Arrow string  | Text prompt (e.g., "the white handle")|
| `points`| Arrow array   | Point prompts as (x,y) pairs          |

## Outputs

| Output  | Type          | Description                           |
|---------|---------------|---------------------------------------|
| `masks` | Arrow array   | Binary mask (0/255), flattened        |

## Environment Variables

| Variable          | Default | Description                       |
|-------------------|---------|-----------------------------------|
| `SAM3_CONFIDENCE` | `0.3`   | Detection confidence threshold    |

## Prerequisites

Request access to the SAM3 checkpoint at https://huggingface.co/facebook/sam3 and authenticate with `hf auth login`.
