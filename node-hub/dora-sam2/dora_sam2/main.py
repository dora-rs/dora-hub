"""TODO: Add docstring."""

import cv2
import numpy as np
import pyarrow as pa
import torch
from dora import Node
from PIL import Image
from sam2.build_sam import build_sam2_camera_predictor
from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download(
    repo_id="facebook/sam2-hiera-tiny", 
    filename="sam2_hiera_tiny.pt"
)
model_cfg = "sam2_hiera_t.yaml"

predictor = build_sam2_camera_predictor(model_cfg, ckpt_path)


def main():
    """TODO: Add docstring."""
    pa.array([])  # initialize pyarrow array
    node = Node()
    frames = {}
    last_pred = None
    labels = None
    return_type = pa.Array
    image_id = None
    is_tracking = False
    object_id = 1
    for event in node:
        event_type = event["type"]

        if event_type == "INPUT":
            event_id = event["id"]

            if "image" in event_id:
                storage = event["value"]
                metadata = event["metadata"]
                encoding = metadata["encoding"]
                width = metadata["width"]
                height = metadata["height"]

                if (
                    encoding == "bgr8"
                    or encoding == "rgb8"
                    or encoding in ["jpeg", "jpg", "jpe", "bmp", "webp", "png"]
                ):
                    channels = 3
                    storage_type = np.uint8
                else:
                    error = f"Unsupported image encoding: {encoding}"
                    raise RuntimeError(error)

                if encoding == "bgr8":
                    frame = (
                        storage.to_numpy()
                        .astype(storage_type)
                        .reshape((height, width, channels))
                    )
                    frame = frame[:, :, ::-1]  # OpenCV image (BGR to RGB)
                elif encoding == "rgb8":
                    frame = (
                        storage.to_numpy()
                        .astype(storage_type)
                        .reshape((height, width, channels))
                    )
                elif encoding in ["jpeg", "jpg", "jpe", "bmp", "webp", "png"]:
                    storage = storage.to_numpy()
                    frame = cv2.imdecode(storage, cv2.IMREAD_COLOR)
                    frame = frame[:, :, ::-1]  # OpenCV image (BGR to RGB)
                else:
                    raise RuntimeError(f"Unsupported image encoding: {encoding}")
                image = Image.fromarray(frame)
                frames[event_id] = image
                if not is_tracking:
                    node.send_output("masks", pa.array([]), metadata={"primitive": "masks"})
                    continue

                with (
                    torch.inference_mode(),
                    torch.autocast(
                        "cuda",
                        dtype=torch.bfloat16,
                    ),
                ):
                    _, out_mask_logits = predictor.track(frames[image_id])
                    if len(out_mask_logits) == 0:
                        node.send_output("masks", pa.array([]), metadata={"primitive": "masks"})
                    else:
                        masks = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()                    
                        match return_type:
                            case pa.Array:
                                node.send_output(
                                    "masks",
                                    pa.array(masks.ravel()),
                                    metadata={
                                        "image_id": image_id,
                                        "width": frames[image_id].width,
                                        "height": frames[image_id].height,
                                        "primitive": "masks"
                                    },
                                )
                            case pa.StructArray:
                                node.send_output(
                                    "masks",
                                    pa.array(
                                        [
                                            {
                                                "masks": masks.ravel(),
                                                "labels": event["value"]["labels"],
                                            },
                                        ],
                                    ),
                                    metadata={
                                        "image_id": image_id,
                                        "width": frames[image_id].width,
                                        "height": frames[image_id].height,
                                        "primitive": "masks"
                                    },
                                )

            if "boxes2d" in event_id:
                if len(event["value"]) == 0:
                    node.send_output("masks", pa.array([]), {"primitive": "masks"})
                    continue
                if isinstance(event["value"], pa.StructArray):
                    boxes2d = event["value"][0].get("bbox").values.to_numpy()
                    labels = (
                        event["value"][0]
                        .get("labels")
                        .values.to_numpy(zero_copy_only=False)
                    )
                    return_type = pa.Array
                else:
                    boxes2d = event["value"].to_numpy()
                    labels = None
                    return_type = pa.Array

                metadata = event["metadata"]
                encoding = metadata["encoding"]
                if encoding != "xyxy":
                    raise RuntimeError(f"Unsupported boxes2d encoding: {encoding}")
                boxes2d = boxes2d.reshape(-1, 4)
                image_id = metadata["image_id"]
                with (
                    torch.inference_mode(),
                    torch.autocast(
                        "cuda",
                        dtype=torch.bfloat16,
                    ),
                ):
                    predictor.set_image(frames[image_id])
                    masks, _scores, last_pred = predictor.predict(
                        box=boxes2d,
                        point_labels=labels,
                        multimask_output=False,
                    )

                    if len(masks.shape) == 4:
                        masks = masks[:, 0, :, :]
                        last_pred = last_pred[:, 0, :, :]
                    else:
                        masks = masks[0, :, :]
                        last_pred = last_pred[0, :, :]

                    masks = masks > 0
                    metadata["image_id"] = image_id
                    metadata["width"] = frames[image_id].width
                    metadata["height"] = frames[image_id].height
                    ## Mask to 3 channel image
                    match return_type:
                        case pa.Array:
                            metadata["primitive"] = "masks"
                            node.send_output("masks", pa.array(masks.ravel()), metadata)
                        case pa.StructArray:
                            metadata["primitive"] = "masks"
                            node.send_output(
                                "masks",
                                pa.array(
                                    [
                                        {
                                            "masks": masks.ravel(),
                                            "labels": event["value"]["labels"],
                                        },
                                    ],
                                ),
                                metadata,
                            )
            elif "points" in event_id:
                points = event["value"].to_numpy().reshape((-1, 2))
                return_type = pa.Array
                if len(frames) == 0:
                    continue
                first_image = next(iter(frames.keys()))
                image_id = event["metadata"].get("image_id", first_image)
                with (
                    torch.inference_mode(),
                    torch.autocast(
                        "cuda",
                        dtype=torch.bfloat16,
                    ),
                ):
                    predictor.load_first_frame(frames[image_id])
                    labels = [1 for i in range(len(points))]
                    _, _, out_mask_logits = predictor.add_new_prompt(
                        frame_idx=0,
                        obj_id=object_id,
                        points=points,
                        labels=labels
                    )
                    is_tracking = True
                    masks = (out_mask_logits[0] > 0.0).cpu().numpy().squeeze()
                    match return_type:
                        case pa.Array:
                            node.send_output(
                                "masks",
                                pa.array(masks.ravel()),
                                metadata={
                                    "image_id": image_id,
                                    "width": frames[image_id].width,
                                    "height": frames[image_id].height,
                                    "primitive": "masks"
                                },
                            )
                        case pa.StructArray:
                            node.send_output(
                                "masks",
                                pa.array(
                                    [
                                        {
                                            "masks": masks.ravel(),
                                            "labels": event["value"]["labels"],
                                        },
                                    ],
                                ),
                                metadata={
                                    "image_id": image_id,
                                    "width": frames[image_id].width,
                                    "height": frames[image_id].height,
                                    "primitive": "masks"
                                },
                            )

        elif event_type == "ERROR":
            print("Event Error:" + event["error"])


if __name__ == "__main__":
    main()
