"""Dora SAM3 node — text-prompted segmentation with Meta SAM 3."""

import os

import cv2
import numpy as np
import pyarrow as pa
import torch
from dora import Node
from PIL import Image


CONFIDENCE = float(os.getenv("SAM3_CONFIDENCE", "0.3"))
USE_FP16 = os.getenv("SAM3_FP16", "true").lower() in ("1", "true", "yes")


def _enable_fp16(model):
    """Convert SAM3 model to fp16 and register hooks to auto-cast inputs."""
    import torchvision.ops

    model = model.half()

    # Auto-cast float tensor inputs to match each module's parameter dtype
    def cast_inputs_hook(module, args, kwargs):
        param = next(module.parameters(recurse=False), None)
        if param is None:
            param = next(module.parameters(), None)
        if param is None:
            return args, kwargs
        dtype = param.dtype
        new_args = tuple(
            a.to(dtype) if isinstance(a, torch.Tensor) and a.is_floating_point() and a.dtype != dtype else a
            for a in args
        )
        new_kwargs = {
            k: v.to(dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() and v.dtype != dtype else v
            for k, v in kwargs.items()
        }
        return new_args, new_kwargs

    for _, mod in model.named_modules():
        if any(True for _ in mod.parameters(recurse=False)):
            mod.register_forward_pre_hook(cast_inputs_hook, with_kwargs=True)

    # Patch roi_align to match input/boxes dtypes
    _orig_roi_align = torchvision.ops.roi_align

    def _patched_roi_align(input, boxes, *args, **kwargs):
        if isinstance(boxes, (list, tuple)):
            boxes = [b.to(input.dtype) if b.is_floating_point() else b for b in boxes]
        elif isinstance(boxes, torch.Tensor) and boxes.is_floating_point():
            boxes = boxes.to(input.dtype)
        return _orig_roi_align(input, boxes, *args, **kwargs)

    torchvision.ops.roi_align = _patched_roi_align
    return model


def main():
    """Run the SAM3 dora node."""
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    print(f"[dora-sam3] Loading SAM3 model (confidence={CONFIDENCE}, fp16={USE_FP16}) ...")
    model = build_sam3_image_model()
    if USE_FP16:
        model = _enable_fp16(model)
    processor = Sam3Processor(model, confidence_threshold=CONFIDENCE)
    print("[dora-sam3] SAM3 model loaded.")

    pa.array([])  # initialize pyarrow
    node = Node()
    frames = {}
    image_states = {}  # image_id -> processor state (backbone features cached)

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

                if encoding in ("bgr8", "rgb8"):
                    channels = 3
                    frame = (
                        storage.to_numpy()
                        .astype(np.uint8)
                        .reshape((height, width, channels))
                    )
                    if encoding == "bgr8":
                        frame = frame[:, :, ::-1]  # BGR to RGB
                elif encoding in ("jpeg", "jpg", "jpe", "bmp", "webp", "png"):
                    frame = cv2.imdecode(
                        storage.to_numpy(), cv2.IMREAD_COLOR
                    )
                    frame = frame[:, :, ::-1]  # BGR to RGB
                else:
                    raise RuntimeError(f"Unsupported image encoding: {encoding}")

                image = Image.fromarray(frame)
                frames[event_id] = image

                # Pre-compute backbone features for this image
                with torch.inference_mode():
                    image_states[event_id] = processor.set_image(image)
                print(f"[dora-sam3] Image set: {event_id} ({width}x{height})")

            elif "text" in event_id:
                # Text prompt for concept segmentation
                text = event["value"][0].as_py()
                metadata = event["metadata"]
                image_id = metadata.get("image_id", None)

                # Find the image to segment
                if image_id and image_id in image_states:
                    state = image_states[image_id]
                elif len(image_states) > 0:
                    image_id = next(iter(image_states.keys()))
                    state = image_states[image_id]
                else:
                    print("[dora-sam3] No image available, skipping text prompt")
                    continue

                print(f"[dora-sam3] Text prompt: '{text}' on {image_id}")
                with torch.inference_mode():
                    # Reset previous prompts and run text segmentation
                    processor.reset_all_prompts(state)
                    state = processor.set_text_prompt(prompt=text, state=state)

                masks = state["masks"]  # [N, 1, H, W] bool tensor
                scores = state["scores"]  # [N]
                boxes = state["boxes"]  # [N, 4] xyxy

                n_instances = masks.shape[0]
                img_h, img_w = masks.shape[2], masks.shape[3]
                score_list = scores.cpu().tolist()
                print(
                    f"[dora-sam3] Found {n_instances} instances "
                    f"(scores: {score_list})"
                )

                if n_instances == 0:
                    node.send_output("masks", pa.array([]), {})
                    continue

                # Use highest-scoring instance instead of merging all
                best_idx = int(scores.argmax())
                merged = masks[best_idx, 0, :, :].cpu().numpy().astype(np.uint8)
                merged *= 255
                print(f"[dora-sam3] Using instance {best_idx} (score={score_list[best_idx]:.3f})")

                node.send_output(
                    "masks",
                    pa.array(merged.ravel()),
                    metadata={
                        "image_id": image_id,
                        "width": img_w,
                        "height": img_h,
                        "n_instances": n_instances,
                    },
                )

            elif "points" in event_id:
                # Point prompts (same interface as dora-sam2)
                points = event["value"].to_numpy().reshape((-1, 2))
                if len(frames) == 0:
                    continue
                first_image = next(iter(frames.keys()))
                image_id = event["metadata"].get("image_id", first_image)

                if image_id not in image_states:
                    continue

                state = image_states[image_id]
                img = frames[image_id]
                img_w, img_h = img.size

                # Optional text hint from metadata to guide SAM3 segmentation
                text_hint = event["metadata"].get("text", "")
                print(f"[dora-sam3] Point prompt: {len(points)} points on {image_id}" +
                      (f" text='{text_hint}'" if text_hint else ""))
                with torch.inference_mode():
                    processor.reset_all_prompts(state)
                    # Set text prompt if provided — gives SAM3 semantic context
                    if text_hint:
                        state = processor.set_text_prompt(prompt=text_hint, state=state)
                    # Add each point as a positive box (small box around point)
                    for px, py in points:
                        # Normalize to [0,1] and create small box
                        cx = float(px) / img_w
                        cy = float(py) / img_h
                        bw = 0.02  # small box around point
                        bh = 0.02
                        state = processor.add_geometric_prompt(
                            box=[cx, cy, bw, bh], label=True, state=state
                        )

                masks = state.get("masks")
                scores = state.get("scores")
                if masks is None or masks.shape[0] == 0:
                    node.send_output("masks", pa.array([]), {})
                    continue

                n = masks.shape[0]
                if scores is not None and n > 1:
                    best_idx = int(scores.argmax())
                    print(f"[dora-sam3] Point: {n} instances, using best (idx={best_idx}, score={scores[best_idx]:.3f})")
                    merged = masks[best_idx, 0, :, :].cpu().numpy().astype(np.uint8)
                else:
                    merged = masks[:, 0, :, :].any(dim=0).cpu().numpy().astype(np.uint8)
                merged *= 255

                node.send_output(
                    "masks",
                    pa.array(merged.ravel()),
                    metadata={
                        "image_id": image_id,
                        "width": img_w,
                        "height": img_h,
                    },
                )

            elif "boxes" in event_id:
                # Box prompt: [x_min, y_min, x_max, y_max] in pixel coords
                box_data = event["value"].to_numpy().reshape((-1, 4))
                if len(frames) == 0:
                    continue
                first_image = next(iter(frames.keys()))
                image_id = event["metadata"].get("image_id", first_image)
                text_hint = event["metadata"].get("text", "")

                if image_id in image_states:
                    state = image_states[image_id]
                else:
                    if image_id in frames:
                        state = processor.preprocess(frames[image_id])
                        image_states[image_id] = state
                    else:
                        continue
                frame = frames.get(image_id)
                if frame is not None:
                    if hasattr(frame, 'shape'):
                        img_h, img_w = frame.shape[0], frame.shape[1]
                    elif hasattr(frame, 'size'):
                        img_w, img_h = frame.size  # PIL Image
                    else:
                        img_h, img_w = 720, 1280
                else:
                    img_h, img_w = 720, 1280

                print(f"[dora-sam3] Box prompt: {len(box_data)} boxes on {image_id} ({img_w}x{img_h})" +
                      (f" text='{text_hint}'" if text_hint else ""))
                with torch.inference_mode():
                    processor.reset_all_prompts(state)
                    if text_hint:
                        state = processor.set_text_prompt(prompt=text_hint, state=state)
                    for x_min, y_min, x_max, y_max in box_data:
                        # Convert pixel xyxy to normalized cxcywh
                        cx = float((x_min + x_max) / 2.0) / img_w
                        cy = float((y_min + y_max) / 2.0) / img_h
                        bw = float(x_max - x_min) / img_w
                        bh = float(y_max - y_min) / img_h
                        state = processor.add_geometric_prompt(
                            box=[cx, cy, bw, bh], label=True, state=state
                        )

                masks = state.get("masks")
                scores = state.get("scores")
                if masks is None or masks.shape[0] == 0:
                    node.send_output("masks", pa.array([]), {})
                    continue

                n = masks.shape[0]
                if scores is not None and n > 1:
                    best_idx = int(scores.argmax())
                    print(f"[dora-sam3] Box: {n} instances, using best (idx={best_idx}, score={scores[best_idx]:.3f})")
                    merged = masks[best_idx, 0, :, :].cpu().numpy().astype(np.uint8)
                else:
                    merged = masks[:, 0, :, :].any(dim=0).cpu().numpy().astype(np.uint8)
                merged *= 255

                node.send_output(
                    "masks",
                    pa.array(merged.ravel()),
                    metadata={
                        "image_id": image_id,
                        "width": img_w,
                        "height": img_h,
                    },
                )

        elif event_type == "ERROR":
            print("[dora-sam3] Error:" + event["error"])


if __name__ == "__main__":
    main()
