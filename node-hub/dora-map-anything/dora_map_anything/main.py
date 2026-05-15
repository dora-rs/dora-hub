"""Dora node for MapAnything 3D reconstruction from RGBD frames."""

import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import torch
from dora import Node
from mapanything.models import MapAnything
from mapanything.utils.geometry import depthmap_to_world_frame
from mapanything.utils.image import preprocess_inputs
from mapanything.utils.viz import predictions_to_glb

NUM_FRAMES = int(os.getenv("NUM_FRAMES", "8"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output")
AS_MESH = os.getenv("AS_MESH", "true").lower() == "true"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[map-anything] Loading MapAnything model on {device}...")
model = MapAnything.from_pretrained("facebook/map-anything").to(device)
print("[map-anything] Model loaded.")


def main():
    """Accumulate RGBD frames from RealSense and run MapAnything 3D reconstruction."""
    node = Node()
    views = deque(maxlen=NUM_FRAMES)
    latest_image = None
    latest_metadata = None

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    for event in node:
        if event["type"] == "INPUT":
            event_id = event["id"]

            if "image" in event_id:
                storage = event["value"]
                metadata = event["metadata"]
                encoding = metadata["encoding"]
                width = metadata["width"]
                height = metadata["height"]

                if encoding in ("bgr8", "rgb8"):
                    frame = (
                        storage.to_numpy()
                        .astype(np.uint8)
                        .reshape((height, width, 3))
                    )
                    if encoding == "bgr8":
                        frame = frame[:, :, ::-1]
                elif encoding in ("jpeg", "jpg", "jpe", "bmp", "webp", "png"):
                    frame = cv2.imdecode(storage.to_numpy(), cv2.IMREAD_COLOR)
                    frame = frame[:, :, ::-1]  # BGR to RGB
                else:
                    print(f"[map-anything] Unsupported encoding: {encoding}")
                    continue

                latest_image = frame
                latest_metadata = metadata

            elif "depth" in event_id:
                if latest_image is None:
                    continue

                storage = event["value"]
                metadata = event["metadata"]
                width = metadata["width"]
                height = metadata["height"]

                # RealSense sends mono16 depth in millimeters
                depth_mm = (
                    storage.to_numpy()
                    .astype(np.uint16)
                    .reshape((height, width))
                )
                depth_m = depth_mm.astype(np.float32) / 1000.0

                # Build intrinsics from RealSense metadata
                fx, fy = latest_metadata["focal_length"]
                ppx, ppy = latest_metadata["resolution"]
                intrinsics = np.array(
                    [[fx, 0, ppx], [0, fy, ppy], [0, 0, 1]],
                    dtype=np.float32,
                )

                view = {
                    "img": latest_image.copy(),
                    "depth_z": depth_m,
                    "intrinsics": intrinsics,
                }
                views.append(view)
                latest_image = None
                latest_metadata = None
                print(
                    f"[map-anything] Accumulated {len(views)}/{NUM_FRAMES} frames"
                )

                if len(views) == NUM_FRAMES:
                    _run_reconstruction(node, list(views))
                    views.clear()

            elif "trigger" in event_id:
                if len(views) > 1:
                    print(
                        f"[map-anything] Manual trigger with {len(views)} frames"
                    )
                    _run_reconstruction(node, list(views))
                    views.clear()
                else:
                    print("[map-anything] Need at least 2 frames to reconstruct")

        elif event["type"] == "ERROR":
            raise RuntimeError(event["error"])


def _run_reconstruction(node, views):
    """Run MapAnything inference and export GLB."""
    print(f"[map-anything] Running reconstruction with {len(views)} views...")
    t0 = time.time()

    views = preprocess_inputs(views)

    outputs = model.infer(
        views,
        memory_efficient_inference=True,
        minibatch_size=1,
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=True,
        mask_edges=True,
    )

    world_points_list = []
    images_list = []
    masks_list = []

    for pred in outputs:
        depthmap = pred["depth_z"][0].squeeze(-1)
        intrinsics = pred["intrinsics"][0]
        camera_pose = pred["camera_poses"][0]

        pts3d, valid_mask = depthmap_to_world_frame(
            depthmap, intrinsics, camera_pose
        )

        mask = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)
        mask = mask & valid_mask.cpu().numpy()

        world_points_list.append(pts3d.cpu().numpy())
        images_list.append(pred["img_no_norm"][0].cpu().numpy())
        masks_list.append(mask)

    predictions = {
        "world_points": np.stack(world_points_list, axis=0),
        "images": np.stack(images_list, axis=0),
        "final_masks": np.stack(masks_list, axis=0),
    }

    scene_3d = predictions_to_glb(predictions, as_mesh=AS_MESH)

    timestamp = int(time.time())
    glb_filename = f"reconstruction_{timestamp}.glb"
    glb_path = str(Path(OUTPUT_DIR) / glb_filename)
    scene_3d.export(glb_path)

    elapsed = time.time() - t0
    print(f"[map-anything] Saved {glb_path} in {elapsed:.1f}s")

    node.send_output("glb_path", pa.array([glb_path]))


if __name__ == "__main__":
    main()
