"""Test node: receives camera image + VLM response, draws grasp points, saves image."""

import json
import os
import time

import cv2
import numpy as np
import pyarrow as pa
from dora import Node

WIDTH = int(os.getenv("IMAGE_WIDTH", "640"))
HEIGHT = int(os.getenv("IMAGE_HEIGHT", "480"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", ".")
NORMALIZED_COORDS = os.getenv("NORMALIZED_COORDS", "true").lower() in ("1", "true", "yes")

node = Node()
latest_image = None
frame_count = 0


def strip_think_tags(text):
    """Strip <think>...</think> blocks from thinking model output."""
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_vlm_response(text):
    text = strip_think_tags(text).strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        data = json.loads(text[start:end])
        p1 = data["p1"]
        p2 = data["p2"]
        if len(p1) == 2 and len(p2) == 2:
            return (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))
    except (json.JSONDecodeError, KeyError, TypeError, IndexError):
        pass
    return None


for event in node:
    if event["type"] == "INPUT":
        event_id = event["id"]

        if event_id == "image":
            latest_image = event["value"].to_numpy().tobytes()

        elif event_id == "vlm_response":
            text = event["value"][0].as_py()
            print(f"VLM raw: {text}")

            if latest_image is None:
                print("No image to draw on")
                continue

            img = np.frombuffer(latest_image, dtype=np.uint8).reshape(
                (HEIGHT, WIDTH, 3)
            )
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            points = parse_vlm_response(text)
            if points is None:
                print(f"Could not parse points from: {text}")
                # Save raw image anyway
                path = os.path.join(OUTPUT_DIR, f"grasp_test_{frame_count:03d}_raw.png")
                cv2.imwrite(path, img_bgr)
                print(f"Saved raw image: {path}")
            else:
                p1, p2 = points
                # Rescale from 0-1000 normalized coords to pixel coords
                if NORMALIZED_COORDS:
                    p1 = (int(p1[0] * WIDTH / 1000), int(p1[1] * HEIGHT / 1000))
                    p2 = (int(p2[0] * WIDTH / 1000), int(p2[1] * HEIGHT / 1000))
                mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                print(f"p1={p1}, p2={p2}, midpoint={mid} (normalized={NORMALIZED_COORDS})")

                # Draw grasp visualization
                # Green circles for finger positions
                cv2.circle(img_bgr, p1, 8, (0, 255, 0), 2)
                cv2.circle(img_bgr, p2, 8, (0, 255, 0), 2)
                # Blue line between fingers (gripper opening direction)
                cv2.line(img_bgr, p1, p2, (255, 100, 0), 2)
                # Red circle at grasp center
                cv2.circle(img_bgr, mid, 6, (0, 0, 255), -1)
                # Labels
                cv2.putText(img_bgr, "p1", (p1[0] + 10, p1[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(img_bgr, "p2", (p2[0] + 10, p2[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(img_bgr, f"VLM: {text.strip()}", (10, HEIGHT - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                path = os.path.join(OUTPUT_DIR, f"grasp_test_{frame_count:03d}.png")
                cv2.imwrite(path, img_bgr)
                print(f"Saved annotated image: {path}")

            frame_count += 1

    elif event["type"] == "STOP":
        break
