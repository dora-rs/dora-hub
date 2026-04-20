"""Sends a static PNG file as image events for offline testing without a camera.

Uses DEFAULT_DEPTH_MM fallback (no real depth data).
Sends one image event per tick, with fake intrinsics in metadata.
"""

import os
import sys

import cv2
import numpy as np
import pyarrow as pa
from dora import Node

IMAGE_PATH = os.getenv("IMAGE_PATH", "")
WIDTH = int(os.getenv("IMAGE_WIDTH", "640"))
HEIGHT = int(os.getenv("IMAGE_HEIGHT", "480"))

if not IMAGE_PATH or not os.path.isfile(IMAGE_PATH):
    print(f"ERROR: IMAGE_PATH not set or file not found: '{IMAGE_PATH}'")
    sys.exit(1)

# Load and resize image
img_bgr = cv2.imread(IMAGE_PATH)
if img_bgr is None:
    print(f"ERROR: Could not read image: {IMAGE_PATH}")
    sys.exit(1)

img_bgr = cv2.resize(img_bgr, (WIDTH, HEIGHT))
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
flat = img_rgb.ravel()
print(f"Loaded static image: {IMAGE_PATH} ({WIDTH}x{HEIGHT})")

node = Node()

for event in node:
    if event["type"] == "INPUT":
        if event["id"] == "tick":
            node.send_output(
                "image",
                pa.array(flat),
                metadata={
                    "encoding": "rgb8",
                    "width": WIDTH,
                    "height": HEIGHT,
                    "focal_length": [600, 600],
                    "resolution": [WIDTH // 2, HEIGHT // 2],
                },
            )
            print("Sent static image")
    elif event["type"] == "STOP":
        break
