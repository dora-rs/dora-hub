"""Simple formatter that takes camera images and sends them with a prompt to dora-qwen-omni."""

import base64
import os

import numpy as np
import pyarrow as pa
from dora import Node

node = Node()

prompt = os.getenv(
    "DEFAULT_QUESTION", "Describe what you see in one short sentence."
)

for event in node:
    if event["type"] == "INPUT" and event["id"] == "image":
        image = event["value"].to_numpy().tobytes()
        # Encode raw image bytes as base64 PNG
        # The image from opencv-video-capture is raw BGR, we need to encode it
        import cv2

        width = int(os.getenv("IMAGE_WIDTH", "640"))
        height = int(os.getenv("IMAGE_HEIGHT", "480"))
        img = np.frombuffer(image, dtype=np.uint8).reshape((height, width, 3))
        _, png_bytes = cv2.imencode(".png", img)
        b64 = base64.b64encode(png_bytes.tobytes()).decode("utf-8")
        image_url = f"data:image/png;base64,{b64}"

        texts = [
            f"<|user|>\n<|vision_start|>\n{image_url}",
            f"<|user|>\n<|im_start|>\n{prompt}",
        ]
        node.send_output("text", pa.array(texts))
        print(f"Sent image + prompt to VLM")
    elif event["type"] == "INPUT" and event["id"] == "tick":
        # Timer tick without image - skip
        pass
