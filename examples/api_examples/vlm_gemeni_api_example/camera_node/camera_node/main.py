import cv2
import numpy as np
import pyarrow as pa
from dora import Node

def main():
    node = Node()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    for event in node:
        if event["type"] == "INPUT" and event["id"] == "tick":
            ret, frame = cap.read()
            if ret:
                # Shrink to 640x480 for faster AI processing
                frame = cv2.resize(frame, (640, 480))
                
                # Zero-Copy Magic: Send the raw BGR byte array instantly
                node.send_output("image", pa.array(frame.ravel()))

    cap.release()

if __name__ == "__main__":
    main()