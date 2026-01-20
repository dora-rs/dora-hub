import cv2
import time
import pyarrow as pa
import numpy as np
from dora import Node

node = Node()

# --- THE HANDSHAKE (Async Wait) ---
print("Waiting for Recorder signal...", flush=True)

for event in node:
    # We listen to events until we get the specific handshake
    if event["type"] == "INPUT" and event["id"] == "recorder_status":
        print(f"Received signal: {event['value'][0].as_py()}")
        break # Exit the wait loop and start the camera!
# ----------------------------------

cap = cv2.VideoCapture(0)
frame_limit = 50 
count = 0

print(" Signal received! Starting Camera Stream...", flush=True)

try:
    while count < frame_limit and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(" Camera read failed (Frame dropped)", flush=True)
            break
            
        # Resize to keep data manageable
        frame = cv2.resize(frame, (640, 480))
        
        node.send_output(
            "image", 
            pa.array(frame.ravel()), 
            metadata={
                "width": 640, 
                "height": 480, 
                "encoding": "bgr8",
                "frame_id": count
            }
        )
        
        # 2. Print EVERY frame so we know if it was actually sent
        print(f"Sent Frame {count}", flush=True)
        
        count += 1
        time.sleep(0.05) # Go slightly slower to be safe (20 FPS)

except Exception as e:
    print(f"Error: {e}")
finally:
    cap.release()
    # 3. Wait before quitting to ensure the last message gets out
    time.sleep(2)
    print(" Camera Node Finished", flush=True)