import os
import time
import cv2
import numpy as np
from PIL import Image
from dora import Node
import google.generativeai as genai

def main():
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not found!")
        return
        
    genai.configure(api_key=api_key)
    # Using 1.5-flash (or 2.5-flash if your account defaults to it)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    node = Node()
    last_processed_time = 0

    for event in node:
        if event["type"] == "INPUT" and event["id"] == "image":
            current_time = time.time()
            
            # Throttle to 12 seconds to respect Google's Free Tier limits
            if current_time - last_processed_time > 12.0:                
                # 1. Grab the raw memory buffer
                raw_buffer = event["value"].buffers()[1]
                
                # 2. Reconstruct and COPY the array
                # Using .copy() is CRITICAL. It moves the data to local Python RAM.
                frame = np.frombuffer(raw_buffer, dtype=np.uint8).reshape((480, 640, 3)).copy()
                
                # 3. Explicitly delete the Arrow reference to instantly free Dora's memory lock!
                del raw_buffer
                del event
                
                # 4. Format for the AI
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                
                prompt = "You are a robot's vision system. Describe what you see in one short sentence. Then, on a new line, say exactly 'STATUS: CLEAR' if there are no immediate obstacles in front of the camera, or 'STATUS: BLOCKED' if there is an object right in front."
                
                try:
                    response = model.generate_content([prompt, pil_img])
                    print("--------------------------------------------------")
                    print(f"🤖 VLM DECISION:\n{response.text.strip()}")
                    print("--------------------------------------------------")
                except Exception as e:
                    print(f"⚠️ API Error: {e}")
                    
                last_processed_time = current_time

if __name__ == "__main__":
    main()