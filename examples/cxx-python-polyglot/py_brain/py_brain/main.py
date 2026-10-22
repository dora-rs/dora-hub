import numpy as np
from dora import Node

def main():
    print("🧠 Python AI Brain Booting Up...")
    node = Node()
    
    for event in node:
        if event["type"] == "INPUT" and event["id"] == "vibration_data":
            # 1. Grab the raw memory buffer from the Arrow array
            # event["value"] is the Arrow array. 
            # We access the raw C-contiguous buffer directly to avoid copying.
            arrow_array = event["value"]
            raw_buffer = arrow_array.buffers()[1]
            
            # 2. The Zero-Copy Decode
            # Instantly map those C++ bytes back into a high-speed Numpy array
            sensor_array = np.frombuffer(raw_buffer, dtype=np.float32)
            
            # 3. Edge AI Inference (Anomaly Detection)
            # Normal vibration is a sine wave (-1.0 to 1.0). 
            # If the max value spikes, we have a mechanical failure!
            max_vibration = np.max(sensor_array)
            
            if max_vibration > 5.0:
                print(f"🚨 AI ALERT: Mechanical anomaly detected! Spike magnitude: {max_vibration:.2f}")
            else:
                print(f"✅ Status Normal. Max vibration: {max_vibration:.2f}")

if __name__ == "__main__":
    main()