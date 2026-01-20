import pandas as pd
import numpy as np
import cv2
import json
import os

output_folder = "recovered_frames"
os.makedirs(output_folder, exist_ok=True)
file_path = "experiment_logs/cam_feed.parquet"

df = pd.read_parquet(file_path)
print(f"Found {len(df)} frames.")

for i in range(len(df)):
    try:
        raw_bytes = df.iloc[i]["data"] # This is now bytes, not a list
        raw_meta = df.iloc[i]["metadata"]
        
        meta_dict = json.loads(raw_meta)
        w, h = meta_dict.get("width", 640), meta_dict.get("height", 480)
        
        # FAST DECODE: buffer -> numpy
        image = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((h, w, 3))
        
        cv2.imwrite(f"{output_folder}/frame_{i}.jpg", image)
        
    except Exception as e:
        print(f"Skipping frame {i}: {e}")

print("Done!")