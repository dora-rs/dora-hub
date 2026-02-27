"""
Generic ONNX Inference Node for dora-rs.
"""
import argparse
import os
import numpy as np
import pyarrow as pa
from dora import Node
import onnxruntime as ort

def preprocess_image(storage: pa.Array, width: int, height: int, encoding: str) -> np.ndarray:
    """Extracts PyArrow array and converts it to NCHW float32 tensor."""
    if encoding in ["bgr8", "rgb8"]:
        channels = 3
        storage_type = np.uint8
    else:
        raise RuntimeError(f"Unsupported image encoding: {encoding}")
        
    # 1. Zero-copy extraction to NumPy
    frame = (
        storage.to_numpy()
        .astype(storage_type)
        .reshape((height, width, channels))
    )
    
    if encoding == "bgr8":
        frame = frame[:, :, ::-1]  # OpenCV image (BGR to RGB)
        
    # 2. Preprocess for generic ONNX (HWC to NCHW)
    frame_chw = np.transpose(frame, (2, 0, 1))
    input_tensor = np.expand_dims(frame_chw, axis=0).astype(np.float32)
    
    # Normalization (0-1)
    input_tensor = input_tensor / 255.0 
    
    return input_tensor

def main():
    parser = argparse.ArgumentParser(description="ONNX Inference Node")
    parser.add_argument("--name", type=str, default="onnx-inference")
    parser.add_argument("--model", type=str, default="model.onnx")
    args = parser.parse_args()
    
    model_path = os.getenv("MODEL", args.model)
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    print(f"Loading ONNX model from {model_path}...")
    session = ort.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name
    
    node = Node(args.name)
    print(f"ONNX Node '{args.name}' initialized and waiting for inputs...")
    
    for event in node:
        if event["type"] == "INPUT" and event["id"] == "image":
            storage = event["value"]
            metadata = event["metadata"]
            
            # Use the extracted function
            input_tensor = preprocess_image(
                storage=storage, 
                width=metadata["width"], 
                height=metadata["height"], 
                encoding=metadata.get("encoding", "rgb8")
            )
            
            # 3. Perform Inference
            results = session.run(None, {input_name: input_tensor})
            
            # 4. Pack output
            output_array = results[0]
            out_pa = pa.array(output_array.ravel())
            
            out_metadata = event["metadata"]
            out_metadata["shape"] = list(output_array.shape)
            out_metadata["type"] = "onnx_tensor"
            
            node.send_output("tensor", out_pa, out_metadata)
            
        elif event["type"] == "ERROR":
            print(f"Received dora error: {event['error']}")

if __name__ == "__main__":
    main()