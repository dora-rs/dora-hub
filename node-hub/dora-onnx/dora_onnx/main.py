"""
Generic ONNX Inference Node for dora-rs.
This node loads an exported .onnx model, accepts image tensors from the dataflow,
performs zero-copy inference, and outputs the raw result tensors.
"""

import argparse
import os

import numpy as np
import pyarrow as pa
from dora import Node
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser(
        description="ONNX Inference Node: Perform high-performance inference using generic ONNX models.",
    )

    parser.add_argument(
        "--name",
        type=str,
        required=False,
        help="The name of the node in the dataflow.",
        default="onnx-inference",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        help="The name of the .onnx model file.",
        default="model.onnx",
    )

    args = parser.parse_args()

    model_path = os.getenv("MODEL", args.model)
    
    # Initialize ONNX Runtime session (tries GPU first, falls back to CPU)
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    print(f"Loading ONNX model from {model_path}...")
    session = ort.InferenceSession(model_path, providers=providers)
    
    # Dynamically get input name expected by the model
    input_name = session.get_inputs()[0].name

    node = Node(args.name)
    print(f"ONNX Node '{args.name}' initialized and waiting for inputs...")

    for event in node:
        event_type = event["type"]

        if event_type == "INPUT":
            event_id = event["id"]

            if event_id == "image":
                storage = event["value"]
                metadata = event["metadata"]
                encoding = metadata.get("encoding", "rgb8")
                width = metadata["width"]
                height = metadata["height"]

                # 1. Zero-copy extraction from PyArrow to NumPy
                if encoding in ["bgr8", "rgb8"]:
                    channels = 3
                    storage_type = np.uint8
                else:
                    raise RuntimeError(f"Unsupported image encoding: {encoding}")

                frame = (
                    storage.to_numpy()
                    .astype(storage_type)
                    .reshape((height, width, channels))
                )
                
                if encoding == "bgr8":
                    frame = frame[:, :, ::-1]  # OpenCV image (BGR to RGB)

                # 2. Preprocess for generic ONNX (HWC to NCHW, uint8 to float32)
                # Most computer vision ONNX models expect shape [batch, channels, height, width]
                frame_chw = np.transpose(frame, (2, 0, 1))
                input_tensor = np.expand_dims(frame_chw, axis=0).astype(np.float32)
                
                # Optional standard normalization (0-1) - typically required by AI models
                input_tensor = input_tensor / 255.0 

                # 3. Perform Inference
                results = session.run(None, {input_name: input_tensor})

                # 4. Pack the raw output tensor back into PyArrow
                output_array = results[0]
                
                # Flatten the array for easy PyArrow transmission, but keep original shape in metadata
                out_pa = pa.array(output_array.ravel())
                
                out_metadata = event["metadata"]
                out_metadata["shape"] = list(output_array.shape)
                out_metadata["type"] = "onnx_tensor"

                # Send output back to the dataflow
                node.send_output(
                    "tensor",
                    out_pa,
                    out_metadata,
                )

        elif event_type == "ERROR":
            print(f"Received dora error: {event['error']}")

if __name__ == "__main__":
    main()