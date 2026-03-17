"""Dora node for TensorFlow Lite inference on edge/constrained devices."""

import argparse
import os

import numpy as np
import pyarrow as pa
from dora import Node


def main():
    """Run the dora-tflite inference node."""
    parser = argparse.ArgumentParser(
        description="dora-tflite: TensorFlow Lite inference node for edge deployment.",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=False,
        help="The name of the node in the dataflow.",
        default="dora-tflite",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        help="Path to the .tflite model file.",
        default="model.tflite",
    )
    args = parser.parse_args()

    model_path = os.getenv("MODEL", args.model)

    try:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path=model_path)
    except ImportError:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=model_path)

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    node = Node(args.name)
    pa.array([])

    for event in node:
        event_type = event["type"]

        if event_type == "INPUT":
            event_id = event["id"]

            if event_id == "tensor":
                storage = event["value"]
                input_data = storage.to_numpy().astype(input_details[0]["dtype"])
                input_shape = input_details[0]["shape"]
                input_data = input_data.reshape(input_shape)

                interpreter.set_tensor(input_details[0]["index"], input_data)
                interpreter.invoke()

                output_data = interpreter.get_tensor(output_details[0]["index"])
                result = pa.array(output_data.ravel().tolist())

                node.send_output("inference", result, event["metadata"])

        elif event_type == "ERROR":
            print(f"Received dora error: {event['error']}")


if __name__ == "__main__":
    main()
