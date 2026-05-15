"""Send a fixed grasp_result to the motion planner after receiving depth data.

Waits for a few depth frames so the motion planner has intrinsics,
then sends a grasp_result JSON with the target grasp pose.
"""

import json
import os

import pyarrow as pa
from dora import Node

TARGET_GRASP = os.getenv("TARGET_GRASP", '{"p1": [469, 486], "p2": [547, 486]}')
# Number of depth frames to wait before sending the grasp
WARMUP_FRAMES = int(os.getenv("WARMUP_FRAMES", "5"))


def main():
    node = Node()
    depth_count = 0
    sent = False

    print(f"[grasp-trigger] Waiting for {WARMUP_FRAMES} depth frames before sending grasp")
    print(f"[grasp-trigger] Target: {TARGET_GRASP}")

    for event in node:
        if event["type"] == "INPUT":
            if event["id"] == "depth":
                depth_count += 1
                w = event["metadata"].get("width", "?")
                h = event["metadata"].get("height", "?")
                has_intrinsics = "focal_length" in event["metadata"]
                print(
                    f"[grasp-trigger] depth frame {depth_count} "
                    f"({w}x{h}, intrinsics={'yes' if has_intrinsics else 'no'})"
                )

                if depth_count >= WARMUP_FRAMES and not sent and has_intrinsics:
                    print(f"[grasp-trigger] Sending grasp_result: {TARGET_GRASP}")
                    node.send_output(
                        "grasp_result",
                        pa.array([TARGET_GRASP]),
                        metadata={},
                    )
                    sent = True
                    print("[grasp-trigger] Grasp result sent! Waiting for trajectory...")

        elif event["type"] == "STOP":
            break


if __name__ == "__main__":
    main()
