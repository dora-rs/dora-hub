"""TODO: Add docstring."""

import os
import time

import cv2
import numpy as np
import pyarrow as pa
from dora import Node

# Use xoq_realsense (relay) when RELAY_PATH is set, otherwise pyrealsense2 (local USB)
relay_path = os.getenv("RELAY_PATH", "")
if relay_path:
    import xoq_realsense as rs
else:
    import pyrealsense2 as rs

RUNNER_CI = True if os.getenv("CI") == "true" else False


def main():
    """TODO: Add docstring."""
    flip = os.getenv("FLIP", "")
    device_serial = os.getenv("DEVICE_SERIAL", "")
    image_height = int(os.getenv("IMAGE_HEIGHT", "480"))
    image_width = int(os.getenv("IMAGE_WIDTH", "640"))
    encoding = os.getenv("ENCODING", "rgb8")

    if relay_path:
        # xoq_realsense: use relay path as device identifier
        device_id = relay_path
        print(f"[realsense] Using xoq relay: {relay_path}")
    else:
        ctx = rs.context()
        devices = ctx.query_devices()
        if devices.size() == 0:
            raise ConnectionError("No realsense camera connected.")
        serials = [device.get_info(rs.camera_info.serial_number) for device in devices]
        if device_serial and (device_serial not in serials):
            raise ConnectionError(
                f"Device with serial {device_serial} not found within: {serials}.",
            )
        device_id = device_serial

    pipeline = rs.pipeline()

    config = rs.config()
    config.enable_device(device_id)
    config.enable_stream(rs.stream.color, image_width, image_height, rs.format.rgb8, 30)
    config.enable_stream(rs.stream.depth, image_width, image_height, rs.format.z16, 30)

    align_to = rs.stream.color
    align = rs.align(align_to)

    profile = pipeline.start(config)

    # Extract intrinsics — xoq_realsense doesn't support profile.get_stream(),
    # so we defer intrinsics to the first frame in that case.
    rgb_intr = None
    if not relay_path:
        rgb_profile = profile.get_stream(rs.stream.color)
        depth_profile = profile.get_stream(rs.stream.depth)
        _depth_intr = depth_profile.as_video_stream_profile().get_intrinsics()
        rgb_intr = rgb_profile.as_video_stream_profile().get_intrinsics()

    node = Node()
    start_time = time.time()

    pa.array([])  # initialize pyarrow array

    for event in node:
        # Run this example in the CI for 10 seconds only.
        if RUNNER_CI and time.time() - start_time > 10:
            break

        event_type = event["type"]

        if event_type == "INPUT":
            event_id = event["id"]

            if event_id == "tick":
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)

                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                if not aligned_depth_frame or not color_frame:
                    continue

                # Get intrinsics from first frame (xoq_realsense path)
                if rgb_intr is None:
                    vsp = color_frame.profile.as_video_stream_profile()
                    rgb_intr = vsp.get_intrinsics()
                    print(f"[realsense] Intrinsics: fx={rgb_intr.fx:.1f} fy={rgb_intr.fy:.1f} "
                          f"ppx={rgb_intr.ppx:.1f} ppy={rgb_intr.ppy:.1f}")

                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                scaled_depth_image = depth_image
                frame = np.asanyarray(color_frame.get_data())

                ## Change rgb to bgr

                if flip == "VERTICAL":
                    frame = cv2.flip(frame, 0)
                elif flip == "HORIZONTAL":
                    frame = cv2.flip(frame, 1)
                elif flip == "BOTH":
                    frame = cv2.flip(frame, -1)

                metadata = event["metadata"]
                metadata["encoding"] = encoding
                metadata["width"] = int(frame.shape[1])
                metadata["height"] = int(frame.shape[0])

                # Get the right encoding
                if encoding == "bgr8":
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                elif encoding in ["jpeg", "jpg", "jpe", "bmp", "webp", "png"]:
                    ret, frame = cv2.imencode("." + encoding, frame)
                    if not ret:
                        print("Error encoding image...")
                        continue

                storage = pa.array(frame.ravel())

                metadata["resolution"] = [int(rgb_intr.ppx), int(rgb_intr.ppy)]
                metadata["focal_length"] = [int(rgb_intr.fx), int(rgb_intr.fy)]
                metadata["timestamp"] = time.time_ns()
                node.send_output("image", storage, metadata)
                metadata["encoding"] = "mono16"
                scaled_depth_image[scaled_depth_image > 5000] = 0
                node.send_output(
                    "depth",
                    pa.array(scaled_depth_image.ravel()),
                    metadata,
                )

        elif event_type == "ERROR":
            raise RuntimeError(event["error"])


if __name__ == "__main__":
    main()
