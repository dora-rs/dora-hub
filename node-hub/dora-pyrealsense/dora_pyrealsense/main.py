"""TODO: Add docstring."""

import os
import time

import cv2
import numpy as np
import pyarrow as pa
from dora import Node

# Use xoq_realsense (relay) when RELAY_PATH is set, otherwise pyrealsense2 (local USB)
relay_path = os.getenv("RELAY_PATH", "")
if not relay_path:
    import pyrealsense2 as rs

RUNNER_CI = True if os.getenv("CI") == "true" else False


def _relay_worker(relay_path, image_width, image_height, frame_queue, stop_event):
    """Run xoq_realsense in a separate process to avoid tokio runtime conflicts with dora."""
    import multiprocessing.shared_memory as shm
    import xoq_realsense as rs2

    pipeline = rs2.pipeline()
    config = rs2.config()
    config.enable_device(relay_path)
    config.enable_stream(rs2.stream.color, image_width, image_height, rs2.format.rgb8, 30)
    config.enable_stream(rs2.stream.depth, image_width, image_height, rs2.format.z16, 30)
    profile = pipeline.start(config)
    align = rs2.align(rs2.stream.color)

    rgb_size = image_width * image_height * 3
    depth_size = image_width * image_height * 2  # uint16
    color_shm = shm.SharedMemory(create=True, size=rgb_size)
    depth_shm = shm.SharedMemory(create=True, size=depth_size)

    # Send shm names to parent
    frame_queue.put(("shm_names", color_shm.name, depth_shm.name))

    intrinsics_sent = False
    try:
        while not stop_event.is_set():
            try:
                frames = pipeline.wait_for_frames()
            except Exception as e:
                print(f"[realsense-worker] wait_for_frames error: {e}", flush=True)
                time.sleep(0.1)
                continue
            aligned = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            if not intrinsics_sent:
                vsp = color_frame.profile.as_video_stream_profile()
                intr = vsp.get_intrinsics()
                frame_queue.put(("intrinsics", intr.fx, intr.fy, intr.ppx, intr.ppy))
                intrinsics_sent = True

            # Copy into shared memory
            color_data = np.array(color_frame.get_data())
            depth_data = np.array(depth_frame.get_data())
            color_buf = np.ndarray(color_data.shape, dtype=color_data.dtype, buffer=color_shm.buf)
            depth_buf = np.ndarray(depth_data.shape, dtype=depth_data.dtype, buffer=depth_shm.buf)
            np.copyto(color_buf, color_data)
            np.copyto(depth_buf, depth_data)

            # Signal frame ready
            try:
                frame_queue.put(("frame",), block=False)
            except Exception:
                pass  # Queue full, skip frame
    finally:
        pipeline.stop()
        color_shm.close()
        depth_shm.close()
        color_shm.unlink()
        depth_shm.unlink()
        print("[realsense-worker] stopped", flush=True)


def main():
    """TODO: Add docstring."""
    flip = os.getenv("FLIP", "")
    device_serial = os.getenv("DEVICE_SERIAL", "")
    image_height = int(os.getenv("IMAGE_HEIGHT", "480"))
    image_width = int(os.getenv("IMAGE_WIDTH", "640"))
    encoding = os.getenv("ENCODING", "rgb8")

    # Optional intrinsics override (xoq_realsense may return placeholders)
    intr_override = os.getenv("CAMERA_INTRINSICS", "")  # "fx,fy,ppx,ppy"

    if relay_path:
        import multiprocessing as mp
        import multiprocessing.shared_memory as shm

        print(f"[realsense] Using xoq relay: {relay_path}")

        frame_queue = mp.Queue(maxsize=2)
        stop_event = mp.Event()
        worker = mp.Process(
            target=_relay_worker,
            args=(relay_path, image_width, image_height, frame_queue, stop_event),
            daemon=True,
        )
        worker.start()

        # Wait for shared memory names
        msg = frame_queue.get(timeout=30)
        assert msg[0] == "shm_names"
        color_shm = shm.SharedMemory(name=msg[1])
        depth_shm = shm.SharedMemory(name=msg[2])

        rgb_intr = None
        node = Node()
        start_time = time.time()
        pa.array([])  # initialize pyarrow

        try:
            for event in node:
                if RUNNER_CI and time.time() - start_time > 10:
                    break

                event_type = event["type"]
                if event_type == "INPUT":
                    event_id = event["id"]
                    if event_id == "tick":
                        # Drain queue to get latest frame + check for intrinsics
                        got_frame = False
                        while not frame_queue.empty():
                            msg = frame_queue.get_nowait()
                            if msg[0] == "intrinsics":
                                fx, fy, ppx, ppy = msg[1], msg[2], msg[3], msg[4]
                                if intr_override:
                                    parts = [float(x) for x in intr_override.split(",")]
                                    fx, fy, ppx, ppy = parts[0], parts[1], parts[2], parts[3]
                                    print(f"[realsense] Using override intrinsics")

                                class Intr:
                                    pass

                                rgb_intr = Intr()
                                rgb_intr.fx = fx
                                rgb_intr.fy = fy
                                rgb_intr.ppx = ppx
                                rgb_intr.ppy = ppy
                                print(
                                    f"[realsense] Intrinsics: fx={fx:.1f} fy={fy:.1f} "
                                    f"ppx={ppx:.1f} ppy={ppy:.1f}"
                                )
                            elif msg[0] == "frame":
                                got_frame = True

                        if not got_frame or rgb_intr is None:
                            continue

                        # Read from shared memory
                        frame = np.ndarray(
                            (image_height, image_width, 3),
                            dtype=np.uint8,
                            buffer=color_shm.buf,
                        ).copy()
                        depth_image = np.ndarray(
                            (image_height, image_width),
                            dtype=np.uint16,
                            buffer=depth_shm.buf,
                        ).copy()

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
                        depth_image[depth_image > 5000] = 0
                        node.send_output(
                            "depth",
                            pa.array(depth_image.ravel()),
                            metadata,
                        )

                elif event_type == "ERROR":
                    raise RuntimeError(event["error"])
        finally:
            stop_event.set()
            worker.join(timeout=5)
            color_shm.close()
            depth_shm.close()
            print("[realsense] Pipeline stopped")

    else:
        # Local USB camera path (unchanged)
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
        rgb_profile = profile.get_stream(rs.stream.color)
        depth_profile = profile.get_stream(rs.stream.depth)
        _depth_intr = depth_profile.as_video_stream_profile().get_intrinsics()
        rgb_intr = rgb_profile.as_video_stream_profile().get_intrinsics()

        node = Node()
        start_time = time.time()
        pa.array([])

        for event in node:
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

                    depth_image = np.asanyarray(aligned_depth_frame.get_data())
                    scaled_depth_image = depth_image
                    frame = np.asanyarray(color_frame.get_data())

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
