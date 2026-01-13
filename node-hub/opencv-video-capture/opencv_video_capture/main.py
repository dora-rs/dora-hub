"""TODO: Add docstring."""

import argparse
import os
import platform
import subprocess
import time

import cv2
import numpy as np
import pyarrow as pa
from dora import Node


def get_macos_cameras() -> list[dict]:
    """Get camera info from macOS system_profiler.

    Returns:
        List of dicts with 'name', 'model_id', and 'unique_id' keys
    """
    cameras = []
    if platform.system() != "Darwin":
        return cameras

    try:
        result = subprocess.run(
            ["system_profiler", "SPCameraDataType"],
            capture_output=True,
            text=True,
        )
        current_camera = {}
        for line in result.stdout.split("\n"):
            line = line.strip()
            # Camera names appear as lines ending with ":" at low indent
            if line.endswith(":") and not line.startswith(
                ("Camera", "Model ID", "Unique ID")
            ):
                if current_camera:
                    cameras.append(current_camera)
                current_camera = {"name": line[:-1]}
            elif line.startswith("Model ID:"):
                current_camera["model_id"] = line.split(":", 1)[1].strip()
            elif line.startswith("Unique ID:"):
                current_camera["unique_id"] = line.split(":", 1)[1].strip()
        if current_camera:
            cameras.append(current_camera)
    except Exception:
        pass

    return cameras


def get_windows_cameras() -> list[dict]:
    """Get camera info from Windows using PowerShell.

    Returns:
        List of dicts with 'name' and 'device_id' keys
    """
    cameras = []
    if platform.system() != "Windows":
        return cameras

    try:
        # Query video capture devices via PowerShell
        ps_command = """
        Get-PnpDevice -Class Camera -Status OK | Select-Object FriendlyName, InstanceId | ConvertTo-Json
        """
        result = subprocess.run(
            ["powershell", "-Command", ps_command],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            import json

            data = json.loads(result.stdout)
            # Handle single device (dict) or multiple devices (list)
            if isinstance(data, dict):
                data = [data]
            for item in data:
                cameras.append({
                    "name": item.get("FriendlyName", "Unknown"),
                    "device_id": item.get("InstanceId", ""),
                })
    except Exception:
        pass

    return cameras


def find_camera_by_id(unique_id: str) -> int | None:
    """Find camera index by unique ID.

    Args:
        unique_id: The unique ID of the camera:
                   - macOS: from 'system_profiler SPCameraDataType'
                   - Linux: from /dev/v4l/by-id/
                   - Windows: from 'Get-PnpDevice -Class Camera' (InstanceId)

    Returns:
        Camera index if found, None otherwise
    """
    if platform.system() == "Darwin":
        cameras = get_macos_cameras()
        for idx, cam in enumerate(cameras):
            if cam.get("unique_id", "").lower() == unique_id.lower():
                return idx

    elif platform.system() == "Linux":
        # On Linux, the unique_id can be the full path or part of the by-id name
        by_id_path = "/dev/v4l/by-id/"
        if os.path.exists(by_id_path):
            for entry in sorted(os.listdir(by_id_path)):
                if unique_id.lower() in entry.lower():
                    real_path = os.path.realpath(os.path.join(by_id_path, entry))
                    if "video" in real_path:
                        return int(real_path.replace("/dev/video", ""))

    elif platform.system() == "Windows":
        cameras = get_windows_cameras()
        for idx, cam in enumerate(cameras):
            if unique_id.lower() in cam.get("device_id", "").lower():
                return idx

    return None


RUNNER_CI = True if os.getenv("CI") == "true" else False

FLIP = os.getenv("FLIP", "")


def main():
    # Handle dynamic nodes, ask for the name of the node in the dataflow, and the same values as the ENV variables.
    """TODO: Add docstring."""
    parser = argparse.ArgumentParser(
        description="OpenCV Video Capture: This node is used to capture video from a camera.",
    )

    parser.add_argument(
        "--name",
        type=str,
        required=False,
        help="The name of the node in the dataflow.",
        default="opencv-video-capture",
    )
    parser.add_argument(
        "--path",
        type=int,
        required=False,
        help="The path of the device to capture (e.g. /dev/video1, or an index like 0, 1...",
        default=0,
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        required=False,
        help="Unique camera ID. macOS: 'system_profiler SPCameraDataType', Linux: /dev/v4l/by-id/, Windows: 'Get-PnpDevice -Class Camera'.",
        default=None,
    )
    parser.add_argument(
        "--image-width",
        type=int,
        required=False,
        help="The width of the image output. Default is the camera width.",
        default=None,
    )
    parser.add_argument(
        "--image-height",
        type=int,
        required=False,
        help="The height of the camera. Default is the camera height.",
        default=None,
    )

    args = parser.parse_args()

    # Check for camera ID first (most reliable), then path/index
    camera_id = os.getenv("CAMERA_ID", args.camera_id)

    if camera_id:
        video_capture_path = find_camera_by_id(camera_id)
        if video_capture_path is None:
            if platform.system() == "Darwin":
                hint = "Run 'system_profiler SPCameraDataType' to list available cameras."
            elif platform.system() == "Windows":
                hint = "Run 'Get-PnpDevice -Class Camera' in PowerShell to list available cameras."
            else:
                hint = "Check /dev/v4l/by-id/ for available camera IDs."
            raise RuntimeError(f"Could not find camera with ID '{camera_id}'. {hint}")
    else:
        video_capture_path = os.getenv("CAPTURE_PATH", args.path)
        if isinstance(video_capture_path, str) and video_capture_path.isnumeric():
            video_capture_path = int(video_capture_path)

    encoding = os.getenv("ENCODING", "bgr8")

    video_capture = cv2.VideoCapture(video_capture_path)

    # Print camera info for debugging
    if video_capture.isOpened():
        if platform.system() == "Darwin":
            cameras = get_macos_cameras()
        elif platform.system() == "Windows":
            cameras = get_windows_cameras()
        else:
            cameras = []

        if isinstance(video_capture_path, int) and video_capture_path < len(cameras):
            cam_info = cameras[video_capture_path]
            print(
                f"Opened camera at index {video_capture_path}: {cam_info.get('name', 'Unknown')}"
            )
            # Print the appropriate ID field per platform
            cam_id = cam_info.get("unique_id") or cam_info.get("device_id", "N/A")
            print(f"  Unique ID: {cam_id}")
        else:
            print(f"Opened camera at index {video_capture_path}")

    image_width = os.getenv("IMAGE_WIDTH", args.image_width)

    if image_width is not None:
        if isinstance(image_width, str) and image_width.isnumeric():
            image_width = int(image_width)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, image_width)

    image_height = os.getenv("IMAGE_HEIGHT", args.image_height)
    if image_height is not None:
        if isinstance(image_height, str) and image_height.isnumeric():
            image_height = int(image_height)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, image_height)

    node = Node(args.name)
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
                ret, frame = video_capture.read()

                if not ret:
                    if not RUNNER_CI:
                        raise RuntimeError(
                            f"Error: cannot read frame from camera at path {video_capture_path}. For resiliency you can use: restart_policy: on-failure in the node definition.",
                        )
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        frame,
                        f"Error: no frame for camera at path {video_capture_path}.",
                        (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.50,
                        (255, 255, 255),
                        1,
                        1,
                    )

                if FLIP == "VERTICAL":
                    frame = cv2.flip(frame, 0)
                elif FLIP == "HORIZONTAL":
                    frame = cv2.flip(frame, 1)
                elif FLIP == "BOTH":
                    frame = cv2.flip(frame, -1)

                # resize the frame
                if (
                    image_width is not None
                    and image_height is not None
                    and (
                        frame.shape[1] != image_width or frame.shape[0] != image_height
                    )
                ):
                    frame = cv2.resize(frame, (image_width, image_height))

                metadata = event["metadata"]
                metadata["encoding"] = encoding
                metadata["width"] = int(frame.shape[1])
                metadata["height"] = int(frame.shape[0])
                metadata["primitive"] = "image"

                # Get the right encoding
                if encoding == "rgb8":
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                elif encoding == "yuv420":
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
                elif encoding in ["jpeg", "jpg", "jpe", "bmp", "webp", "png"]:
                    ret, frame = cv2.imencode("." + encoding, frame)
                    if not ret:
                        print("Error encoding image...")
                        continue

                storage = pa.array(frame.ravel())

                node.send_output("image", storage, metadata)

        elif event_type == "ERROR":
            raise RuntimeError(event["error"])


if __name__ == "__main__":
    main()
