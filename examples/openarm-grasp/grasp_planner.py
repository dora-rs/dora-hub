"""Grasp planner — state machine that orchestrates VLM-guided grasping.

Inputs: tick, vlm_response, depth, image, joint_state, user_feedback
Outputs: grasp_pose (xyzrpy), gripper_cmd (float), trigger (string),
         vlm_request (trigger to formatter), text (status)

Env vars:
    IMAGE_WIDTH, IMAGE_HEIGHT: camera resolution (default 640x480)
    CAMERA_TRANSFORM: tx ty tz qw qx qy qz (camera-to-robot frame)
    APPROACH_HEIGHT: height above grasp point for pre-grasp (default 0.10 m)
    LIFT_HEIGHT: height to lift after grasping (default 0.15 m)
    ARRIVAL_THRESHOLD: joint angle threshold for "arrived" (default 0.05 rad)
    ARRIVAL_TIMEOUT: max wait for arrival in seconds (default 5.0)
    GRIPPER_CLOSE_WAIT: seconds to wait after closing gripper (default 0.5)
    DATASET_PATH: path to JSONL log file (default ./grasp_dataset.jsonl)
    HOME_JOINTS: comma-separated home joint angles (7 values, radians)
"""

import json
import os
import time
from enum import Enum

import numpy as np
import pyarrow as pa
from dora import Node
from scipy.spatial.transform import Rotation

# --- Configuration ---

WIDTH = int(os.getenv("IMAGE_WIDTH", "640"))
HEIGHT = int(os.getenv("IMAGE_HEIGHT", "480"))
NORMALIZED_COORDS = os.getenv("NORMALIZED_COORDS", "true").lower() in ("1", "true", "yes")

# Camera-to-robot transform: tx ty tz qw qx qy qz
_ct = os.getenv("CAMERA_TRANSFORM", "0 0 0 1 0 0 0").split()
CAMERA_T = np.array([float(_ct[0]), float(_ct[1]), float(_ct[2])])
CAMERA_R = Rotation.from_quat(
    [float(_ct[4]), float(_ct[5]), float(_ct[6]), float(_ct[3])]  # xyzw for scipy
)

APPROACH_HEIGHT = float(os.getenv("APPROACH_HEIGHT", "0.10"))
LIFT_HEIGHT = float(os.getenv("LIFT_HEIGHT", "0.15"))
ARRIVAL_THRESHOLD = float(os.getenv("ARRIVAL_THRESHOLD", "0.05"))
ARRIVAL_TIMEOUT = float(os.getenv("ARRIVAL_TIMEOUT", "5.0"))
GRIPPER_CLOSE_WAIT = float(os.getenv("GRIPPER_CLOSE_WAIT", "0.5"))
DATASET_PATH = os.getenv("DATASET_PATH", "./grasp_dataset.jsonl")

GRIPPER_OPEN = -1.0472  # rad (motor space)
GRIPPER_CLOSED = 0.0

_home = os.getenv("HOME_JOINTS", "0,0,0,0,0,0,0")
HOME_JOINTS = np.array([float(x) for x in _home.split(",")], dtype=np.float32)

DEPTH_PATCH_SIZE = 5  # pixels, for averaging depth


class State(Enum):
    IDLE = "IDLE"
    WAITING_VLM = "WAITING_VLM"
    PRE_GRASP = "PRE_GRASP"
    APPROACH = "APPROACH"
    CLOSE_GRIPPER = "CLOSE_GRIPPER"
    LIFT = "LIFT"
    EVALUATE = "EVALUATE"
    RETURN_HOME = "RETURN_HOME"


# --- Utility functions ---


def pixel_to_3d_camera(u, v, depth_map, fx, fy, cx, cy):
    """Convert pixel (u, v) to 3D point in camera frame using depth map.

    Averages depth over a patch to handle noise/holes.
    Returns [x, y, z] in meters, or None if depth is invalid.
    """
    h, w = depth_map.shape
    half = DEPTH_PATCH_SIZE // 2
    u_min = max(0, int(u) - half)
    u_max = min(w, int(u) + half + 1)
    v_min = max(0, int(v) - half)
    v_max = min(h, int(v) + half + 1)

    patch = depth_map[v_min:v_max, u_min:u_max].astype(np.float64)
    valid = patch[patch > 0]
    if len(valid) == 0:
        return None

    z = np.median(valid) / 1000.0  # mm -> m
    if z < 0.01 or z > 2.0:
        return None

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z])


def camera_to_robot(point_cam):
    """Transform a 3D point from camera frame to robot frame."""
    return CAMERA_R.apply(point_cam) + CAMERA_T


def compute_grasp_pose(p1_robot, p2_robot):
    """Compute grasp pose (xyzrpy) from two 3D points in robot frame.

    - Position = midpoint of p1 and p2
    - Gripper X-axis = normalized(p2 - p1) = finger opening direction
    - Gripper Z-axis = [0, 0, -1] = approach from above
    - Y-axis = cross(Z, X), re-orthogonalized
    - Returns [x, y, z, roll, pitch, yaw] (XYZ Euler angles)
    """
    midpoint = (p1_robot + p2_robot) / 2.0

    # Gripper axes
    x_axis = p2_robot - p1_robot
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-6:
        return None
    x_axis = x_axis / x_norm

    z_axis = np.array([0.0, 0.0, -1.0])  # approach from above

    y_axis = np.cross(z_axis, x_axis)
    y_norm = np.linalg.norm(y_axis)
    if y_norm < 1e-6:
        # Points are vertical — fallback
        y_axis = np.array([0.0, 1.0, 0.0])
    else:
        y_axis = y_axis / y_norm

    # Re-orthogonalize x_axis
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    rot_matrix = np.column_stack([x_axis, y_axis, z_axis])
    rpy = Rotation.from_matrix(rot_matrix).as_euler("xyz")

    return np.concatenate([midpoint, rpy]).astype(np.float32)


def strip_think_tags(text):
    """Strip <think>...</think> blocks from thinking model output."""
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def parse_vlm_response(text):
    """Parse VLM JSON response to extract two pixel points.

    Expected format: {"p1":[x1,y1],"p2":[x2,y2]}
    Returns (p1, p2) as tuples or None on failure.
    """
    text = strip_think_tags(text).strip()
    # Try to find JSON in the response
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        data = json.loads(text[start:end])
        p1 = data["p1"]
        p2 = data["p2"]
        if len(p1) == 2 and len(p2) == 2:
            return (p1[0], p1[1]), (p2[0], p2[1])
    except (json.JSONDecodeError, KeyError, TypeError, IndexError):
        pass
    return None


def log_grasp(dataset_path, entry):
    """Append a grasp attempt entry to JSONL dataset."""
    with open(dataset_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"Logged grasp attempt to {dataset_path}")


# --- Main node ---


def main():
    node = Node()

    state = State.IDLE
    state_enter_time = time.time()

    # Cached data
    latest_depth = None
    latest_image_bytes = None
    current_joint_state = None
    camera_intrinsics = None  # (fx, fy, cx, cy)

    # Current grasp attempt data
    grasp_data = {}

    def set_state(new_state):
        nonlocal state, state_enter_time
        print(f"State: {state.value} -> {new_state.value}")
        state = new_state
        state_enter_time = time.time()
        node.send_output("text", pa.array([f"state:{new_state.value}"]))

    def send_grasp_pose(pose):
        """Send xyzrpy to IK node."""
        metadata = {"encoding": "xyzrpy"}
        node.send_output(
            "grasp_pose", pa.array(pose, type=pa.float32()), metadata
        )
        print(
            f"Sent grasp pose: xyz=[{pose[0]:.3f},{pose[1]:.3f},{pose[2]:.3f}] "
            f"rpy=[{pose[3]:.3f},{pose[4]:.3f},{pose[5]:.3f}]"
        )

    def send_gripper(value):
        """Send gripper command (radians)."""
        node.send_output(
            "gripper_cmd", pa.array([value], type=pa.float32())
        )
        label = "OPEN" if value < -0.5 else "CLOSED"
        print(f"Gripper: {label} ({value:.4f} rad)")

    def check_arrival():
        """Check if current joints are close to the last IK target."""
        if current_joint_state is None:
            return False
        elapsed = time.time() - state_enter_time
        if elapsed > ARRIVAL_TIMEOUT:
            print(f"Arrival timeout after {elapsed:.1f}s")
            return True
        # We rely on timeout for now — joint state comparison can be added
        # when we have the IK solution cached
        return elapsed > 1.0  # minimum settle time

    for event in node:
        if event["type"] == "INPUT":
            event_id = event["id"]

            # --- Always update cached data ---
            if event_id == "image":
                latest_image_bytes = event["value"].to_numpy().tobytes()
                metadata = event["metadata"]
                if "focal_length" in metadata and "resolution" in metadata:
                    fl = metadata["focal_length"]
                    pp = metadata["resolution"]  # actually principal point
                    camera_intrinsics = (
                        float(fl[0]),
                        float(fl[1]),
                        float(pp[0]),
                        float(pp[1]),
                    )
                continue

            if event_id == "depth":
                raw = event["value"].to_numpy()
                latest_depth = raw.reshape((HEIGHT, WIDTH)).astype(np.uint16)
                continue

            if event_id == "joint_state":
                current_joint_state = event["value"].to_numpy().astype(
                    np.float32
                )
                continue

            # --- State machine transitions ---

            if event_id == "tick":
                if state == State.IDLE:
                    # Do nothing — wait for user "go" command
                    pass

                elif state == State.PRE_GRASP:
                    if check_arrival():
                        set_state(State.APPROACH)
                        # Send approach pose (lower to grasp point)
                        approach_pose = grasp_data["grasp_pose"].copy()
                        send_grasp_pose(approach_pose)

                elif state == State.APPROACH:
                    if check_arrival():
                        set_state(State.CLOSE_GRIPPER)
                        send_gripper(GRIPPER_CLOSED)

                elif state == State.CLOSE_GRIPPER:
                    elapsed = time.time() - state_enter_time
                    if elapsed >= GRIPPER_CLOSE_WAIT:
                        set_state(State.LIFT)
                        lift_pose = grasp_data["grasp_pose"].copy()
                        lift_pose[2] += LIFT_HEIGHT
                        send_grasp_pose(lift_pose)

                elif state == State.LIFT:
                    if check_arrival():
                        set_state(State.EVALUATE)
                        print(
                            "Grasp complete. Type 'success' or 'fail' to log result."
                        )

                elif state == State.RETURN_HOME:
                    if check_arrival():
                        set_state(State.IDLE)
                        print("Ready. Type 'go' to start next grasp attempt.")

            elif event_id == "user_feedback":
                cmd = event["value"][0].as_py().strip().lower()

                if state == State.IDLE and cmd == "go":
                    if latest_image_bytes is None:
                        print("No image received yet — wait for camera")
                        continue
                    print("Starting grasp attempt — requesting VLM inference")
                    set_state(State.WAITING_VLM)
                    # Trigger VLM formatter
                    node.send_output("vlm_request", pa.array(["go"]))
                    grasp_data = {"timestamp": time.time()}

                elif state == State.EVALUATE:
                    if cmd in ("success", "fail"):
                        grasp_data["result"] = cmd
                        grasp_data["final_joint_state"] = (
                            current_joint_state.tolist()
                            if current_joint_state is not None
                            else None
                        )
                        log_grasp(DATASET_PATH, grasp_data)
                        print(f"Logged: {cmd}. Returning home...")
                        set_state(State.RETURN_HOME)
                        send_gripper(GRIPPER_OPEN)
                        # Send home joint angles directly to arm (bypassing IK)
                        node.send_output(
                            "home_command",
                            pa.array(HOME_JOINTS, type=pa.float32()),
                            {"encoding": "jointstate"},
                        )
                    else:
                        print(
                            "Type 'success' or 'fail' to log the grasp result"
                        )

                elif cmd == "stop":
                    print("Emergency stop — returning to IDLE")
                    send_gripper(GRIPPER_OPEN)
                    set_state(State.IDLE)

            elif event_id == "vlm_response":
                if state != State.WAITING_VLM:
                    print(
                        f"VLM response in unexpected state {state.value}, ignoring"
                    )
                    continue

                text = event["value"][0].as_py()
                print(f"VLM response: {text}")

                points = parse_vlm_response(text)
                if points is None:
                    print("Failed to parse VLM response — returning to IDLE")
                    set_state(State.IDLE)
                    continue

                (u1, v1), (u2, v2) = points
                if NORMALIZED_COORDS:
                    u1 = u1 * WIDTH / 1000
                    v1 = v1 * HEIGHT / 1000
                    u2 = u2 * WIDTH / 1000
                    v2 = v2 * HEIGHT / 1000
                print(f"Grasp points: p1=({u1:.0f},{v1:.0f}), p2=({u2:.0f},{v2:.0f})")

                if camera_intrinsics is None:
                    print("No camera intrinsics — returning to IDLE")
                    set_state(State.IDLE)
                    continue

                if latest_depth is None:
                    print("No depth data — returning to IDLE")
                    set_state(State.IDLE)
                    continue

                fx, fy, cx, cy = camera_intrinsics

                # Pixel to 3D (camera frame)
                p1_cam = pixel_to_3d_camera(u1, v1, latest_depth, fx, fy, cx, cy)
                p2_cam = pixel_to_3d_camera(u2, v2, latest_depth, fx, fy, cx, cy)

                if p1_cam is None or p2_cam is None:
                    print("Invalid depth at grasp points — returning to IDLE")
                    set_state(State.IDLE)
                    continue

                print(
                    f"3D camera frame: p1={p1_cam.tolist()}, p2={p2_cam.tolist()}"
                )

                # Camera to robot frame
                p1_robot = camera_to_robot(p1_cam)
                p2_robot = camera_to_robot(p2_cam)
                print(
                    f"3D robot frame: p1={p1_robot.tolist()}, p2={p2_robot.tolist()}"
                )

                # Compute grasp pose
                pose = compute_grasp_pose(p1_robot, p2_robot)
                if pose is None:
                    print("Failed to compute grasp pose — returning to IDLE")
                    set_state(State.IDLE)
                    continue

                # Store data for logging
                grasp_data.update(
                    {
                        "pixel_p1": [u1, v1],
                        "pixel_p2": [u2, v2],
                        "depth_p1_mm": float(
                            latest_depth[int(v1), int(u1)]
                        ),
                        "depth_p2_mm": float(
                            latest_depth[int(v2), int(u2)]
                        ),
                        "cam_p1": p1_cam.tolist(),
                        "cam_p2": p2_cam.tolist(),
                        "robot_p1": p1_robot.tolist(),
                        "robot_p2": p2_robot.tolist(),
                        "grasp_pose": pose.tolist(),
                        "vlm_raw": text,
                    }
                )

                # Execute: open gripper, move to pre-grasp (above target)
                set_state(State.PRE_GRASP)
                send_gripper(GRIPPER_OPEN)
                pre_grasp_pose = pose.copy()
                pre_grasp_pose[2] += APPROACH_HEIGHT  # raise Z
                send_grasp_pose(pre_grasp_pose)

        elif event["type"] == "STOP":
            print("Received STOP")
            break

    print("Grasp planner exiting")


if __name__ == "__main__":
    main()
