#!/usr/bin/env python3
"""Standalone grasp motion test — no dora required.

Generates a grasp trajectory on CUDA using real RealSense depth,
then plays it back through the fake-can-server via xoq-can.

Usage:
    python standalone_grasp_test.py
"""

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import can
import numpy as np
import torch
import pytorch_kinematics as pk

from dora_motion_planner.collision_model import CapsuleCollisionModel, OPENARM_CAPSULES
from dora_motion_planner.grasp_utils import grasp_pose_from_jaw_pixels
from dora_motion_planner.main import solve_ik
from dora_motion_planner.pointcloud import (
    depth_to_pointcloud,
    parse_camera_transform,
    transform_points,
    pointcloud_to_tensor,
)
from dora_motion_planner.trajectory_optimizer import TrajectoryOptimizer

# --- Config ---
URDF_PATH = Path(__file__).parent / ".." / "openarm" / "openarm_v10.urdf"
CAMERA_TRANSFORM_STR = "-0.23 0.71 0.3 90 -45 0"
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
END_EFFECTOR_LINK = "openarm_left_hand_tcp"
DEVICE = "cuda"
GRASP_JSON = '{"p1": [469, 486], "p2": [547, 486]}'
CAN_CHANNEL = "537074820a4fb3b0b08fa321734736c44999c2aa20e754d51863f06582b089e7"

# Trajectory params
NUM_WAYPOINTS = 40
NUM_SEEDS = 4
MAX_ITERS = 100
DOWNSAMPLE_STRIDE = 8
SAFETY_MARGIN = 0.02
GRASP_DEPTH_OFFSET = -0.01
FLOOR_HEIGHT = 0.005
APPROACH_MARGIN = 0.03

# Damiao motor protocol
Q_MAX = 12.5
V_MAX = 45.0
KP_MAX = 500.0
KD_MAX = 5.0
T_MAX = 18.0
MOTOR_IDS = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]
ENABLE_CMD = bytes([0xFF] * 6 + [0xFF, 0xFC])
DISABLE_CMD = bytes([0xFF] * 6 + [0xFF, 0xFD])


def float_to_uint(x, x_min, x_max, bits):
    span = x_max - x_min
    x_clamped = max(x_min, min(x_max, x))
    return round((x_clamped - x_min) / span * ((1 << bits) - 1))


def encode_mit_command(p_des=0.0, v_des=0.0, kp=0.0, kd=0.0, t_ff=0.0):
    p = float_to_uint(p_des, -Q_MAX, Q_MAX, 16)
    v = float_to_uint(v_des, -V_MAX, V_MAX, 12)
    kp_int = float_to_uint(kp, 0, KP_MAX, 12)
    kd_int = float_to_uint(kd, 0, KD_MAX, 12)
    t = float_to_uint(t_ff, -T_MAX, T_MAX, 12)
    return bytes([
        (p >> 8) & 0xFF, p & 0xFF,
        (v >> 4) & 0xFF, ((v & 0xF) << 4) | ((kp_int >> 8) & 0xF),
        kp_int & 0xFF, (kd_int >> 4) & 0xFF,
        ((kd_int & 0xF) << 4) | ((t >> 8) & 0xF), t & 0xFF,
    ])


def build_chain():
    with open(URDF_PATH) as f:
        urdf_str = f.read()
    root = ET.fromstring(urdf_str)
    for link in root.findall(".//link"):
        for tag in ("visual", "collision", "inertial"):
            for elem in link.findall(tag):
                link.remove(elem)
    return pk.build_serial_chain_from_urdf(
        ET.tostring(root, encoding="unicode"), END_EFFECTOR_LINK
    )


def get_depth_from_realsense():
    """Capture aligned depth frame from the RealSense.

    Uses align-to-color so depth intrinsics match color resolution and
    depth pixels are registered to color pixels (the grasp coordinates
    come from the color image).
    """
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device("233522074606")
    config.enable_stream(rs.stream.depth, IMAGE_WIDTH, IMAGE_HEIGHT, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, IMAGE_WIDTH, IMAGE_HEIGHT, rs.format.rgb8, 30)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    try:
        # Skip a few frames for auto-exposure
        for _ in range(30):
            pipeline.wait_for_frames()

        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()

        # Get intrinsics from the aligned depth (= color intrinsics)
        intr = depth_frame.profile.as_video_stream_profile().intrinsics
        fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy

        depth_data = np.asanyarray(depth_frame.get_data()).ravel().astype(np.uint16)
        print(f"[realsense] Aligned depth frame: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
        print(f"[realsense] Intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

        # Depth stats
        depth_2d = depth_data.reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
        valid = depth_2d[depth_2d > 0]
        if len(valid) > 0:
            print(f"[realsense] Depth range: {valid.min()}–{valid.max()} mm, "
                  f"median={np.median(valid):.0f} mm, "
                  f"valid pixels: {len(valid)}/{IMAGE_WIDTH*IMAGE_HEIGHT} "
                  f"({100*len(valid)/(IMAGE_WIDTH*IMAGE_HEIGHT):.1f}%)")
        else:
            print("[realsense] WARNING: No valid depth pixels!")

        return depth_data, (fx, fy, cx, cy)
    finally:
        pipeline.stop()


def diagnose_depth(depth, fx, fy, cx, cy, u1, v1, u2, v2):
    """Print depth values around the jaw pixels to diagnose issues."""
    depth_2d = depth.reshape(IMAGE_HEIGHT, IMAGE_WIDTH)

    for label, u, v in [("jaw1", u1, v1), ("jaw2", u2, v2)]:
        ui, vi = int(round(u)), int(round(v))
        # 11x11 patch
        half = 5
        v_lo = max(0, vi - half)
        v_hi = min(IMAGE_HEIGHT, vi + half + 1)
        u_lo = max(0, ui - half)
        u_hi = min(IMAGE_WIDTH, ui + half + 1)
        patch = depth_2d[v_lo:v_hi, u_lo:u_hi]
        valid = patch[(patch > 10) & (patch < 5000)]
        center_val = depth_2d[vi, ui] if 0 <= vi < IMAGE_HEIGHT and 0 <= ui < IMAGE_WIDTH else 0
        print(f"  {label} pixel=({ui},{vi}): center_depth={center_val}mm, "
              f"patch valid={len(valid)}/{patch.size}, "
              f"median={np.median(valid):.0f}mm" if len(valid) > 0 else
              f"  {label} pixel=({ui},{vi}): center_depth={center_val}mm, NO valid depth in patch!")

    # Also show depth at image center and nearby regions
    for label, u, v in [("center", IMAGE_WIDTH//2, IMAGE_HEIGHT//2),
                         ("offset1", IMAGE_WIDTH//2 + 100, IMAGE_HEIGHT//2),
                         ("offset2", IMAGE_WIDTH//2, IMAGE_HEIGHT//2 + 50)]:
        val = depth_2d[v, u]
        print(f"  {label} pixel=({u},{v}): depth={val}mm")


def find_valid_grasp_region(depth, fx, fy, cx, cy, cam_t, cam_rot):
    """Find a region with valid depth and suggest grasp coordinates."""
    depth_2d = depth.reshape(IMAGE_HEIGHT, IMAGE_WIDTH)

    # Look for the closest object (smallest valid depth) in the center region
    center_region = depth_2d[IMAGE_HEIGHT//4:3*IMAGE_HEIGHT//4,
                             IMAGE_WIDTH//4:3*IMAGE_WIDTH//4]
    valid_mask = (center_region > 100) & (center_region < 2000)

    if not np.any(valid_mask):
        print("  No valid depth in center region (100-2000mm)")
        return None

    # Find the shallowest (closest) cluster — likely the object
    min_depth = center_region[valid_mask].min()
    object_mask = valid_mask & (center_region < min_depth + 50)  # within 50mm of closest

    if np.sum(object_mask) < 10:
        print(f"  Too few object pixels ({np.sum(object_mask)})")
        return None

    ys, xs = np.where(object_mask)
    # Convert back to full image coords
    xs = xs + IMAGE_WIDTH // 4
    ys = ys + IMAGE_HEIGHT // 4

    cx_obj = int(np.median(xs))
    cy_obj = int(np.median(ys))
    spread = max(20, int(np.std(xs)))

    print(f"  Found object: center=({cx_obj},{cy_obj}), depth={min_depth}mm, "
          f"spread={spread}px, pixels={np.sum(object_mask)}")

    # Suggest jaw positions: horizontal grasp across the object
    u1 = cx_obj - spread // 2
    v1 = cy_obj
    u2 = cx_obj + spread // 2
    v2 = cy_obj

    # Convert to 0-1000 normalized
    p1_norm = [int(u1 * 1000 / IMAGE_WIDTH), int(v1 * 1000 / IMAGE_HEIGHT)]
    p2_norm = [int(u2 * 1000 / IMAGE_WIDTH), int(v2 * 1000 / IMAGE_HEIGHT)]
    print(f"  Suggested grasp: p1={p1_norm}, p2={p2_norm} (pixels: ({u1},{v1})-({u2},{v2}))")

    return u1, v1, u2, v2


def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"=== Standalone Grasp Motion Test ===")
    print(f"Device: {device}")

    # 1. Build kinematic chain
    print("\n[1/7] Loading URDF and building chain...")
    chain = build_chain()
    joint_limits = chain.get_joint_limits()
    num_joints = len(chain.get_joint_parameter_names(exclude_fixed=True))
    ik_chain = chain.to(dtype=torch.float32, device=str(device))
    print(f"  Chain: {num_joints} joints, EE={END_EFFECTOR_LINK}")

    # 2. Get depth from RealSense
    print("\n[2/7] Capturing depth from RealSense...")
    depth, (fx, fy, cx, cy) = get_depth_from_realsense()

    # 3. Parse grasp and compute 3D pose
    print(f"\n[3/7] Computing grasp pose from {GRASP_JSON}...")
    cam_t, cam_rot = parse_camera_transform(CAMERA_TRANSFORM_STR)
    data = json.loads(GRASP_JSON)
    u1 = float(data["p1"][0]) * IMAGE_WIDTH / 1000.0
    v1 = float(data["p1"][1]) * IMAGE_HEIGHT / 1000.0
    u2 = float(data["p2"][0]) * IMAGE_WIDTH / 1000.0
    v2 = float(data["p2"][1]) * IMAGE_HEIGHT / 1000.0
    print(f"  Jaw pixels: ({u1:.0f},{v1:.0f}) ({u2:.0f},{v2:.0f})")

    # Diagnose depth at jaw locations
    print("\n  Depth diagnosis:")
    diagnose_depth(depth, fx, fy, cx, cy, u1, v1, u2, v2)

    result = grasp_pose_from_jaw_pixels(
        u1, v1, u2, v2, depth, fx, fy, cx, cy,
        cam_translation=cam_t, cam_rotation=cam_rot,
        width=IMAGE_WIDTH, height=IMAGE_HEIGHT,
        grasp_depth_offset=GRASP_DEPTH_OFFSET,
        floor_height=FLOOR_HEIGHT, approach_margin=APPROACH_MARGIN,
    )

    if result is None:
        print("\n  Original grasp pixels have invalid depth. Searching for valid region...")
        found = find_valid_grasp_region(depth, fx, fy, cx, cy, cam_t, cam_rot)
        if found is None:
            print("  ERROR: No valid grasp region found. Is the camera pointing at a scene?")
            return
        u1, v1, u2, v2 = found
        print(f"\n  Retrying with auto-detected pixels: ({u1:.0f},{v1:.0f}) ({u2:.0f},{v2:.0f})")
        result = grasp_pose_from_jaw_pixels(
            u1, v1, u2, v2, depth, fx, fy, cx, cy,
            cam_translation=cam_t, cam_rotation=cam_rot,
            width=IMAGE_WIDTH, height=IMAGE_HEIGHT,
            grasp_depth_offset=GRASP_DEPTH_OFFSET,
            floor_height=FLOOR_HEIGHT, approach_margin=APPROACH_MARGIN,
        )
        if result is None:
            print("  ERROR: Still failed to compute grasp pose")
            return

    grasp_xyzrpy, pregrasp_xyzrpy = result
    print(f"\n  Grasp:     pos=({grasp_xyzrpy[0]:.4f}, {grasp_xyzrpy[1]:.4f}, {grasp_xyzrpy[2]:.4f})")
    print(f"  Pre-grasp: pos=({pregrasp_xyzrpy[0]:.4f}, {pregrasp_xyzrpy[1]:.4f}, {pregrasp_xyzrpy[2]:.4f})")

    # 4. Solve IK
    print("\n[4/7] Solving IK...")
    current_joints = torch.zeros(num_joints, dtype=torch.float32, device=device)

    t0 = time.time()
    q_pregrasp = solve_ik(ik_chain, pregrasp_xyzrpy, current_joints, joint_limits, device,
                          num_seeds=16, max_iters=1000)
    if q_pregrasp is None:
        print("  ERROR: IK failed for pre-grasp")
        return
    print(f"  Pre-grasp IK: {time.time()-t0:.1f}s")

    t0 = time.time()
    q_grasp = solve_ik(ik_chain, grasp_xyzrpy, q_pregrasp, joint_limits, device,
                       num_seeds=16, max_iters=1000)
    if q_grasp is None:
        print("  ERROR: IK failed for grasp")
        return
    print(f"  Grasp IK: {time.time()-t0:.1f}s")

    # 5. Build point cloud and optimize trajectory
    print(f"\n[5/7] Optimizing trajectory ({NUM_WAYPOINTS} waypoints, {NUM_SEEDS} seeds, {MAX_ITERS} iters)...")
    pc_cam = depth_to_pointcloud(depth, fx, fy, cx, cy,
                                  width=IMAGE_WIDTH, height=IMAGE_HEIGHT,
                                  stride=DOWNSAMPLE_STRIDE)
    pc_robot = transform_points(pc_cam, cam_t, cam_rot)
    pc_tensor = pointcloud_to_tensor(pc_robot, device) if len(pc_robot) > 0 else None
    print(f"  Point cloud: {len(pc_robot)} points")

    capsule_model = CapsuleCollisionModel(OPENARM_CAPSULES, device)
    optimizer = TrajectoryOptimizer(chain=chain, capsule_model=capsule_model,
                                    joint_limits=joint_limits, device=device,
                                    safety_margin=SAFETY_MARGIN)

    half = NUM_WAYPOINTS // 2
    t0 = time.time()
    traj1, cost1 = optimizer.optimize(q_start=current_joints, q_goal=q_pregrasp,
                                       point_cloud=pc_tensor, T=half,
                                       num_seeds=NUM_SEEDS, max_iters=MAX_ITERS)
    print(f"  Phase 1 (start→pre-grasp): cost={cost1:.4f}, {time.time()-t0:.1f}s")

    t0 = time.time()
    traj2, cost2 = optimizer.optimize(q_start=q_pregrasp, q_goal=q_grasp,
                                       point_cloud=pc_tensor, T=half,
                                       num_seeds=max(2, NUM_SEEDS // 2),
                                       max_iters=MAX_ITERS // 2)
    print(f"  Phase 2 (pre-grasp→grasp): cost={cost2:.4f}, {time.time()-t0:.1f}s")

    full_traj = torch.cat([traj1, traj2[1:]], dim=0).numpy().astype(np.float32)
    total_wp = full_traj.shape[0]
    print(f"  Combined: {total_wp} waypoints, total cost={cost1+cost2:.4f}")

    # 6. Connect to fake-can-server and enable motors
    print(f"\n[6/7] Connecting to fake-can-server...")
    bus = can.interface.Bus(channel=CAN_CHANNEL, interface="socketcan", fd=False)

    for mid in MOTOR_IDS:
        bus.send(can.Message(arbitration_id=mid, data=ENABLE_CMD, is_fd=False))
        bus.recv(timeout=0.5)
    print("  Motors enabled")

    # 7. Play back trajectory
    print(f"\n[7/7] Playing trajectory ({total_wp} waypoints at 10Hz)...")
    kp, kd = 30.0, 1.0

    for i, waypoint in enumerate(full_traj):
        for j in range(min(7, len(waypoint))):
            cmd = encode_mit_command(p_des=float(waypoint[j]), kp=kp, kd=kd)
            bus.send(can.Message(arbitration_id=MOTOR_IDS[j], data=cmd, is_fd=False))

        for _ in range(7):
            bus.recv(timeout=0.1)

        if i % 5 == 0 or i == total_wp - 1:
            angles = ", ".join(f"{a:7.3f}" for a in waypoint[:7])
            print(f"  [{i+1:3d}/{total_wp}] [{angles}]")

        time.sleep(0.1)

    print(f"\n=== Trajectory complete! ===")
    print(f"  Final position: [{', '.join(f'{a:.3f}' for a in full_traj[-1][:7])}]")

    for mid in MOTOR_IDS:
        bus.send(can.Message(arbitration_id=mid, data=DISABLE_CMD, is_fd=False))
    bus.shutdown()
    print("  Motors disabled, CAN bus closed.")


if __name__ == "__main__":
    main()
