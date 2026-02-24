#!/usr/bin/env python3
"""General-purpose grasp trajectory planner.

Reads an openarm-config JSON (exported from the web viewer) and grasp
targets (from the dora selector+SAM3 pipeline or manual JSON), then
plans collision-aware trajectories and saves them as JSON + animated GIF.

Usage:
    python grasp_trajectory.py \\
        --config openarm-config.json \\
        --targets targets.json \\
        --output output/ \\
        --device cuda

    # Single inline target
    python grasp_trajectory.py \\
        --config openarm-config.json \\
        --targets '{"p1": [430, 410], "p2": [480, 410]}' \\
        --no-gif

    # Specify camera and robot labels
    python grasp_trajectory.py \\
        --config openarm-config.json \\
        --targets targets.json \\
        --camera champagne-realsense \\
        --robot champagne-arm
"""

import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from dora_motion_planner.collision_model import (
    CapsuleCollisionModel,
    OPENARM_CAPSULES,
    OPENARM_RIGHT_CAPSULES,
    OPENARM_GRIPPER_BOXES,
)
from dora_motion_planner.grasp_utils import grasp_pose_from_jaw_pixels
from dora_motion_planner.main import (
    _ik_batch_adam,
    validate_trajectory,
    _log_collision_check,
)
from dora_motion_planner.collision_model import (
    capsule_halfplane_distance,
    capsule_points_distance,
)
from dora_motion_planner.pointcloud import (
    compute_table_plane,
    depth_to_pointcloud,
    parse_camera_transform,
    transform_points,
    pointcloud_to_tensor,
)
from dora_motion_planner.trajectory_optimizer import TrajectoryOptimizer
from dora_motion_planner.trajectory_json import save as save_trajectory

# --- Constants ---
DEFAULT_URDF = Path(__file__).parent / ".." / "openarm" / "openarm_v10.urdf"
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
LEFT_EE = "openarm_left_hand_tcp"
RIGHT_EE = "openarm_right_hand_tcp"

# Trajectory defaults
NUM_WAYPOINTS = 90
NUM_SEEDS = 1
MAX_ITERS = 400
DOWNSAMPLE_STRIDE = 8
UPSAMPLE_FACTOR = 4
SAFETY_MARGIN = 0.02
GRASP_DEPTH_OFFSET = 0.03   # 30mm above object surface
FLOOR_HEIGHT = 0.02          # min 20mm above ground (capsule clearance)
APPROACH_MARGIN = 0.12       # pre-grasp 120mm above grasp
IK_TOL = 0.08


# ---- Config loading ----

def load_config(path):
    """Parse and validate an openarm-config v3 JSON file."""
    with open(path) as f:
        cfg = json.load(f)
    version = cfg.get("version")
    if version != 3:
        print(f"[config] WARNING: expected version 3, got {version}")
    return cfg


def resolve_camera(config, label=None):
    """Find a realsense entry and return (transform_str, relay_path).

    Args:
        config: Parsed openarm-config dict.
        label: Camera label to match. If None, uses first enabled entry.

    Returns:
        (transform_str, relay_path) or None.
    """
    cameras = config.get("realsense", [])
    if not cameras:
        print("[config] No realsense entries in config")
        return None

    entry = None
    if label:
        for cam in cameras:
            if cam.get("label") == label:
                entry = cam
                break
        if entry is None:
            labels = [c.get("label", "?") for c in cameras]
            print(f"[config] Camera '{label}' not found. Available: {labels}")
            return None
    else:
        for cam in cameras:
            if cam.get("enabled", True):
                entry = cam
                break
        if entry is None:
            entry = cameras[0]

    pos = entry["position"]
    rot = entry["rotation"]
    transform_str = (f"{pos['x']} {pos['y']} {pos['z']} "
                     f"{rot['roll']} {rot['pitch']} {rot['yaw']}")
    relay = entry["path"]

    print(f"[config] Camera: {entry.get('label', '?')}")
    print(f"[config] Relay: {relay}")
    print(f"[config] Transform: {transform_str}")
    return transform_str, relay


def resolve_robot(config, label=None, camera_relay_path=None):
    """Find an armPairs entry by label or by matching camera machine ID.

    When *label* is given, matches by exact label.  Otherwise, if
    *camera_relay_path* is provided, auto-matches by extracting the
    machine-ID segment from the relay paths (e.g. ``anon/<machine>/...``).

    Returns the matched entry dict, or None.
    """
    pairs = config.get("armPairs", [])
    if not pairs:
        return None
    if label:
        for pair in pairs:
            if pair.get("label") == label:
                print(f"[config] Robot: {pair.get('label')}")
                return pair
        labels = [p.get("label", "?") for p in pairs]
        print(f"[config] Robot '{label}' not found. Available: {labels}")
        return None

    # Auto-match: extract machine ID from camera relay path
    if camera_relay_path:
        parts = camera_relay_path.split("/")
        cam_machine = parts[1] if len(parts) >= 3 else None
        if cam_machine:
            for pair in pairs:
                for key in ("leftPath", "rightPath"):
                    path = pair.get(key, "")
                    p = path.split("/")
                    if len(p) >= 3 and p[1] == cam_machine:
                        print(f"[config] Robot: {pair.get('label')} (auto-matched via {cam_machine})")
                        return pair

    return None


def arm_offset_urdf(pair):
    """Convert an armPairs position from scene (Y-up) to URDF (Z-up) frame.

    Scene coords ``(x_s, y_s, z_s)`` → URDF ``(x_s, -z_s, y_s)``.
    Returns a (3,) float32 array, or zeros if no position is set.
    """
    pos = pair.get("position", {})
    x_s = float(pos.get("x", 0))
    y_s = float(pos.get("y", 0))
    z_s = float(pos.get("z", 0))
    offset = np.array([x_s, -z_s, y_s], dtype=np.float32)
    return offset


# ---- Target parsing ----

def load_targets(arg):
    """Load grasp targets from a file path or inline JSON string.

    Accepts:
        - File path to a JSON file
        - Inline JSON string

    Formats:
        Multi-target: {"targets": [{"label": "...", "p1": [...], "p2": [...]}, ...]}
        Single-target: {"p1": [...], "p2": [...]}

    Returns list of target dicts, each with "label", "p1", "p2".
    """
    # Try as file first
    if os.path.isfile(arg):
        with open(arg) as f:
            data = json.load(f)
    else:
        data = json.loads(arg)

    if "targets" in data:
        targets = data["targets"]
    elif "p1" in data and "p2" in data:
        targets = [data]
    else:
        raise ValueError(f"Unrecognised target format: {list(data.keys())}")

    # Ensure labels
    for i, t in enumerate(targets):
        if "label" not in t:
            t["label"] = f"target_{i}"

    return targets


# ---- Depth capture ----

def capture_depth(relay_path, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    """Capture aligned depth frame via xoq_realsense relay.

    Returns (depth_flat, (fx, fy, cx, cy)).
    """
    import xoq_realsense as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(relay_path)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, 30)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    try:
        print("[realsense] Warming up (10 frames)...")
        for _ in range(10):
            pipeline.wait_for_frames()

        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()

        vsp = depth_frame.profile.as_video_stream_profile()
        intr = vsp.get_intrinsics()
        fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy

        depth_data = np.asanyarray(depth_frame.get_data()).ravel().astype(np.uint16)
        print(f"[realsense] Depth frame: {width}x{height}")
        print(f"[realsense] Intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

        depth_2d = depth_data.reshape(height, width)
        valid = depth_2d[depth_2d > 0]
        if len(valid) > 0:
            print(f"[realsense] Depth range: {valid.min()}-{valid.max()}mm, "
                  f"median={np.median(valid):.0f}mm")

        return depth_data, (fx, fy, cx, cy)
    finally:
        pipeline.stop()


# ---- URDF / chain helpers ----

def build_chain(urdf_path, ee_link):
    """Build a pytorch_kinematics serial chain from URDF."""
    import pytorch_kinematics as pk

    with open(urdf_path) as f:
        urdf_str = f.read()
    root = ET.fromstring(urdf_str)
    for link in root.findall(".//link"):
        for tag in ("visual", "collision", "inertial"):
            for elem in link.findall(tag):
                link.remove(elem)
    return pk.build_serial_chain_from_urdf(
        ET.tostring(root, encoding="unicode"), ee_link
    )


def validate_ik_config(ik_chain, capsule_model, q, table_plane, table_polygon,
                       margin, device, point_cloud=None):
    """Check if a single joint configuration collides with the table or point cloud.

    Returns list of (link_name, min_distance) for any capsule link closer
    than *margin* to the table plane or point cloud.  Empty list = no collision.
    """
    collisions = []
    has_table = table_plane is not None
    has_pc = point_cloud is not None and len(point_cloud) > 0

    if not has_table and not has_pc:
        return collisions

    with torch.no_grad():
        transforms = ik_chain.forward_kinematics(q.unsqueeze(0).to(device), end_only=False)
        for link_name in capsule_model.link_names:
            if link_name not in transforms:
                continue
            p0_w, p1_w = capsule_model.capsule_endpoints_world(link_name, transforms)
            radius = capsule_model.radii[link_name]

            if has_table:
                plane_z, bounds = table_plane
                sd = capsule_halfplane_distance(
                    p0_w, p1_w, radius, plane_z, bounds,
                    plane_polygon=table_polygon,
                )
                d = float(sd[0])
                if d < margin:
                    collisions.append((f"{link_name}[table]", d))

            if has_pc:
                sd = capsule_points_distance(p0_w, p1_w, radius, point_cloud)
                d = float(sd.min())
                if d < margin:
                    collisions.append((link_name, d))

    return collisions


# ---- Core planning ----

@dataclass
class GraspResult:
    label: str
    arm: str
    trajectory: np.ndarray
    grasp_xyzrpy: list
    pregrasp_xyzrpy: list
    ik_pre_err: float
    ik_grasp_err: float
    cost: float
    collisions: list
    object_width: float = 0.0  # jaw-to-jaw distance in metres
    pc_robot: np.ndarray = field(default=None)
    table_corners: np.ndarray = field(default=None)  # (4, 3) frustum-projected table polygon


def plan_single_target(
    target, depth, intrinsics, cam_t, cam_rot, device, urdf_path,
    width=IMAGE_WIDTH, height=IMAGE_HEIGHT,
):
    """Plan a grasp trajectory for a single target.

    Args:
        target: Dict with "label", "p1", "p2" (0-1000 normalized coords).
        depth: Flat uint16 depth array.
        intrinsics: (fx, fy, cx, cy).
        cam_t: Camera translation from parse_camera_transform.
        cam_rot: Camera rotation matrix from parse_camera_transform.
        device: torch.device.
        urdf_path: Path to URDF.
        width, height: Image dimensions.

    Returns:
        GraspResult or None on failure.
    """
    label = target["label"]
    fx, fy, cx, cy = intrinsics

    # Denormalize p1/p2 from 0-1000 to pixel coordinates
    u1 = float(target["p1"][0]) * width / 1000.0
    v1 = float(target["p1"][1]) * height / 1000.0
    u2 = float(target["p2"][0]) * width / 1000.0
    v2 = float(target["p2"][1]) * height / 1000.0
    print(f"  Jaw pixels: ({u1:.0f},{v1:.0f}) ({u2:.0f},{v2:.0f})")

    # Diagnose depth at jaw locations
    depth_2d = depth.reshape(height, width)
    for jlabel, u, v in [("jaw1", u1, v1), ("jaw2", u2, v2)]:
        ui, vi = int(round(u)), int(round(v))
        val = depth_2d[vi, ui] if 0 <= vi < height and 0 <= ui < width else 0
        print(f"  {jlabel} pixel=({ui},{vi}): depth={val}mm")

    # Compute 3D grasp pose
    result = grasp_pose_from_jaw_pixels(
        u1, v1, u2, v2, depth, fx, fy, cx, cy,
        cam_translation=cam_t, cam_rotation=cam_rot,
        width=width, height=height,
        grasp_depth_offset=GRASP_DEPTH_OFFSET,
        floor_height=FLOOR_HEIGHT, approach_margin=APPROACH_MARGIN,
    )
    if result is None:
        print(f"  ERROR: Invalid depth at jaw points for '{label}'")
        return None

    grasp_xyzrpy, pregrasp_xyzrpy, object_top_z, object_width = result
    print(f"  Grasp:     pos=({grasp_xyzrpy[0]:.4f}, {grasp_xyzrpy[1]:.4f}, {grasp_xyzrpy[2]:.4f})")
    print(f"  Pre-grasp: pos=({pregrasp_xyzrpy[0]:.4f}, {pregrasp_xyzrpy[1]:.4f}, {pregrasp_xyzrpy[2]:.4f})")

    # Build point cloud (shared across arm attempts)
    pc_cam = depth_to_pointcloud(depth, fx, fy, cx, cy,
                                  width=width, height=height,
                                  stride=DOWNSAMPLE_STRIDE)
    pc_robot = transform_points(pc_cam, cam_t, cam_rot)
    print(f"  Point cloud: {len(pc_robot)} points (raw)")

    # Fit table plane from camera frustum projection
    table_corners_robot = None
    table_polygon = None
    table_plane = None
    if len(pc_robot) > 100:
        table_plane, table_polygon = compute_table_plane(
            cam_t, cam_rot, fx, fy, cx, cy, width, height, pc_robot,
            device=device,
        )
        plane_z, bounds = table_plane
        print(f"  Table plane: z={plane_z:.3f}m, "
              f"x=[{bounds[0]:.2f},{bounds[1]:.2f}], y=[{bounds[2]:.2f},{bounds[3]:.2f}]")

        # Remove table surface points from the SDF — the table plane cost
        # already prevents the arm from going below the surface.  Keeping
        # table surface points in the SDF creates a double-penalty that
        # prevents the arm from passing *over* the table cleanly.
        # Keep only obstacle points that stick up above the table.
        z_above = plane_z + 0.02  # 20mm above table = still surface
        mask = pc_robot[:, 2] > z_above
        pc_robot = pc_robot[mask]
        print(f"  Point cloud: {len(pc_robot)} points (after z>{z_above:.3f}m filter, obstacles only)")

        if table_polygon is not None:
            poly_np = table_polygon.cpu().numpy()
            table_corners_robot = np.column_stack([poly_np, np.full(len(poly_np), plane_z)])
    else:
        print("  WARNING: No point cloud for table plane detection")

    # Adjust grasp Z to midpoint between object top and table surface.
    # This centres the gripper on the object rather than grasping near the top.
    # For flat objects, clamp so the gripper doesn't go into the table.
    if table_plane is not None:
        table_z = table_plane[0]
        mid_z = (object_top_z + table_z) / 2.0
        min_grasp_z = table_z + GRASP_DEPTH_OFFSET  # at least 30mm above table
        mid_z = max(mid_z, min_grasp_z)
        print(f"  Grasp Z: object_top={object_top_z:.3f}m, table={table_z:.3f}m "
              f"→ mid={mid_z:.3f}m (min={min_grasp_z:.3f}m)")
        grasp_xyzrpy[2] = mid_z
        pregrasp_xyzrpy[2] = mid_z + APPROACH_MARGIN

    pc_tensor = pointcloud_to_tensor(pc_robot, device) if len(pc_robot) > 0 else None

    # Rank arms by IK error
    arm_candidates = []
    for arm_name, ee_link, caps in [
        ("left", LEFT_EE, OPENARM_CAPSULES),
        ("right", RIGHT_EE, OPENARM_RIGHT_CAPSULES),
    ]:
        try_chain = build_chain(urdf_path, ee_link)
        try_ik = try_chain.to(dtype=torch.float32, device=str(device))
        try_nj = len(try_chain.get_joint_parameter_names(exclude_fixed=True))
        try_limits = try_chain.get_joint_limits()
        lower = torch.tensor(try_limits[0], dtype=torch.float32, device=device)
        upper = torch.tensor(try_limits[1], dtype=torch.float32, device=device)
        target_pos = torch.tensor(pregrasp_xyzrpy[:3], dtype=torch.float32, device=device)
        _, err = _ik_batch_adam(
            try_ik, target_pos, None,
            torch.zeros(try_nj, device=device), lower, upper, device,
            num_seeds=16, max_iters=500, rot_weight=0.0,
        )
        print(f"  {arm_name} arm IK error: {err:.4f}m")
        if err <= IK_TOL:
            arm_candidates.append((arm_name, err, ee_link, caps))

    if not arm_candidates:
        print(f"  ERROR: No arm can reach the target")
        return None

    # Sort by IK error (best first)
    arm_candidates.sort(key=lambda x: x[1])

    # Try each arm until we get a collision-free trajectory (or use the best)
    best_result = None
    for arm_name, arm_err, ee_link, capsules in arm_candidates:
        print(f"\n  --- Trying {arm_name} arm (IK err={arm_err:.4f}m) ---")

        chain = build_chain(urdf_path, ee_link)
        joint_limits = chain.get_joint_limits()
        num_joints = len(chain.get_joint_parameter_names(exclude_fixed=True))
        ik_chain = chain.to(dtype=torch.float32, device=str(device))
        capsule_model = CapsuleCollisionModel(capsules, device)

        current_joints = torch.zeros(num_joints, dtype=torch.float32, device=device)
        lower = torch.tensor(joint_limits[0], dtype=torch.float32, device=device)
        upper = torch.tensor(joint_limits[1], dtype=torch.float32, device=device)

        # Solve IK for pre-grasp — validate only against table (hard surface).
        # Point cloud validation is too strict for endpoints since the gripper
        # is supposed to approach the object, and the optimizer handles interior
        # waypoints.  Retry with fresh seeds if config collides with table.
        IK_MAX_RETRIES = 3
        q_pregrasp = None
        for ik_try in range(IK_MAX_RETRIES):
            t0 = time.time()
            q_candidate, pre_err = _ik_batch_adam(
                ik_chain, torch.tensor(pregrasp_xyzrpy[:3], dtype=torch.float32, device=device),
                None, current_joints, lower, upper, device,
                num_seeds=32, max_iters=2000, rot_weight=0.0,
            )
            if pre_err > IK_TOL:
                print(f"  Pre-grasp IK failed (err={pre_err:.4f}m), skipping arm")
                break
            pre_collisions = validate_ik_config(
                ik_chain, capsule_model, q_candidate, table_plane, table_polygon,
                SAFETY_MARGIN, device,
            )
            if not pre_collisions:
                q_pregrasp = q_candidate
                print(f"  Pre-grasp IK: err={pre_err:.4f}m, {time.time()-t0:.1f}s")
                break
            worst = min(pre_collisions, key=lambda c: c[1])
            print(f"  Pre-grasp IK attempt {ik_try+1}: table collision "
                  f"link={worst[0]} dist={worst[1]*1000:.1f}mm, retrying...")
        if q_pregrasp is None:
            print(f"  Pre-grasp IK: no table-clear config after {IK_MAX_RETRIES} attempts, skipping arm")
            continue

        # Solve IK for grasp — no table validation needed since the
        # descend phase is unoptimized and collision-skipped (the gripper
        # IS supposed to be near the table at the grasp pose).
        t0 = time.time()
        q_grasp, grasp_err = _ik_batch_adam(
            ik_chain, torch.tensor(grasp_xyzrpy[:3], dtype=torch.float32, device=device),
            None, q_pregrasp, lower, upper, device,
            num_seeds=32, max_iters=2000, rot_weight=0.0,
        )
        if grasp_err > IK_TOL:
            print(f"  Grasp IK failed (err={grasp_err:.4f}m), skipping arm")
            continue
        print(f"  Grasp IK: err={grasp_err:.4f}m, {time.time()-t0:.1f}s")

        optimizer = TrajectoryOptimizer(chain=chain, capsule_model=capsule_model,
                                        joint_limits=joint_limits, device=device)
        if table_plane is not None:
            optimizer.set_table(table_plane, table_polygon)

        # --- Lift phase: raise the arm vertically above the table first ---
        # Solve IK at several intermediate heights and interpolate between
        # consecutive solutions.  This gives a Cartesian-linear vertical
        # path that avoids table collisions (joint-space linear goes
        # sideways through the table).
        segments = []
        q_phase1_start = current_joints
        if table_plane is not None:
            with torch.no_grad():
                home_fk = ik_chain.forward_kinematics(current_joints.unsqueeze(0))
                home_ee = home_fk.get_matrix()[0, :3, 3]  # (3,)
            clearance_z = table_plane[0] + 0.15  # 150mm above table
            if home_ee[2].item() < clearance_z:
                print(f"  Lift phase: raising EE from z={home_ee[2].item():.3f}m "
                      f"to z={clearance_z:.3f}m")
                # Solve IK at intermediate heights every ~50mm
                z_start = home_ee[2].item()
                n_ik = max(3, int((clearance_z - z_start) / 0.05) + 1)
                z_vals = np.linspace(z_start, clearance_z, n_ik)
                ik_waypoints = [current_joints]
                lift_ok = True
                for z_i in z_vals[1:]:
                    pos_i = home_ee.clone()
                    pos_i[2] = z_i
                    q_i, err_i = _ik_batch_adam(
                        ik_chain, pos_i, None,
                        ik_waypoints[-1], lower, upper, device,
                        num_seeds=16, max_iters=500, rot_weight=0.0,
                    )
                    if err_i > IK_TOL:
                        print(f"  Lift IK failed at z={z_i:.3f}m (err={err_i:.4f}m)")
                        lift_ok = False
                        break
                    ik_waypoints.append(q_i)

                if lift_ok:
                    # Interpolate between consecutive IK solutions
                    steps_per_seg = max(2, (NUM_WAYPOINTS // 8) // len(ik_waypoints))
                    lift_parts = []
                    for k in range(len(ik_waypoints) - 1):
                        t_frac = torch.linspace(0, 1, steps_per_seg + 1, device=device).unsqueeze(1)
                        seg = ik_waypoints[k] + t_frac * (ik_waypoints[k + 1] - ik_waypoints[k])
                        lift_parts.append(seg if k == 0 else seg[1:])  # skip duplicate
                    traj_lift = torch.cat(lift_parts, dim=0).cpu()
                    print(f"  Lift: {traj_lift.shape[0]} waypoints ({n_ik} IK solutions)")
                    segments.append(traj_lift)
                    q_phase1_start = ik_waypoints[-1]

        t0 = time.time()
        traj1, cost1 = optimizer.optimize(q_start=q_phase1_start, q_goal=q_pregrasp,
                                           point_cloud=pc_tensor, T=NUM_WAYPOINTS,
                                           num_seeds=NUM_SEEDS, max_iters=MAX_ITERS)
        print(f"  Phase 1 (approach): cost={cost1:.4f}, {time.time()-t0:.1f}s")

        # Phase 2: linear descent from pre-grasp to grasp (no optimizer —
        # the gripper is supposed to be near the object here)
        DESCEND_STEPS = 10
        t_frac = torch.linspace(0, 1, DESCEND_STEPS, device=q_pregrasp.device).unsqueeze(1)
        traj2 = q_pregrasp + t_frac * (q_grasp - q_pregrasp)
        traj2 = traj2.cpu()
        print(f"  Phase 2 (descend): {DESCEND_STEPS} linear waypoints (pre-grasp → grasp)")

        # Assemble full trajectory
        segments.append(traj1)
        segments.append(traj2[1:])  # skip duplicate waypoint at junction
        # Remove duplicate waypoints at segment junctions
        assembled = [segments[0]]
        for seg in segments[1:]:
            assembled.append(seg[1:])  # skip first wp (= last of previous)
        full_traj = torch.cat(assembled, dim=0)

        # Smooth junction regions
        def _smooth_junction(traj, junction_idx, radius=5, passes=5):
            jstart = max(1, junction_idx - radius)
            jend = min(traj.shape[0] - 1, junction_idx + radius)
            for _ in range(passes):
                for i in range(jstart, jend):
                    traj[i] = (traj[i - 1] + traj[i + 1]) / 2.0

        # Find junction points
        wp_offset = 0
        for k, seg in enumerate(segments[:-1]):
            wp_offset += seg.shape[0] - 1  # -1 because we skip duplicates
            _smooth_junction(full_traj, wp_offset)

        total_wp = full_traj.shape[0]
        total_cost = cost1
        descend_start = total_wp - DESCEND_STEPS  # first descend waypoint index

        collisions = validate_trajectory(
            ik_chain, capsule_model, full_traj, pc_tensor, SAFETY_MARGIN,
            table_plane=optimizer.table_plane, table_polygon=table_polygon,
        )
        # Filter: skip t=0 (home position before lift) and the descend phase
        # (linear pre-grasp→grasp where gripper IS near the object).  Only
        # flag table collisions in the descend region (hard surface).
        hard_collisions = [
            c for c in collisions
            if c[0] > 0 and (c[0] < descend_start or "[table]" in c[1])
        ]
        soft_collisions = [c for c in collisions if c not in hard_collisions]
        if soft_collisions:
            n_soft = len(set(c[0] for c in soft_collisions))
            print(f"  (Ignoring {n_soft} grasp-endpoint point-cloud margin violations)")
        _log_collision_check(hard_collisions, total_wp)
        collisions = hard_collisions

        # Upsample via cubic spline for smoother execution
        from scipy.interpolate import CubicSpline
        traj_coarse = full_traj.cpu().numpy().astype(np.float64)
        n_coarse = traj_coarse.shape[0]
        n_fine = (n_coarse - 1) * UPSAMPLE_FACTOR + 1
        t_coarse = np.linspace(0, 1, n_coarse)
        t_fine = np.linspace(0, 1, n_fine)
        cs = CubicSpline(t_coarse, traj_coarse, bc_type='clamped')
        traj_fine = cs(t_fine)
        # Clamp to joint limits
        lower_np = lower.cpu().numpy()
        upper_np = upper.cpu().numpy()
        traj_fine = np.clip(traj_fine, lower_np, upper_np)
        print(f"  Upsampled: {n_coarse} → {n_fine} waypoints (×{UPSAMPLE_FACTOR})")
        traj_np = traj_fine.astype(np.float32)
        result = GraspResult(
            label=label,
            arm=arm_name,
            trajectory=traj_np,
            grasp_xyzrpy=list(grasp_xyzrpy),
            pregrasp_xyzrpy=list(pregrasp_xyzrpy),
            ik_pre_err=float(pre_err),
            ik_grasp_err=float(grasp_err),
            cost=float(total_cost),
            collisions=collisions,
            object_width=object_width,
            pc_robot=pc_robot,
            table_corners=table_corners_robot,
        )

        if not collisions:
            print(f"  {arm_name} arm: collision-free!")
            return result

        # Keep best result so far (fewest collisions)
        if best_result is None or len(collisions) < len(best_result.collisions):
            best_result = result

    if best_result is not None:
        n = len(set(c[0] for c in best_result.collisions))
        print(f"\n  WARNING: No collision-free arm found. Using {best_result.arm} arm ({n} colliding waypoints)")
    return best_result


# ---- Visualization ----

def _cylinder_mesh(p0, p1, radius, n_facets=8):
    """Generate a cylinder surface mesh between two 3D points."""
    axis = p1 - p0
    length = np.linalg.norm(axis)
    if length < 1e-8:
        return None, None, None
    axis_hat = axis / length
    if abs(axis_hat[0]) < 0.9:
        perp = np.cross(axis_hat, [1, 0, 0])
    else:
        perp = np.cross(axis_hat, [0, 1, 0])
    perp /= np.linalg.norm(perp)
    perp2 = np.cross(axis_hat, perp)
    theta = np.linspace(0, 2 * np.pi, n_facets + 1)
    t = np.array([0.0, 1.0])
    theta_grid, t_grid = np.meshgrid(theta, t)
    circle = radius * (np.cos(theta_grid)[..., None] * perp + np.sin(theta_grid)[..., None] * perp2)
    center = p0 + t_grid[..., None] * axis
    pts = center + circle
    return pts[..., 0], pts[..., 1], pts[..., 2]


def _box_faces(corners):
    """Generate surface patches for the 6 faces of a box from 8 corners."""
    face_indices = [
        [0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4],
        [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5],
    ]
    faces = []
    for idx in face_indices:
        face = corners[idx].reshape(2, 2, 3)
        faces.append(face)
    return faces


def precompute_frame(ik_chain, capsule_model, q, device):
    """Compute capsule meshes, gripper boxes, skeleton, and EE position for one waypoint."""
    with torch.no_grad():
        transforms = ik_chain.forward_kinematics(q.unsqueeze(0).to(device), end_only=False)

    # Capsules
    capsules = []
    for link_name in capsule_model.link_names:
        if link_name not in transforms:
            continue
        p0_w, p1_w = capsule_model.capsule_endpoints_world(link_name, transforms)
        p0 = p0_w[0].cpu().numpy()
        p1 = p1_w[0].cpu().numpy()
        radius = capsule_model.radii[link_name]
        result = _cylinder_mesh(p0, p1, radius)
        if result[0] is not None:
            capsules.append(result)

    # Gripper boxes
    boxes = []
    for box_name in capsule_model.box_link_names:
        parent = capsule_model.box_parent_link[box_name]
        if parent not in transforms:
            continue
        tf = transforms[parent]
        mat = tf.get_matrix()[0].cpu().numpy()
        R = mat[:3, :3]
        t = mat[:3, 3]
        c = capsule_model.box_centers[box_name].cpu().numpy()
        h = capsule_model.box_half_extents[box_name].cpu().numpy()
        signs = np.array([
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
        ], dtype=np.float32)
        corners_local = c + signs * h
        corners_world = (R @ corners_local.T).T + t
        boxes.append(corners_world)

    # Skeleton
    link_positions = {}
    for name, tf in transforms.items():
        pos = tf.get_matrix()[0, :3, 3].cpu().numpy()
        link_positions[name] = pos
    ordered = sorted(
        [(k, v) for k, v in link_positions.items() if "link" in k.lower()],
        key=lambda x: x[0],
    )
    skeleton = np.array([v for _, v in ordered]) if len(ordered) >= 2 else None

    # EE position
    with torch.no_grad():
        fk = ik_chain.forward_kinematics(q.unsqueeze(0))
        ee_pos = fk.get_matrix()[0, :3, 3].cpu().numpy()

    return capsules, boxes, skeleton, ee_pos


def render_trajectory_gif(
    result, urdf_path, device, output_gif, output_png,
):
    """Render a 4-view animated GIF + final-frame PNG of the trajectory."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    arm = result.arm
    traj_np = result.trajectory
    pc_robot = result.pc_robot
    table_corners = result.table_corners
    num_wp = traj_np.shape[0]

    # Build chain and capsule model (on CPU for viz)
    cpu = torch.device("cpu")
    if arm == "right":
        ee_link = RIGHT_EE
        capsule_defs = OPENARM_RIGHT_CAPSULES
    else:
        ee_link = LEFT_EE
        capsule_defs = OPENARM_CAPSULES

    chain = build_chain(urdf_path, ee_link)
    ik_chain = chain.to(dtype=torch.float32, device="cpu")
    try:
        capsule_model = CapsuleCollisionModel(capsule_defs, cpu, boxes=OPENARM_GRIPPER_BOXES)
    except Exception:
        capsule_model = CapsuleCollisionModel(capsule_defs, cpu)

    traj_tensor = torch.tensor(traj_np, dtype=torch.float32, device=cpu)
    colliding_indices = set(c[0] for c in result.collisions)

    # Precompute all frames
    print(f"  Precomputing {num_wp} frames...")
    frames_data = []
    ee_trail = []
    for i in range(num_wp):
        q = traj_tensor[i]
        capsules, boxes, skeleton, ee_pos = precompute_frame(ik_chain, capsule_model, q, cpu)
        ee_trail.append(ee_pos)
        frames_data.append((capsules, boxes, skeleton, ee_pos, i in colliding_indices))
    ee_trail = np.array(ee_trail)

    # Figure setup
    fig = plt.figure(figsize=(20, 16))
    views = [
        (1, "Top", 90, -90),
        (2, "Front", 0, -45),
        (3, "Side", 25, -60),
        (4, "Right", 0, 45),
    ]
    axes = []
    for idx, title, elev, azim in views:
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        ax.view_init(elev=elev, azim=azim)
        ax._view_title = title
        axes.append(ax)

    # Point cloud for viz (subsample to 5000 points)
    if pc_robot is not None and len(pc_robot) > 0:
        stride_viz = max(1, len(pc_robot) // 5000)
        pc_viz = pc_robot[::stride_viz]
    else:
        pc_viz = None

    # Axis limits
    pts_for_bounds = [ee_trail]
    if pc_viz is not None:
        pts_for_bounds.append(pc_viz)
    all_pts = np.vstack(pts_for_bounds)
    center = all_pts.mean(axis=0)
    max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 * 1.1

    cmap = plt.cm.coolwarm

    def draw_frame(frame_idx):
        capsules, boxes, skeleton, ee_pos, is_colliding = frames_data[frame_idx]
        t_frac = frame_idx / max(1, num_wp - 1)
        color = cmap(t_frac)
        arm_color = "red" if is_colliding else color
        status = "COLLISION" if is_colliding else "clear"

        for ax in axes:
            elev, azim = ax.elev, ax.azim
            view_title = ax._view_title
            ax.cla()
            ax.view_init(elev=elev, azim=azim)

            if pc_viz is not None:
                ax.scatter(pc_viz[:, 0], pc_viz[:, 1], pc_viz[:, 2],
                           c="gray", s=0.5, alpha=0.1)

            # Table plane quadrilateral
            if table_corners is not None:
                tc = table_corners
                outline = np.vstack([tc, tc[:1]])
                ax.plot3D(outline[:, 0], outline[:, 1], outline[:, 2],
                          "-", color="saddlebrown", linewidth=2, alpha=1.0)
                poly = Poly3DCollection([tc], alpha=0.3, facecolor="burlywood",
                                        edgecolor="saddlebrown", linewidth=1.5)
                ax.add_collection3d(poly)

            if frame_idx > 0:
                trail = ee_trail[:frame_idx + 1]
                ax.plot3D(trail[:, 0], trail[:, 1], trail[:, 2],
                          "--", color="black", linewidth=1, alpha=0.5)

            for X, Y, Z in capsules:
                ax.plot_surface(X, Y, Z, color=arm_color, alpha=0.7, shade=True)

            for corners in boxes:
                for face in _box_faces(corners):
                    ax.plot_surface(
                        face[..., 0], face[..., 1], face[..., 2],
                        color=arm_color, alpha=0.5, shade=True,
                    )

            if skeleton is not None:
                ax.plot3D(skeleton[:, 0], skeleton[:, 1], skeleton[:, 2],
                          "o-", color="black", alpha=0.8, markersize=2, linewidth=1)

            mk = "x" if is_colliding else "o"
            ax.scatter(*ee_pos, c="red" if is_colliding else "black",
                       s=40, marker=mk, zorder=10)
            ax.scatter(*ee_trail[0], c="blue", s=50, marker="^", zorder=10)
            ax.scatter(*ee_trail[-1], c="red", s=50, marker="v", zorder=10)

            ax.set_xlim(center[0] - max_range, center[0] + max_range)
            ax.set_ylim(center[1] - max_range, center[1] + max_range)
            ax.set_zlim(center[2] - max_range, center[2] + max_range)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(view_title, fontsize=11)

        fig.suptitle(
            f"{result.label} ({arm} arm)  |  "
            f"Waypoint {frame_idx}/{num_wp-1} [{status}]  |  "
            f"{len(colliding_indices)} colliding",
            fontsize=13,
        )

    # Subsample frames for GIF (target ~30 frames max for speed)
    max_gif_frames = 30
    if num_wp > max_gif_frames:
        step = num_wp // max_gif_frames
        gif_indices = list(range(0, num_wp, step))
        if gif_indices[-1] != num_wp - 1:
            gif_indices.append(num_wp - 1)
    else:
        gif_indices = list(range(num_wp))

    print(f"  Rendering {len(gif_indices)}/{num_wp} frames...")
    anim = FuncAnimation(fig, draw_frame, frames=gif_indices, interval=200, repeat=True)

    output_gif.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(output_gif), writer="pillow", fps=6, dpi=80)
    print(f"  Saved GIF: {output_gif}")

    draw_frame(num_wp - 1)
    fig.savefig(str(output_png), dpi=120, bbox_inches="tight")
    print(f"  Saved PNG: {output_png}")
    plt.close(fig)


# ---- Output saving ----

def save_result(result, output_dir, camera_transform_str):
    """Save trajectory JSON for a single GraspResult."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{result.label}_trajectory.json"

    # Gripper: open throughout, close proportionally to object width at the end.
    # Total jaw opening = 88mm (44mm per finger). Gripper value 0.0=fully open,
    # 1.0=fully closed. Close to object width minus a small grip margin.
    GRIPPER_MAX_OPENING = 0.088  # 88mm total (2 × 44mm per finger)
    GRIP_MARGIN = 0.005          # 5mm squeeze past object surface
    num_wp = result.trajectory.shape[0]
    gripper = np.zeros(num_wp, dtype=np.float32)
    # How much to close: object_width is what we need to reach, minus margin
    close_to = max(0.0, result.object_width - GRIP_MARGIN)
    # Gripper value = fraction of travel from open to closed
    grip_close = 1.0 - (close_to / GRIPPER_MAX_OPENING)
    grip_close = np.clip(grip_close, 0.0, 1.0)
    gripper[-1] = grip_close
    print(f"  Gripper: object={result.object_width*1000:.0f}mm, "
          f"close to {close_to*1000:.0f}mm opening, "
          f"grip={grip_close:.2f} (0=open, 1=closed)")

    save_trajectory(
        str(out_path), result.trajectory,
        arm=result.arm, dt=0.1, kp=30.0, kd=1.0,
        gripper=gripper,
        extra_metadata={
            "target": result.label,
            "camera_transform": camera_transform_str,
            "ik_tolerance": IK_TOL,
            "pre_grasp_err": result.ik_pre_err,
            "grasp_err": result.ik_grasp_err,
            "total_cost": result.cost,
            "num_collisions": len(result.collisions),
            "grasp_xyzrpy": [float(v) for v in result.grasp_xyzrpy],
            "pregrasp_xyzrpy": [float(v) for v in result.pregrasp_xyzrpy],
        },
    )
    print(f"  Trajectory saved: {out_path}")
    return out_path


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(
        description="General-purpose grasp trajectory planner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True,
                        help="Path to openarm-config v3 JSON")
    parser.add_argument("--targets", required=True,
                        help="Target JSON file path or inline JSON string")
    parser.add_argument("--robot", default=None,
                        help="Arm pair label (for logging; arm selection is by IK error)")
    parser.add_argument("--camera", default=None,
                        help="RealSense label (first enabled if omitted)")
    parser.add_argument("--output", default="./output",
                        help="Output directory (default: ./output)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Compute device (default: cuda)")
    parser.add_argument("--no-gif", action="store_true",
                        help="Skip GIF rendering")
    parser.add_argument("--urdf", default=None,
                        help="URDF path override")
    args = parser.parse_args()

    # Resolve config
    print("=== Grasp Trajectory Planner ===\n")
    config = load_config(args.config)
    camera_result = resolve_camera(config, args.camera)
    if camera_result is None:
        print("ERROR: Could not resolve camera from config")
        sys.exit(1)
    camera_transform_str, relay_path = camera_result

    robot_pair = resolve_robot(config, args.robot, camera_relay_path=relay_path)

    # Resolve paths and device
    urdf_path = Path(args.urdf) if args.urdf else DEFAULT_URDF
    output_dir = Path(args.output)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"URDF: {urdf_path}")

    # Load targets
    targets = load_targets(args.targets)
    print(f"\nTargets: {len(targets)}")
    for t in targets:
        print(f"  - {t['label']}: p1={t['p1']}, p2={t['p2']}")

    # Capture depth (once for all targets)
    print(f"\n[1/{len(targets)+1}] Capturing depth from RealSense...")
    depth, intrinsics = capture_depth(relay_path)
    cam_t, cam_rot = parse_camera_transform(camera_transform_str)

    # Adjust camera translation to arm-local frame
    if robot_pair is not None:
        offset = arm_offset_urdf(robot_pair)
        if np.any(offset != 0):
            print(f"[config] Arm offset (URDF): [{offset[0]:.3f}, {offset[1]:.3f}, {offset[2]:.3f}]")
            cam_t = cam_t - offset
            print(f"[config] Adjusted cam_t: [{cam_t[0]:.3f}, {cam_t[1]:.3f}, {cam_t[2]:.3f}]")

    # Plan each target
    results = []
    for i, target in enumerate(targets):
        step = i + 2
        print(f"\n[{step}/{len(targets)+1}] Planning '{target['label']}'...")
        result = plan_single_target(
            target, depth, intrinsics, cam_t, cam_rot, device, urdf_path,
        )
        if result is None:
            print(f"  SKIPPED: planning failed for '{target['label']}'")
            continue
        results.append(result)

        # Save trajectory JSON
        save_result(result, output_dir, camera_transform_str)

        # Render GIF
        if not args.no_gif:
            gif_path = output_dir / f"{result.label}_trajectory.gif"
            png_path = output_dir / f"{result.label}_final.png"
            render_trajectory_gif(result, urdf_path, device, gif_path, png_path)

    # Summary
    print(f"\n=== Done! ===")
    print(f"  Planned: {len(results)}/{len(targets)} targets")
    for r in results:
        coll_str = f", {len(r.collisions)} collisions" if r.collisions else ""
        print(f"  - {r.label}: {r.arm} arm, {r.trajectory.shape[0]} waypoints, "
              f"cost={r.cost:.4f}{coll_str}")
    print(f"  Output: {output_dir}/")


if __name__ == "__main__":
    main()
