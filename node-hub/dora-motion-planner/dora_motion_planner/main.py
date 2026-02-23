"""Dora node: collision-aware motion planning + playback for OpenArm.

Plans collision-free trajectories using gradient-based optimisation with
differentiable FK and capsule collision costs.  Includes a built-in
trajectory player: after planning, the node plays back waypoints as
``joint_command`` on each ``tick``.

Supported target_pose encodings:
    jointstate:  float32 array of joint angles (7,)
    xyzrpy:      float32 array [x, y, z, roll, pitch, yaw] (6,)
    xyzquat:     float32 array [x, y, z, qw, qx, qy, qz] (7,)
    jaw_pixels:  float32 array [u1, v1, u2, v2] — two gripper jaw positions
                 on the RGBD image.  The node deprojects them to 3D, computes
                 a top-down grasp pose, and plans a 2-phase trajectory:
                 pre-grasp (approach from above) then grasp.

Additional input:
    grasp_result: JSON string from grasp_selector.py, e.g.
                  {"p1":[x,y],"p2":[x,y]} in 0-1000 normalized coords.
                  Converted to pixel coords and handled as jaw_pixels.

Env vars:
    URDF_PATH:            Path to the URDF file.
    LEFT_END_EFFECTOR_LINK:  Left arm end-effector link (default "openarm_left_hand_tcp").
    RIGHT_END_EFFECTOR_LINK: Right arm end-effector link (default "openarm_right_hand_tcp").
    END_EFFECTOR_LINK:    Backward-compat alias — sets LEFT_END_EFFECTOR_LINK if the new
                          var is not set.
    CAMERA_TRANSFORM:     6 values "x y z roll pitch yaw" in degrees (extrinsic XYZ),
                          or 7 values "tx ty tz qw qx qy qz" (quaternion).  Default identity.
    DOWNSAMPLE_STRIDE:    Depth image downsample factor (default 8).
    SAFETY_MARGIN:        Collision margin in metres (default 0.02).
    NUM_WAYPOINTS:        Trajectory waypoints (default 200).
    NUM_SEEDS:            Multi-start seeds (default 8).
    MAX_ITERS:            Adam iterations per seed (default 500).
    DEVICE:               "cuda" or "cpu" (default "cuda").
    GRASP_DEPTH_OFFSET:   Z offset from object surface for grasp (default -0.01m).
    FLOOR_HEIGHT:         Minimum grasp Z in robot frame (default 0.005m).
    APPROACH_MARGIN:      Height above grasp for pre-grasp waypoint (default 0.03m).
    EXPORT_PATH:          If set, auto-save every planned trajectory to this JSON path.
    MAX_JOINT_STEP:       Max per-step joint angle change in radians (default 0.05).
    PLAYBACK:             "true"/"false" — enable built-in playback on tick (default "true").
"""

import json
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytorch_kinematics as pk
import torch
from dora import Node

from .collision_model import (
    CapsuleCollisionModel,
    OPENARM_CAPSULES,
    OPENARM_RIGHT_CAPSULES,
    OPENARM_GRIPPER_BOXES,
    OPENARM_RIGHT_GRIPPER_BOXES,
    capsule_points_distance,
    capsule_halfplane_distance,
    box_points_distance,
)
from .grasp_utils import grasp_pose_from_jaw_pixels
from .pointcloud import (
    compute_table_plane,
    depth_to_pointcloud,
    parse_camera_transform,
    transform_points,
    pointcloud_to_tensor,
)
from .trajectory_json import save as save_trajectory_json, load as load_trajectory_json
from .trajectory_optimizer import TrajectoryOptimizer

# Configuration from env vars
URDF_PATH = os.getenv("URDF_PATH", "")
_LEGACY_EE_LINK = os.getenv("END_EFFECTOR_LINK", "")
LEFT_END_EFFECTOR_LINK = os.getenv(
    "LEFT_END_EFFECTOR_LINK", _LEGACY_EE_LINK or "openarm_left_hand_tcp"
)
RIGHT_END_EFFECTOR_LINK = os.getenv(
    "RIGHT_END_EFFECTOR_LINK", "openarm_right_hand_tcp"
)
CAMERA_TRANSFORM_STR = os.getenv("CAMERA_TRANSFORM", "0.79 0.81 1.23 90 -45 0")
DOWNSAMPLE_STRIDE = int(os.getenv("DOWNSAMPLE_STRIDE", "8"))
SAFETY_MARGIN = float(os.getenv("SAFETY_MARGIN", "0.02"))
NUM_WAYPOINTS = int(os.getenv("NUM_WAYPOINTS", "200"))
NUM_SEEDS = int(os.getenv("NUM_SEEDS", "8"))
MAX_ITERS = int(os.getenv("MAX_ITERS", "200"))
DEVICE = os.getenv("DEVICE", "cuda")
GRASP_DEPTH_OFFSET = float(os.getenv("GRASP_DEPTH_OFFSET", "-0.01"))
FLOOR_HEIGHT = float(os.getenv("FLOOR_HEIGHT", "0.005"))
APPROACH_MARGIN = float(os.getenv("APPROACH_MARGIN", "0.03"))
IMAGE_WIDTH = int(os.getenv("IMAGE_WIDTH", "640"))
IMAGE_HEIGHT = int(os.getenv("IMAGE_HEIGHT", "480"))
MAX_JOINT_STEP = float(os.getenv("MAX_JOINT_STEP", "0.1"))
EXPORT_PATH = os.getenv("EXPORT_PATH", "")
PLAYBACK = os.getenv("PLAYBACK", "true").lower() in ("1", "true", "yes")


def build_chain(urdf_path: str, end_effector_link: str) -> pk.Chain:
    """Build a serial kinematic chain from URDF, stripping visual/collision meshes."""
    with open(urdf_path) as f:
        urdf_str = f.read()
    root = ET.fromstring(urdf_str)
    for link in root.findall(".//link"):
        for tag in ("visual", "collision", "inertial"):
            for elem in link.findall(tag):
                link.remove(elem)
    clean_urdf = ET.tostring(root, encoding="unicode")
    return pk.build_serial_chain_from_urdf(clean_urdf, end_effector_link)


@dataclass
class ArmConfig:
    """Per-arm state: kinematic chain, optimizer, and current joint positions."""

    name: str
    ik_chain: pk.Chain
    optimizer: TrajectoryOptimizer
    joint_limits: tuple
    num_joints: int
    current_joints: torch.Tensor


def _init_arm(name, urdf_path, ee_link, capsules, device, boxes=None):
    """Build chain, capsule model, optimizer, and initial state for one arm."""
    chain = build_chain(urdf_path, ee_link)
    joint_limits = chain.get_joint_limits()
    num_joints = len(chain.get_joint_parameter_names(exclude_fixed=True))

    capsule_model = CapsuleCollisionModel(capsules, device, boxes=boxes)
    optimizer = TrajectoryOptimizer(
        chain=chain,
        capsule_model=capsule_model,
        joint_limits=joint_limits,
        device=device,
        collision_alpha=50.0,
        max_joint_step=MAX_JOINT_STEP,
    )
    ik_chain = chain.to(dtype=torch.float32, device=str(device))
    current_joints = torch.zeros(num_joints, dtype=torch.float32, device=device)

    with torch.no_grad():
        home_fk = ik_chain.forward_kinematics(
            torch.zeros(1, num_joints, device=device)
        )
        home_pos = home_fk.get_matrix()[0, :3, 3].cpu().numpy()
        print(f"[motion-planner] {name} arm: {num_joints} joints, EE={ee_link}")
        print(
            f"[motion-planner] {name} home EE: "
            f"[{home_pos[0]:.4f}, {home_pos[1]:.4f}, {home_pos[2]:.4f}]"
        )
        print(
            f"[motion-planner] {name} joint limits: "
            f"{[[f'{v:.2f}' for v in lim] for lim in joint_limits]}"
        )

        # Verify collision primitive link names match FK transform keys
        all_transforms = ik_chain.forward_kinematics(
            torch.zeros(1, num_joints, device=device), end_only=False
        )
        fk_keys = set(all_transforms.keys())
        capsule_keys = set(capsules.keys())
        # Box parent links are what we actually look up in FK transforms
        box_parent_keys = set(
            b.parent_link or name for name, b in boxes.items()
        ) if boxes else set()
        all_prim_keys = capsule_keys | box_parent_keys
        matched = all_prim_keys & fk_keys
        missing = all_prim_keys - fk_keys
        if missing:
            print(
                f"[motion-planner] WARNING: {name} collision links NOT in FK transforms: "
                f"{missing} — collision avoidance will be DISABLED for these links"
            )
            print(f"[motion-planner] FK transform keys: {sorted(fk_keys)}")
        else:
            n_boxes = len(boxes) if boxes else 0
            print(
                f"[motion-planner] {name} collision model: "
                f"{len(capsule_keys)} capsules + {n_boxes} boxes, "
                f"all {len(matched)} FK links matched"
            )

    return ArmConfig(name, ik_chain, optimizer, joint_limits, num_joints, current_joints)


def select_arm(metadata, target_y=None):
    """Select arm: metadata ``arm`` field > Y-position sign > default left."""
    arm = metadata.get("arm", "")
    if arm in ("left", "right"):
        return arm
    if target_y is not None:
        return "left" if target_y > 0 else "right"
    return "left"


def _estimate_target_y(u1, v1, u2, v2, depth, intrinsics, image_size, cam_t, cam_rot):
    """Estimate robot-frame Y of a grasp target from the jaw-pixel midpoint."""
    if depth is None or intrinsics is None:
        return None
    mid_u = (u1 + u2) / 2
    mid_v = (v1 + v2) / 2
    w, h = image_size
    fx, fy, cx, cy = intrinsics
    iu = int(np.clip(mid_u, 0, w - 1))
    iv = int(np.clip(mid_v, 0, h - 1))
    depth_2d = depth.reshape(h, w) if depth.ndim == 1 else depth
    z_mm = float(depth_2d[iv, iu])
    if z_mm <= 0:
        return None
    z = z_mm * 0.001
    x_cam = (mid_u - cx) * z / fx
    y_cam = (mid_v - cy) * z / fy
    pt_robot = cam_rot @ np.array([x_cam, y_cam, z]) + cam_t
    return float(pt_robot[1])


def _ik_batch_adam(
    ik_chain,
    target_pos,
    target_rot,
    current_joints,
    lower,
    upper,
    device,
    num_seeds=48,
    max_iters=2000,
    rot_weight=0.0,
):
    """Batched Adam IK: all seeds optimised in parallel.

    Returns (best_q, best_pos_err) or (None, err) if no seed converges.
    """
    nj = len(lower)
    q_batch = lower + (upper - lower) * torch.rand(num_seeds, nj, device=device)
    q_batch[0] = current_joints.clone()
    q_batch = q_batch.clone().detach().requires_grad_(True)

    opt = torch.optim.Adam([q_batch], lr=0.02)
    target_pos_b = target_pos.unsqueeze(0).expand(num_seeds, -1)

    for it in range(max_iters):
        opt.zero_grad()
        q_c = torch.clamp(q_batch, lower, upper)
        fk = ik_chain.forward_kinematics(q_c)
        fk_pos = fk.get_matrix()[:, :3, 3]
        pos_loss = ((fk_pos - target_pos_b) ** 2).sum(dim=1)
        loss = pos_loss.sum()

        if rot_weight > 0 and target_rot is not None and it >= max_iters // 2:
            fk_rot = fk.get_matrix()[:, :3, :3]
            eye = torch.eye(3, device=device).unsqueeze(0)
            rot_diff = eye - torch.bmm(
                fk_rot.transpose(1, 2),
                target_rot.unsqueeze(0).expand(num_seeds, -1, -1),
            )
            ramp = min(1.0, (it - max_iters // 2) / (max_iters // 4))
            loss = loss + rot_weight * ramp * (rot_diff**2).sum()

        loss.backward()
        opt.step()
        with torch.no_grad():
            q_batch.clamp_(lower, upper)

    with torch.no_grad():
        q_c = torch.clamp(q_batch, lower, upper)
        fk = ik_chain.forward_kinematics(q_c)
        fk_pos = fk.get_matrix()[:, :3, 3]
        pos_errs = (fk_pos - target_pos_b).norm(dim=1)
        # Among seeds with similar accuracy (within 5mm of best),
        # prefer the one closest to current_joints to avoid unnecessary rotation.
        best_err = pos_errs.min()
        good_mask = pos_errs < best_err + 0.005
        joint_dists = (q_c - current_joints.unsqueeze(0)).norm(dim=1)
        joint_dists[~good_mask] = float("inf")
        best_idx = int(joint_dists.argmin())
        return q_c[best_idx], float(pos_errs[best_idx])


def solve_ik(
    ik_chain,
    target_xyzrpy,
    current_joints,
    joint_limits,
    device,
    num_seeds=48,
    max_iters=2000,
):
    """Solve IK via batched Adam gradient descent.

    Tries full pose IK first; if that fails, falls back to position-only
    (the arm reaches the right position with whatever orientation it can).

    Returns q_goal tensor or None.
    """
    t_ik_start = time.perf_counter()
    target_pos = torch.tensor(target_xyzrpy[:3], dtype=torch.float32, device=device)
    target_rot = pk.transforms.euler_angles_to_matrix(
        torch.tensor(target_xyzrpy[3:6], dtype=torch.float32, device=device),
        convention="XYZ",
    )
    lower = torch.tensor(joint_limits[0], dtype=torch.float32, device=device)
    upper = torch.tensor(joint_limits[1], dtype=torch.float32, device=device)

    # Try with soft orientation first
    q, err = _ik_batch_adam(
        ik_chain,
        target_pos,
        target_rot,
        current_joints,
        lower,
        upper,
        device,
        num_seeds,
        max_iters,
        rot_weight=0.1,
    )
    if err <= 0.02:
        t_ik = time.perf_counter() - t_ik_start
        print(f"[motion-planner] IK ok: pos_err={err:.4f}m (with orientation, {t_ik:.2f}s)")
        return q

    # Fallback: position-only
    q, err = _ik_batch_adam(
        ik_chain,
        target_pos,
        None,
        current_joints,
        lower,
        upper,
        device,
        num_seeds,
        max_iters,
        rot_weight=0.0,
    )
    t_ik = time.perf_counter() - t_ik_start
    if err <= 0.02:
        print(f"[motion-planner] IK ok: pos_err={err:.4f}m (position-only fallback, {t_ik:.2f}s)")
        return q

    print(f"[motion-planner] IK failed: pos_err={err:.4f}m (best of {num_seeds} seeds, {t_ik:.2f}s)")
    return None


def validate_trajectory(ik_chain, capsule_model, trajectory, point_cloud, margin,
                        table_plane=None, table_polygon=None):
    """Check trajectory for collisions with point cloud and table plane.

    Args:
        table_plane: Optional (plane_z, (x_min, x_max, y_min, y_max)) table surface.
        table_polygon: Optional (V, 2) tensor — rotated table polygon (overrides AABB).

    Returns list of (waypoint_idx, link_name, min_distance) for any waypoint
    where a capsule link is closer than *margin* to the point cloud or table.
    """
    collisions = []
    has_pc = point_cloud is not None and len(point_cloud) > 0

    if not has_pc and table_plane is None:
        return []

    device = point_cloud.device if has_pc else trajectory.device
    traj = trajectory.to(device)

    with torch.no_grad():
        transforms = ik_chain.forward_kinematics(traj, end_only=False)
        fk_keys = set(transforms.keys())
        capsule_links = [n for n in capsule_model.link_names if n in fk_keys]

        if len(capsule_links) == 0:
            print(
                "[motion-planner] Collision check: 0 capsule links found in FK "
                "transforms — collision avoidance is broken!"
            )
            print(f"[motion-planner]   capsule links: {capsule_model.link_names}")
            print(f"[motion-planner]   FK keys: {sorted(fk_keys)}")
            return []

        for link_name in capsule_links:
            p0_w, p1_w = capsule_model.capsule_endpoints_world(link_name, transforms)
            radius = capsule_model.radii[link_name]

            # Point cloud collisions
            if has_pc:
                sd = capsule_points_distance(p0_w, p1_w, radius, point_cloud)
                min_dist_per_t = sd.min(dim=1).values
                for t_idx in range(traj.shape[0]):
                    d = float(min_dist_per_t[t_idx])
                    if d < margin:
                        collisions.append((t_idx, link_name, d))

            # Table plane collisions
            if table_plane is not None:
                plane_z, bounds = table_plane
                sd_table = capsule_halfplane_distance(
                    p0_w, p1_w, radius, plane_z, bounds,
                    plane_polygon=table_polygon,
                )
                for t_idx in range(traj.shape[0]):
                    d = float(sd_table[t_idx])
                    if d < margin:
                        collisions.append((t_idx, f"{link_name}[table]", d))

        # Box collisions (gripper) — point cloud + table
        box_links = [
            n for n in capsule_model.box_link_names
            if capsule_model.box_parent_link[n] in fk_keys
        ]
        for link_name in box_links:
            parent = capsule_model.box_parent_link[link_name]
            if has_pc:
                sd = box_points_distance(
                    transforms,
                    parent,
                    capsule_model.box_centers[link_name],
                    capsule_model.box_half_extents[link_name],
                    point_cloud,
                )
                min_dist_per_t = sd.min(dim=1).values
                for t_idx in range(traj.shape[0]):
                    d = float(min_dist_per_t[t_idx])
                    if d < margin:
                        collisions.append((t_idx, link_name, d))

            # Table plane check for gripper boxes
            if table_plane is not None:
                plane_z, bounds = table_plane
                tf = transforms[parent]
                mat = tf.get_matrix()
                R = mat[:, :3, :3]
                t_vec = mat[:, :3, 3]
                center = capsule_model.box_centers[link_name]
                half_ext = capsule_model.box_half_extents[link_name]
                center_w = torch.einsum("bij,j->bi", R, center) + t_vec
                box_radius = half_ext.norm().item()
                sd_table = capsule_halfplane_distance(
                    center_w, center_w, box_radius, plane_z, bounds,
                    plane_polygon=table_polygon,
                )
                for t_idx in range(traj.shape[0]):
                    d = float(sd_table[t_idx])
                    if d < margin:
                        collisions.append((t_idx, f"{link_name}[table]", d))

    return collisions


def _log_collision_check(collisions, total_waypoints):
    """Log collision validation results."""
    if not collisions:
        print("[motion-planner] Collision check: trajectory clear")
        return
    colliding_waypoints = len(set(c[0] for c in collisions))
    worst = min(collisions, key=lambda c: c[2])
    print(
        f"[motion-planner] Collision check: {colliding_waypoints}/{total_waypoints} "
        f"waypoints in collision"
    )
    for t_idx, link_name, dist in collisions[:10]:
        print(f"[motion-planner]   t={t_idx} link={link_name} dist={dist:.4f}m")
    if len(collisions) > 10:
        print(f"[motion-planner]   ... and {len(collisions) - 10} more")
    print(
        f"[motion-planner]   worst: t={worst[0]} penetration={-worst[2]*1000:.1f}mm"
    )


def plan_and_send(node, optimizer, q_start, q_goal, pc_tensor, num_joints,
                   ik_chain=None, capsule_model=None, arm="left",
                   table_plane=None, table_polygon=None):
    """Run trajectory optimisation, send the result, and return it.

    Returns ``(traj_np, metadata)`` — the (T, J) float32 trajectory and its
    metadata dict, so the caller can store it for playback / export.
    """
    pc_info = f"{len(pc_tensor)} pts" if pc_tensor is not None else "no pc"
    print(
        f"[motion-planner] Planning: {NUM_WAYPOINTS} waypoints, "
        f"{NUM_SEEDS} seeds, {MAX_ITERS} iters, {pc_info}"
    )

    t0 = time.perf_counter()
    best_traj, best_cost = optimizer.optimize(
        q_start=q_start,
        q_goal=q_goal,
        point_cloud=pc_tensor,
        T=NUM_WAYPOINTS,
        num_seeds=NUM_SEEDS,
        max_iters=MAX_ITERS,
    )
    t_plan = time.perf_counter() - t0

    print(f"[motion-planner] Best cost: {best_cost:.4f} (planned in {t_plan:.2f}s)")

    # Post-optimization collision validation
    if ik_chain is not None and capsule_model is not None:
        _log_collision_check(
            validate_trajectory(
                ik_chain, capsule_model, best_traj, pc_tensor, SAFETY_MARGIN,
                table_plane=table_plane, table_polygon=table_polygon,
            ),
            best_traj.shape[0],
        )

    traj_np = best_traj.numpy().astype(np.float32)
    dt = 1.0 / 30.0
    out_metadata = {
        "num_waypoints": NUM_WAYPOINTS,
        "num_joints": num_joints,
        "dt": dt,
        "encoding": "trajectory",
        "arm": arm,
    }
    node.send_output(
        "joint_trajectory",
        pa.array(traj_np.ravel(), type=pa.float32()),
        metadata=out_metadata,
    )
    return traj_np, out_metadata


def plan_grasp_from_pixels(
    u1,
    v1,
    u2,
    v2,
    node,
    optimizer,
    ik_chain,
    capsule_model,
    joint_limits,
    device,
    current_joints,
    latest_depth,
    latest_intrinsics,
    latest_image_size,
    cam_t,
    cam_rot,
    pc_tensor,
    num_joints,
    arm="left",
    table_plane=None,
    table_polygon=None,
):
    """Plan a 2-phase grasp trajectory from two jaw pixel positions.

    Returns ``(traj_np, metadata)`` on success, or ``None`` on failure.
    """
    if latest_depth is None or latest_intrinsics is None:
        print("[motion-planner] grasp: no depth/intrinsics yet, skipping")
        return None

    w, h = latest_image_size
    fx, fy, cx, cy = latest_intrinsics

    result = grasp_pose_from_jaw_pixels(
        u1,
        v1,
        u2,
        v2,
        latest_depth,
        fx,
        fy,
        cx,
        cy,
        cam_t,
        cam_rot,
        width=w,
        height=h,
        grasp_depth_offset=GRASP_DEPTH_OFFSET,
        floor_height=FLOOR_HEIGHT,
        approach_margin=APPROACH_MARGIN,
    )
    if result is None:
        print("[motion-planner] grasp: invalid depth at jaw points, skipping")
        return None

    grasp_xyzrpy, pregrasp_xyzrpy = result
    print(
        f"[motion-planner] grasp: center={np.round(grasp_xyzrpy[:3], 4)}, "
        f"pregrasp_z={pregrasp_xyzrpy[2]:.4f}"
    )

    q_pregrasp = solve_ik(
        ik_chain, pregrasp_xyzrpy, current_joints, joint_limits, device
    )
    if q_pregrasp is None:
        print("[motion-planner] grasp: IK failed for pre-grasp, skipping")
        return None

    q_grasp = solve_ik(ik_chain, grasp_xyzrpy, q_pregrasp, joint_limits, device)
    if q_grasp is None:
        print("[motion-planner] grasp: IK failed for grasp, skipping")
        return None

    q_start = current_joints.clone()
    half_waypoints = NUM_WAYPOINTS // 2
    t_grasp_start = time.perf_counter()

    print("[motion-planner] Phase 1: start -> pre-grasp")
    traj1, cost1 = optimizer.optimize(
        q_start=q_start,
        q_goal=q_pregrasp,
        point_cloud=pc_tensor,
        T=half_waypoints,
        num_seeds=NUM_SEEDS,
        max_iters=MAX_ITERS,
    )

    print("[motion-planner] Phase 2: pre-grasp -> grasp")
    traj2, cost2 = optimizer.optimize(
        q_start=q_pregrasp,
        q_goal=q_grasp,
        point_cloud=pc_tensor,
        T=half_waypoints,
        num_seeds=NUM_SEEDS,
        max_iters=MAX_ITERS,
    )
    t_grasp = time.perf_counter() - t_grasp_start

    full_traj = torch.cat([traj1, traj2[1:]], dim=0)
    total_waypoints = full_traj.shape[0]
    print(
        f"[motion-planner] Combined: {total_waypoints} waypoints, "
        f"cost={cost1 + cost2:.4f} (2-phase planning: {t_grasp:.2f}s)"
    )

    # Post-optimization collision validation
    if capsule_model is not None:
        _log_collision_check(
            validate_trajectory(
                ik_chain, capsule_model, full_traj, pc_tensor, SAFETY_MARGIN,
                table_plane=table_plane, table_polygon=table_polygon,
            ),
            total_waypoints,
        )

    traj_np = full_traj.numpy().astype(np.float32)
    dt = 1.0 / 30.0
    out_metadata = {
        "num_waypoints": total_waypoints,
        "num_joints": num_joints,
        "dt": dt,
        "encoding": "trajectory",
        "arm": arm,
        "grasp_pose": grasp_xyzrpy.tolist(),
        "pregrasp_waypoint": half_waypoints - 1,
    }
    node.send_output(
        "joint_trajectory",
        pa.array(traj_np.ravel(), type=pa.float32()),
        metadata=out_metadata,
    )
    return traj_np, out_metadata


def build_pointcloud(
    latest_depth, latest_intrinsics, latest_image_size, cam_t, cam_rot, device
):
    """Build GPU point cloud tensor from latest depth, or None.

    Returns ``(pc_tensor, pc_robot)`` — the GPU tensor and the numpy
    array in robot frame (needed for table plane computation).
    """
    if latest_depth is None or latest_intrinsics is None:
        return None, None
    t0 = time.perf_counter()
    w, h = latest_image_size
    fx, fy, cx, cy = latest_intrinsics
    pc_cam = depth_to_pointcloud(
        latest_depth,
        fx,
        fy,
        cx,
        cy,
        width=w,
        height=h,
        stride=DOWNSAMPLE_STRIDE,
    )
    if len(pc_cam) == 0:
        return None, None
    pc_robot = transform_points(pc_cam, cam_t, cam_rot)
    pc_tensor = pointcloud_to_tensor(pc_robot, device)
    t_pc = time.perf_counter() - t0
    print(f"[motion-planner] Point cloud: {len(pc_robot)} pts ({t_pc:.3f}s)")
    return pc_tensor, pc_robot


def _setup_table_plane(optimizer, pc_robot, cam_t, cam_rot, latest_intrinsics,
                       latest_image_size, device):
    """Compute table plane from camera frustum and set on optimizer.

    Returns ``(table_plane, table_polygon)`` for passing to validation.
    """
    if pc_robot is None or latest_intrinsics is None:
        return None, None
    fx, fy, cx, cy = latest_intrinsics
    w, h = latest_image_size
    table_plane, table_polygon = compute_table_plane(
        cam_t, cam_rot, fx, fy, cx, cy, w, h, pc_robot, device=device,
    )
    optimizer.set_table(table_plane, table_polygon)
    plane_z, bounds = table_plane
    print(
        f"[motion-planner] Table plane: z={plane_z:.3f}, "
        f"bounds=({bounds[0]:.2f}, {bounds[1]:.2f}, {bounds[2]:.2f}, {bounds[3]:.2f})"
    )
    return table_plane, table_polygon


def _set_playback(state, traj_np, traj_meta):
    """Store a newly planned trajectory for playback and optionally export."""
    state["trajectory"] = traj_np                        # (T, J)
    state["step"] = 0
    state["playing"] = True
    state["play_arm"] = traj_meta.get("arm", "left")
    state["play_start"] = None                           # set on first tick
    state["play_dt"] = float(traj_meta.get("dt", 0.1))

    if EXPORT_PATH:
        extra = {}
        for k in ("grasp_pose", "pregrasp_waypoint", "encoding"):
            if k in traj_meta:
                extra[k] = traj_meta[k]
        save_trajectory_json(
            EXPORT_PATH,
            traj_np,
            arm=traj_meta.get("arm", "left"),
            dt=state["play_dt"],
            extra_metadata=extra,
        )
        print(f"[motion-planner] Exported trajectory → {EXPORT_PATH}")


def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[motion-planner] Device: {device}")

    urdf_path = URDF_PATH
    if not urdf_path or not Path(urdf_path).exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    # Initialise both arms
    arms: dict[str, ArmConfig] = {}

    left_arm = _init_arm(
        "left", urdf_path, LEFT_END_EFFECTOR_LINK, OPENARM_CAPSULES, device,
        boxes=OPENARM_GRIPPER_BOXES,
    )
    arms["left"] = left_arm

    try:
        right_arm = _init_arm(
            "right",
            urdf_path,
            RIGHT_END_EFFECTOR_LINK,
            OPENARM_RIGHT_CAPSULES,
            device,
            boxes=OPENARM_RIGHT_GRIPPER_BOXES,
        )
        arms["right"] = right_arm
    except Exception as e:
        print(f"[motion-planner] Right arm not available: {e}")

    cam_t, cam_rot = parse_camera_transform(CAMERA_TRANSFORM_STR)

    # Shared sensor state
    latest_depth = None
    latest_intrinsics = None
    latest_image_size = (IMAGE_WIDTH, IMAGE_HEIGHT)

    # Table plane state (computed once per depth frame, shared across arms)
    table_plane = None
    table_polygon = None

    # Playback state
    playback = {
        "trajectory": None,   # (T, J) numpy array
        "step": 0,
        "playing": False,
        "play_arm": "left",
        "play_start": None,   # monotonic timestamp of first tick
        "play_dt": 1.0 / 30.0,
    }

    node = Node()
    print(
        f"[motion-planner] Ready ({', '.join(arms.keys())} arm(s)), "
        f"playback={'on' if PLAYBACK else 'off'}, "
        f"export={'on → ' + EXPORT_PATH if EXPORT_PATH else 'off'}"
    )

    for event in node:
        if event["type"] == "INPUT":
            event_id = event["id"]
            metadata = event["metadata"]

            # --- Tick: playback current trajectory ---
            if event_id == "tick" and PLAYBACK:
                traj = playback["trajectory"]
                if traj is None or not playback["playing"]:
                    continue

                T = traj.shape[0]
                dt = playback["play_dt"]

                if playback["play_start"] is None:
                    playback["play_start"] = time.monotonic()
                    playback["step"] = 0
                    print("[motion-planner] Playback started")

                # Advance based on wall-clock time
                elapsed = time.monotonic() - playback["play_start"]
                step = min(int(elapsed / dt), T - 1)
                playback["step"] = step

                waypoint = traj[step]
                out_metadata = metadata.copy()
                out_metadata["encoding"] = "jointstate"
                out_metadata["arm"] = playback["play_arm"]
                node.send_output(
                    "joint_command",
                    pa.array(waypoint, type=pa.float32()),
                    metadata=out_metadata,
                )

                if step >= T - 1 and playback["playing"]:
                    playback["playing"] = False
                    print("[motion-planner] Playback complete, holding final position")

            # --- Depth / intrinsics (shared across arms) ---
            elif event_id == "depth":
                latest_depth = event["value"].to_numpy().astype(np.uint16)
                latest_image_size = (
                    int(metadata.get("width", IMAGE_WIDTH)),
                    int(metadata.get("height", IMAGE_HEIGHT)),
                )
                if "focal_length" in metadata and "resolution" in metadata:
                    fl = metadata["focal_length"]
                    pp = metadata["resolution"]
                    latest_intrinsics = (
                        float(fl[0]),
                        float(fl[1]),
                        float(pp[0]),
                        float(pp[1]),
                    )
                    if latest_depth is not None:
                        w, h = latest_image_size
                        print(
                            f"[motion-planner] intrinsics: fx={latest_intrinsics[0]:.1f} "
                            f"fy={latest_intrinsics[1]:.1f} cx={latest_intrinsics[2]:.1f} "
                            f"cy={latest_intrinsics[3]:.1f} image={w}x{h}"
                        )

            # --- Joint state updates ---
            elif event_id in ("joint_state", "left_joint_state"):
                arm = arms.get("left")
                if arm:
                    js = event["value"].to_numpy().astype(np.float32)
                    arm.current_joints = torch.tensor(
                        js[: arm.num_joints], dtype=torch.float32, device=device
                    )

            elif event_id == "right_joint_state":
                arm = arms.get("right")
                if arm:
                    js = event["value"].to_numpy().astype(np.float32)
                    arm.current_joints = torch.tensor(
                        js[: arm.num_joints], dtype=torch.float32, device=device
                    )

            # --- Target pose ---
            elif event_id == "target_pose":
                encoding = metadata.get("encoding", "jointstate")
                target_data = event["value"].to_numpy().astype(np.float32)

                # Determine target Y for automatic arm selection
                target_y = None
                if encoding in ("xyzrpy", "xyzquat"):
                    target_y = float(target_data[1])
                elif encoding == "jaw_pixels":
                    target_y = _estimate_target_y(
                        target_data[0],
                        target_data[1],
                        target_data[2],
                        target_data[3],
                        latest_depth,
                        latest_intrinsics,
                        latest_image_size,
                        cam_t,
                        cam_rot,
                    )

                arm_name = select_arm(metadata, target_y)
                arm = arms.get(arm_name)
                if arm is None:
                    print(
                        f"[motion-planner] {arm_name} arm not available, skipping"
                    )
                    continue
                print(
                    f"[motion-planner] Selected {arm_name} arm for {encoding} target"
                )

                q_start = arm.current_joints.clone()
                pc_tensor, pc_robot = build_pointcloud(
                    latest_depth,
                    latest_intrinsics,
                    latest_image_size,
                    cam_t,
                    cam_rot,
                    device,
                )

                # Compute table plane from camera frustum
                table_plane, table_polygon = _setup_table_plane(
                    arm.optimizer, pc_robot, cam_t, cam_rot,
                    latest_intrinsics, latest_image_size, device,
                )

                result = None

                if encoding == "jaw_pixels":
                    u1, v1, u2, v2 = target_data[:4]
                    result = plan_grasp_from_pixels(
                        u1,
                        v1,
                        u2,
                        v2,
                        node,
                        arm.optimizer,
                        arm.ik_chain,
                        arm.optimizer.capsules,
                        arm.joint_limits,
                        device,
                        arm.current_joints,
                        latest_depth,
                        latest_intrinsics,
                        latest_image_size,
                        cam_t,
                        cam_rot,
                        pc_tensor,
                        arm.num_joints,
                        arm=arm_name,
                        table_plane=table_plane,
                        table_polygon=table_polygon,
                    )

                elif encoding == "jointstate":
                    q_goal = torch.tensor(
                        target_data[: arm.num_joints],
                        dtype=torch.float32,
                        device=device,
                    )
                    result = plan_and_send(
                        node,
                        arm.optimizer,
                        q_start,
                        q_goal,
                        pc_tensor,
                        arm.num_joints,
                        ik_chain=arm.ik_chain,
                        capsule_model=arm.optimizer.capsules,
                        arm=arm_name,
                        table_plane=table_plane,
                        table_polygon=table_polygon,
                    )

                elif encoding in ("xyzrpy", "xyzquat"):
                    if encoding == "xyzrpy":
                        q_goal = solve_ik(
                            arm.ik_chain,
                            target_data[:6],
                            arm.current_joints,
                            arm.joint_limits,
                            device,
                        )
                    else:
                        pos = target_data[:3]
                        target_tf = pk.Transform3d(
                            pos=torch.tensor(pos, dtype=torch.float32),
                            rot=torch.tensor(
                                target_data[3:7], dtype=torch.float32
                            ),
                        ).to(device)
                        q_init = (
                            arm.current_joints.unsqueeze(0).requires_grad_(True)
                        )
                        ik_solver = pk.PseudoInverseIK(
                            arm.ik_chain,
                            max_iterations=100_000,
                            retry_configs=q_init,
                            joint_limits=torch.tensor(arm.joint_limits),
                            early_stopping_any_converged=True,
                            debug=False,
                            lr=0.05,
                            pos_tolerance=0.005,
                            rot_tolerance=0.05,
                        )
                        ik_result = ik_solver.solve(target_tf)
                        err_pos = float(ik_result.err_pos.min())
                        if err_pos > 0.01:
                            print(
                                f"[motion-planner] IK failed: pos_err={err_pos:.4f}"
                            )
                            continue
                        q_goal = ik_result.solutions.detach()[0, 0].to(device)

                    if q_goal is None:
                        continue
                    result = plan_and_send(
                        node,
                        arm.optimizer,
                        q_start,
                        q_goal,
                        pc_tensor,
                        arm.num_joints,
                        ik_chain=arm.ik_chain,
                        capsule_model=arm.optimizer.capsules,
                        arm=arm_name,
                        table_plane=table_plane,
                        table_polygon=table_polygon,
                    )

                else:
                    print(f"[motion-planner] Unknown encoding: {encoding}")

                if result is not None:
                    _set_playback(playback, result[0], result[1])

            # --- Grasp result ---
            elif event_id == "grasp_result":
                raw_text = event["value"][0].as_py()
                try:
                    data = json.loads(raw_text)
                except (json.JSONDecodeError, TypeError):
                    print(
                        f"[motion-planner] grasp_result: failed to parse JSON: {raw_text}"
                    )
                    continue

                p1 = data.get("p1")
                p2 = data.get("p2")
                if not p1 or not p2 or len(p1) < 2 or len(p2) < 2:
                    print(f"[motion-planner] grasp_result: missing p1/p2: {data}")
                    continue

                w, h = latest_image_size
                u1 = float(p1[0]) * w / 1000.0
                v1 = float(p1[1]) * h / 1000.0
                u2 = float(p2[0]) * w / 1000.0
                v2 = float(p2[1]) * h / 1000.0
                print(
                    f"[motion-planner] grasp_result: jaw1=({u1:.0f},{v1:.0f}) "
                    f"jaw2=({u2:.0f},{v2:.0f})"
                )

                # Arm selection from metadata or deprojected Y
                target_y = _estimate_target_y(
                    u1,
                    v1,
                    u2,
                    v2,
                    latest_depth,
                    latest_intrinsics,
                    latest_image_size,
                    cam_t,
                    cam_rot,
                )
                arm_name = select_arm(metadata, target_y)
                arm = arms.get(arm_name)
                if arm is None:
                    print(
                        f"[motion-planner] {arm_name} arm not available, skipping"
                    )
                    continue
                print(f"[motion-planner] Selected {arm_name} arm for grasp")

                pc_tensor, pc_robot = build_pointcloud(
                    latest_depth,
                    latest_intrinsics,
                    latest_image_size,
                    cam_t,
                    cam_rot,
                    device,
                )

                # Compute table plane from camera frustum
                table_plane, table_polygon = _setup_table_plane(
                    arm.optimizer, pc_robot, cam_t, cam_rot,
                    latest_intrinsics, latest_image_size, device,
                )

                result = plan_grasp_from_pixels(
                    u1,
                    v1,
                    u2,
                    v2,
                    node,
                    arm.optimizer,
                    arm.ik_chain,
                    arm.optimizer.capsules,
                    arm.joint_limits,
                    device,
                    arm.current_joints,
                    latest_depth,
                    latest_intrinsics,
                    latest_image_size,
                    cam_t,
                    cam_rot,
                    pc_tensor,
                    arm.num_joints,
                    arm=arm_name,
                    table_plane=table_plane,
                    table_polygon=table_polygon,
                )
                if result is not None:
                    _set_playback(playback, result[0], result[1])

        elif event["type"] == "STOP":
            break


if __name__ == "__main__":
    main()
