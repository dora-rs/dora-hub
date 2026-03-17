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
    CAMERA_TRANSFORM:     6 values "x y z roll pitch yaw" in degrees — scene-frame
                          convention Ry(yaw) @ Rx(roll) @ Ry(pitch), matching the
                          openarm.html viewer's IMU-calibrated output.
                          Or 7 values "tx ty tz qw qx qy qz" (quaternion in URDF frame).
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
    PLAYBACK:             "true"/"false"/"confirm" — enable built-in playback on tick.
                          "confirm" plans but waits for an ``execute`` input to start.
"""

import json
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pinocchio as pin
import pyarrow as pa
import pytorch_kinematics as pk
import torch
from dora import Node
from scipy.spatial.transform import Rotation as ScipyRotation, Slerp

from .collision_model import (
    CapsuleCollisionModel,
    OPENARM_CAPSULES,
    OPENARM_RIGHT_CAPSULES,
    OPENARM_GRIPPER_BOXES,
    OPENARM_RIGHT_GRIPPER_BOXES,
    capsule_points_distance,
    capsule_halfplane_distance,
    capsule_capsule_distance,
    box_points_distance,
)
from .grasp_utils import grasp_pose_from_jaw_pixels, place_pose_from_pixel
from .pointcloud import (
    compute_table_plane,
    depth_to_pointcloud,
    filter_below_table,
    parse_camera_transform,
    transform_points,
    pointcloud_to_tensor,
)
from .trajectory_json import (
    save as save_trajectory_json,
    build as build_trajectory_json,
    load as load_trajectory_json,
    GRIPPER_OPEN_RAD,
    GRIPPER_CLOSED_RAD,
)
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
CAMERA_TRANSFORM_STR = os.getenv("CAMERA_TRANSFORM", "0.79 0.81 1.23 90 0 -45")
DOWNSAMPLE_STRIDE = int(os.getenv("DOWNSAMPLE_STRIDE", "8"))
SAFETY_MARGIN = float(os.getenv("SAFETY_MARGIN", "0.02"))
NUM_WAYPOINTS = int(os.getenv("NUM_WAYPOINTS", "200"))
NUM_SEEDS = int(os.getenv("NUM_SEEDS", "4"))
MAX_ITERS = int(os.getenv("MAX_ITERS", "200"))
DEVICE = os.getenv("DEVICE", "cuda")
GRASP_DEPTH_OFFSET = float(os.getenv("GRASP_DEPTH_OFFSET", "-0.01"))
PLACE_DEPTH_OFFSET = float(os.getenv("PLACE_DEPTH_OFFSET", "0.05"))
FLOOR_HEIGHT = float(os.getenv("FLOOR_HEIGHT", "0.005"))
APPROACH_MARGIN = float(os.getenv("APPROACH_MARGIN", "0.15"))
JAW_CONTACT_DEPTH = float(os.getenv("JAW_CONTACT_DEPTH", "0.0"))
APPROACH_ANGLE_DEG = float(os.getenv("APPROACH_ANGLE_DEG", "70.0"))
IMAGE_WIDTH = int(os.getenv("IMAGE_WIDTH", "640"))
IMAGE_HEIGHT = int(os.getenv("IMAGE_HEIGHT", "480"))
MAX_JOINT_STEP = float(os.getenv("MAX_JOINT_STEP", "0.1"))
EXPORT_PATH = os.getenv("EXPORT_PATH", "")
EXPORT_DIR = os.getenv("EXPORT_DIR", "")
PLAYBACK_MODE = os.getenv("PLAYBACK", "true").lower()  # "true", "false", "confirm"
PLAYBACK = PLAYBACK_MODE in ("1", "true", "yes", "confirm")

# Pick-and-place: optional place target (xyz or xyzrpy in robot frame).
# When set, grasp planning extends to: pick → close → transport → place → open → home.
PLACE_TARGET_STR = os.getenv("PLACE_TARGET", "")
DWELL_STEPS = int(os.getenv("DWELL_STEPS", "15"))  # hold waypoints for gripper action
CARTESIAN_SPEED = float(os.getenv("CARTESIAN_SPEED", "0.15"))  # m/s EE speed
PLAYBACK_HZ = 30  # trajectory playback rate

# Gripper motor angle constants (imported from trajectory_json)
GRIPPER_TRAVEL_MM = 44.0      # jaw travel range in mm


def compute_gripper_close_rad(object_width_m):
    """Compute gripper close angle based on object width.

    Starts from fully closed and opens just enough to leave a gap
    slightly smaller than the object (5mm squeeze) for a firm hold.
    """
    gap_mm = max(0, object_width_m * 1000 - 30)  # 30mm squeeze margin
    # Fraction of travel to open from closed: 0 = fully closed, 1 = fully open
    open_frac = min(1.0, gap_mm / GRIPPER_TRAVEL_MM)
    return GRIPPER_CLOSED_RAD + open_frac * (GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD)

# Parse place target at module level
_place_target = None
if PLACE_TARGET_STR.strip():
    _vals = [float(v) for v in PLACE_TARGET_STR.split()]
    if len(_vals) == 3:
        _place_target = np.array(_vals, dtype=np.float32)  # xyz only
    elif len(_vals) == 6:
        _place_target = np.array(_vals, dtype=np.float32)  # xyzrpy
    else:
        print(f"[motion-planner] WARNING: PLACE_TARGET must be 3 (xyz) or 6 (xyzrpy) values, got {len(_vals)}")


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
class PinIK:
    """Pinocchio-based fast IK solver for a single arm."""

    model: pin.Model
    data: pin.Data
    frame_id: int
    joint_indices: list  # indices into full model q for this arm's 7 joints

    def solve(self, target_pos, q_init=None, max_iter=200, eps=1e-3, damp=1e-6,
              target_rot=None, rot_weight=0.5, nullspace_weight=1.0):
        """IK solver. Position-only by default; add target_rot (3x3) for 6-DOF.

        Uses nullspace projection to keep joints close to q_init, preventing
        wrist flips in redundant configurations.

        Returns (q_arm, pos_err) — q_arm is 7-DOF.
        """
        q = pin.neutral(self.model)
        q_ref = q.copy()
        if q_init is not None:
            for i, ji in enumerate(self.joint_indices):
                q[ji] = float(q_init[i])
                q_ref[ji] = float(q_init[i])
        use_rot = target_rot is not None
        nv = self.model.nv
        for _ in range(max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            oMf = self.data.oMf[self.frame_id]
            pos_err = target_pos - oMf.translation
            if use_rot:
                rot_err = pin.log3(target_rot @ oMf.rotation.T)
                err = np.concatenate([pos_err, rot_weight * rot_err])
                if np.linalg.norm(pos_err) < eps and np.linalg.norm(rot_err) < eps * 5:
                    break
                J = pin.computeFrameJacobian(
                    self.model, self.data, q, self.frame_id, pin.LOCAL_WORLD_ALIGNED
                )
                J[3:, :] *= rot_weight
            else:
                err = pos_err
                if np.linalg.norm(err) < eps:
                    break
                J = pin.computeFrameJacobian(
                    self.model, self.data, q, self.frame_id, pin.LOCAL_WORLD_ALIGNED
                )[:3, :]
            JJt = J @ J.T + damp * np.eye(J.shape[0])
            J_pinv = J.T @ np.linalg.solve(JJt, np.eye(J.shape[0]))
            dq = J_pinv @ err
            # Nullspace projection: use redundancy to stay close to q_init
            if nullspace_weight > 0 and q_init is not None:
                N = np.eye(nv) - J_pinv @ J
                dq += N @ (nullspace_weight * (q_ref - q))
            q = pin.integrate(self.model, q, dq)
            q = np.clip(q, self.model.lowerPositionLimit, self.model.upperPositionLimit)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        final_err = np.linalg.norm(target_pos - self.data.oMf[self.frame_id].translation)
        q_arm = np.array([q[ji] for ji in self.joint_indices], dtype=np.float32)
        return q_arm, final_err


def _build_pin_ik(urdf_path, ee_link_name, arm_side):
    """Build a PinIK solver for one arm."""
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    frame_id = model.getFrameId(ee_link_name)
    # Map arm joints: L_J1..L_J7 or R_J1..R_J7 → model q indices
    prefix = "L_" if arm_side == "left" else "R_"
    joint_indices = []
    for i in range(1, model.njoints):
        name = model.names[i]
        if name.startswith(prefix) and name.endswith(tuple(f"J{j}" for j in range(1, 8))):
            joint_indices.append(i - 1)  # q index = joint index - 1
    return PinIK(model, data, frame_id, joint_indices)


@dataclass
class ArmConfig:
    """Per-arm state: kinematic chain, optimizer, and current joint positions."""

    name: str
    ik_chain: pk.Chain
    optimizer: TrajectoryOptimizer
    joint_limits: tuple
    num_joints: int
    current_joints: torch.Tensor
    pin_ik: PinIK = None


def _init_arm(name, urdf_path, ee_link, capsules, device, boxes=None):
    """Build chain, capsule model, optimizer, and initial state for one arm."""
    chain = build_chain(urdf_path, ee_link)
    joint_limits = chain.get_joint_limits()
    num_joints = len(chain.get_joint_parameter_names(exclude_fixed=True))

    ik_chain = chain.to(dtype=torch.float32, device=str(device))

    capsule_model = CapsuleCollisionModel(capsules, device, boxes=boxes)
    optimizer = TrajectoryOptimizer(
        chain=ik_chain,
        capsule_model=capsule_model,
        joint_limits=joint_limits,
        device=device,
        collision_alpha=50.0,
        max_joint_step=MAX_JOINT_STEP,
    )
    current_joints = torch.zeros(num_joints, dtype=torch.float32, device=device)

    with torch.no_grad():
        home_fk = ik_chain.forward_kinematics(
            torch.zeros(1, num_joints, device=device)
        )
        home_pos = home_fk.get_matrix()[0, :3, 3].cpu().numpy()
        print(f"[motion-planner] {name} arm: {num_joints}J, EE={ee_link}")

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
                f"[motion-planner] WARNING: {name} collision links NOT in FK: {missing}"
            )

    # Build pinocchio-based fast IK solver
    pin_solver = _build_pin_ik(urdf_path, ee_link, name)

    return ArmConfig(name, ik_chain, optimizer, joint_limits, num_joints, current_joints,
                     pin_ik=pin_solver)


def select_arm(metadata, target_y=None, target_x=None):
    """Select arm closest to target. Robot is 45deg angled: dividing line is Y = -X."""
    arm = metadata.get("arm", "")
    if arm in ("left", "right"):
        return arm
    if target_x is not None and target_y is not None:
        return "right" if (target_x + target_y) < 0 else "left"
    if target_y is not None:
        return "left" if target_y < 0 else "right"
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
    pt_robot = cam_rot.as_matrix() @ np.array([x_cam, y_cam, z]) + cam_t
    return float(pt_robot[0]), float(pt_robot[1])


def _fk_pos_chain(ik_chain, q):
    """Get end-effector XYZ for a single joint config (standalone helper)."""
    with torch.no_grad():
        q_dev = q.to(ik_chain.device) if q.device != ik_chain.device else q
        fk = ik_chain.forward_kinematics(q_dev.unsqueeze(0))
        return fk.get_matrix()[0, :3, 3].cpu().numpy()


def _ik_batch_adam(
    ik_chain,
    target_pos,
    target_rot,
    current_joints,
    lower,
    upper,
    device,
    num_seeds=4,
    max_iters=2000,
    rot_weight=0.0,
):
    """Batched Adam IK: all seeds optimised in parallel.

    Returns (best_q, best_pos_err) or (None, err) if no seed converges.
    """
    nj = len(lower)
    if torch.is_tensor(current_joints):
        current_joints = current_joints.to(device)
    else:
        current_joints = torch.tensor(current_joints, dtype=torch.float32, device=device)
    if target_rot is not None and rot_weight > 0:
        # Mixed seeding: half from current_joints (good position), half from home (good rotation)
        q_batch = lower + (upper - lower) * torch.rand(num_seeds, nj, device=device)
        q_batch[0] = current_joints.clone()
        # Seeds 1..half: perturbations of current_joints
        half = num_seeds // 2
        for si in range(1, half):
            q_batch[si] = current_joints + 0.3 * torch.randn(nj, device=device)
        # Seeds half..end: perturbations of home (zero joints = good wrist orientation)
        q_home = torch.zeros(nj, device=device)
        for si in range(half, num_seeds):
            q_batch[si] = q_home + 0.3 * torch.randn(nj, device=device)
        q_batch = torch.clamp(q_batch, lower, upper)
    else:
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

        if rot_weight > 0 and target_rot is not None:
            fk_rot = fk.get_matrix()[:, :3, :3]
            eye = torch.eye(3, device=device).unsqueeze(0)
            rot_diff = eye - torch.bmm(
                fk_rot.transpose(1, 2),
                target_rot.unsqueeze(0).expand(num_seeds, -1, -1),
            )
            rot_loss = (rot_diff**2).sum()
            # Two-phase: first half emphasizes rotation, second half balances both
            if it < max_iters // 3:
                loss = pos_loss.sum() + rot_weight * 5.0 * rot_loss
            else:
                loss = pos_loss.sum() + rot_weight * rot_loss
        else:
            loss = pos_loss.sum()

        loss.backward()
        opt.step()
        with torch.no_grad():
            q_batch.clamp_(lower, upper)

    with torch.no_grad():
        q_c = torch.clamp(q_batch, lower, upper)
        fk = ik_chain.forward_kinematics(q_c)
        fk_mat = fk.get_matrix()
        fk_pos = fk_mat[:, :3, 3]
        pos_errs = (fk_pos - target_pos_b).norm(dim=1)
        # Compute rotation error if target_rot is given
        if target_rot is not None and rot_weight > 0:
            fk_rot = fk_mat[:, :3, :3]
            eye = torch.eye(3, device=device).unsqueeze(0)
            rot_diff = eye - torch.bmm(
                fk_rot.transpose(1, 2),
                target_rot.unsqueeze(0).expand(num_seeds, -1, -1),
            )
            rot_errs = (rot_diff**2).sum(dim=(1, 2)).sqrt()
            # Combined score: position + weighted rotation
            combined = pos_errs + 0.1 * rot_errs
            best_combined = combined.min()
            good_mask = combined < best_combined + 0.01
        else:
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
    num_seeds=4,
    max_iters=500,
    pos_threshold=0.005,
    position_only=False,
    pin_ik=None,
):
    """Solve IK — pinocchio first (fast), Adam fallback (GPU, orientation).

    Args:
        pos_threshold: Max acceptable position error in meters (default 0.02).
        position_only: If True, skip orientation and solve position-only directly.
        pin_ik: Optional PinIK solver for fast position-only IK.

    Returns q_goal tensor or None.
    """
    t_ik_start = time.perf_counter()
    target_xyz = target_xyzrpy[:3]

    # Try pinocchio first (position-only, ~0.1ms) — skip if orientation matters
    if pin_ik is not None and position_only:
        q_init = current_joints.cpu().numpy() if torch.is_tensor(current_joints) else current_joints
        q_np, err = pin_ik.solve(np.array(target_xyz, dtype=np.float64), q_init=q_init)
        if err <= pos_threshold:
            q_t = torch.tensor(q_np, dtype=torch.float32, device=device)
            ee_check = _fk_pos_chain(ik_chain, q_t)
            cross_err = float(np.linalg.norm(ee_check - target_xyz))
            if cross_err <= pos_threshold:
                return q_t
            else:
                print(f"[IK] PinIK cross-check failed: pin_err={err*1000:.1f}mm, "
                      f"pk_err={cross_err*1000:.1f}mm — falling back to Adam")

    # Fallback: batched Adam (GPU)
    target_pos = torch.tensor(target_xyz, dtype=torch.float32, device=device)
    target_rot = pk.transforms.euler_angles_to_matrix(
        torch.tensor(target_xyzrpy[3:6], dtype=torch.float32, device=device),
        convention="XYZ",
    )
    lower = torch.tensor(joint_limits[0], dtype=torch.float32, device=device)
    upper = torch.tensor(joint_limits[1], dtype=torch.float32, device=device)

    if not position_only:
        q, err = _ik_batch_adam(
            ik_chain, target_pos, target_rot, current_joints,
            lower, upper, device, num_seeds, max_iters, rot_weight=20.0,
        )
        if err <= pos_threshold:
            # Verify full rotation matches (not just position)
            with torch.no_grad():
                fk_mat = ik_chain.forward_kinematics(q.unsqueeze(0)).get_matrix()[0]
                rot_err_mat = torch.eye(3, device=device) - fk_mat[:3,:3].T @ target_rot
                rot_frob = float((rot_err_mat ** 2).sum().sqrt())
            if rot_frob > 1.0:
                # Rotation mismatch (wrist flip) - retry with very high rot_weight
                q2, err2 = _ik_batch_adam(
                    ik_chain, target_pos, target_rot, current_joints,
                    lower, upper, device, num_seeds=32, max_iters=2000, rot_weight=100.0,
                )
                if err2 <= pos_threshold * 3:
                    fk2 = ik_chain.forward_kinematics(q2.unsqueeze(0)).get_matrix()[0]
                    rot_frob2 = float((torch.eye(3, device=device) - fk2[:3,:3].T @ target_rot).pow(2).sum().sqrt())
                    if rot_frob2 < rot_frob:
                        return q2
            return q
        # Retry with more seeds and relaxed threshold
        q2, err2 = _ik_batch_adam(
            ik_chain, target_pos, target_rot, current_joints,
            lower, upper, device, num_seeds=16, max_iters=1000, rot_weight=20.0,
        )
        if err2 <= pos_threshold * 2:
            print(f"[IK] Orientation IK: relaxed pass (err={err2*1000:.1f}mm, 16 seeds)")
            return q2

    q, err = _ik_batch_adam(
        ik_chain, target_pos, None, current_joints,
        lower, upper, device, num_seeds, max_iters, rot_weight=0.0,
    )
    t_ik = time.perf_counter() - t_ik_start
    if err <= pos_threshold:
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

    if has_pc:
        device = point_cloud.device
    elif table_polygon is not None:
        device = table_polygon.device
    else:
        device = trajectory.device
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

        # Body capsule collisions (support pole)
        # Skip the first N links (mounted on/adjacent to the body).
        if capsule_model.body_capsule is not None:
            B = traj.shape[0]
            body_p0 = capsule_model.body_p0.unsqueeze(0).expand(B, -1)
            body_p1 = capsule_model.body_p1.unsqueeze(0).expand(B, -1)
            body_r = capsule_model.body_radius
            skip = capsule_model.body_skip_links
            for link_name in capsule_links[skip:]:
                p0_w, p1_w = capsule_model.capsule_endpoints_world(link_name, transforms)
                r = capsule_model.radii[link_name]
                sd = capsule_capsule_distance(p0_w, p1_w, r, body_p0, body_p1, body_r)
                for t_idx in range(B):
                    d = float(sd[t_idx])
                    if d < margin:
                        collisions.append((t_idx, f"{link_name}[body]", d))

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
    if collisions:
        colliding_waypoints = len(set(c[0] for c in collisions))
        worst = min(collisions, key=lambda c: c[2])
        print(
            f"[motion-planner] Collision: {colliding_waypoints}/{total_waypoints} wp, "
            f"worst penetration={-worst[2]*1000:.1f}mm"
        )


def plan_and_send(node, optimizer, q_start, q_goal, pc_tensor, num_joints,
                   ik_chain=None, capsule_model=None, arm="left",
                   table_plane=None, table_polygon=None):
    """Run trajectory optimisation, send the result, and return it.

    Returns ``(traj_np, metadata)`` — the (T, J) float32 trajectory and its
    metadata dict, so the caller can store it for playback / export.
    """
    pc_info = f"{len(pc_tensor)} pts" if pc_tensor is not None else "no pc"
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
    print(f"[motion-planner] Planned: cost={best_cost:.4f} ({t_plan:.1f}s)")

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
    place_uv=None,
    pin_ik=None,
):
    """Plan a grasp (or pick-and-place) trajectory from two jaw pixel positions.

    When ``place_uv`` is None, plans a 2-phase approach+grasp trajectory.
    When ``place_uv=(u, v)`` is given, plans a full pick-and-place:
    start → pre-grasp → grasp → dwell(close) → pre-place → place →
    dwell(open) → pre-place → home.

    Returns ``(traj_np, metadata)`` on success, or ``None`` on failure.
    """
    if latest_depth is None or latest_intrinsics is None:
        print("[motion-planner] grasp: no depth/intrinsics yet, skipping")
        return None, "No depth/intrinsics data"

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
        jaw_contact_depth=JAW_CONTACT_DEPTH,
        approach_angle_deg=APPROACH_ANGLE_DEG,
    )
    if result is None:
        print("[motion-planner] grasp: invalid depth at jaw points, skipping")
        return None, "Invalid depth at grasp point"

    grasp_xyzrpy, pregrasp_xyzrpy, object_top_z, object_width = result

    # Per-arm calibration offset (metres)
    # URDF frame: X=forward, Y=left/right, Z=up
    cal_offset = np.array([0.03, -0.05, 0.0, 0.0, 0.0, 0.0])
    grasp_xyzrpy = grasp_xyzrpy + cal_offset
    pregrasp_xyzrpy = pregrasp_xyzrpy + cal_offset

    q_pregrasp = solve_ik(
        ik_chain, pregrasp_xyzrpy, current_joints, joint_limits, device,
        num_seeds=NUM_SEEDS,
    )
    if q_pregrasp is None:
        print("[motion-planner] grasp: IK failed for pre-grasp, skipping")
        return None, "IK failed for pre-grasp pose (unreachable)"

    q_grasp = solve_ik(ik_chain, grasp_xyzrpy, q_pregrasp, joint_limits, device,
                       num_seeds=NUM_SEEDS)
    if q_grasp is None:
        print("[motion-planner] grasp: IK failed for grasp, skipping")
        return None, "IK failed for grasp pose (unreachable)"

    # --- Pick-and-place: solve IK for place/preplace poses ---
    q_place = None
    q_preplace = None
    place_xyzrpy = None

    if place_uv is not None:
        place_u, place_v = place_uv
        place_result = place_pose_from_pixel(
            place_u, place_v,
            latest_depth, fx, fy, cx, cy,
            cam_t, cam_rot,
            grasp_rpy=grasp_xyzrpy[3:6],
            width=w, height=h,
            place_depth_offset=PLACE_DEPTH_OFFSET,
            floor_height=FLOOR_HEIGHT,
            approach_margin=APPROACH_MARGIN,
            jaw_contact_depth=JAW_CONTACT_DEPTH,
        )
        if place_result is not None:
            place_xyzrpy, preplace_xyzrpy = place_result
            place_xyzrpy = place_xyzrpy + cal_offset
            preplace_xyzrpy = preplace_xyzrpy + cal_offset
            q_preplace = solve_ik(
                ik_chain, preplace_xyzrpy, q_grasp, joint_limits, device,
                num_seeds=NUM_SEEDS,
            )
            if q_preplace is not None:
                q_place = solve_ik(
                    ik_chain, place_xyzrpy, q_preplace, joint_limits, device,
                    num_seeds=NUM_SEEDS,
                )
            if q_place is None:
                print("[motion-planner] grasp: Place IK failed, rejecting")
                return None, "Place target unreachable (IK failed)"
        else:
            print("[motion-planner] grasp: Place depth invalid, rejecting")
            return None, "Place target has invalid depth"


    q_start = current_joints.clone()
    t_plan_start = time.perf_counter()

    result = _build_pick_place_trajectory(
        node, optimizer, ik_chain, capsule_model,
        q_start, q_pregrasp, q_grasp, q_preplace, q_place,
        pc_tensor, num_joints, arm, object_width,
        grasp_xyzrpy, place_xyzrpy,
        joint_limits=joint_limits, device=device,
        table_plane=table_plane, table_polygon=table_polygon,
        pin_ik=pin_ik,
    )
    if result is None:
        return None, "Trajectory planning failed"
    return result, None


def _build_pick_place_trajectory(
    node, optimizer, ik_chain, capsule_model,
    q_start, q_pregrasp, q_grasp, q_preplace, q_place,
    pc_tensor, num_joints, arm, object_width,
    grasp_xyzrpy, place_xyzrpy,
    joint_limits=None, device=None,
    table_plane=None, table_polygon=None,
    pin_ik=None,
):
    """Build a full pick-and-place trajectory with dwell waypoints and gripper actions.

    Phases:
      1. start → pre-grasp          (long, collision-aware)
      2. pre-grasp → grasp           (short, approach)
      dwell: hold at grasp           (gripper closes)
      3. grasp → pre-place           (long, collision-aware)
      4. pre-place → place           (short, descent)
      dwell: hold at place           (gripper opens)
      5. place → pre-place           (short, retreat)
      6. pre-place → home            (long, collision-aware)

    Returns ``(traj_np, metadata)`` on success, or ``None`` on failure.
    """
    t0 = time.perf_counter()
    q_home = torch.zeros_like(q_start)

    segments = []
    phase_labels = []  # (start_wp, label) for each segment
    phase_count = [0]

    def _fk_pos(q):
        return _fk_pos_chain(ik_chain, q)

    # Staging XY to route around the center pole between arms
    if arm == "left":
        STAGING_XY = (-0.2, -0.3)
    else:
        STAGING_XY = (-0.3, 0.1)

    def _cartesian_segment(label, q_s, q_g, safe_z=False, staging=False, target_rot=None, speed=None):
        """Cartesian-space linear interpolation with IK at keypoints.

        Interpolates EE position linearly in Cartesian space, solves IK at
        each keypoint (seeded from previous solution for continuity), then
        fills in waypoints with short joint-space lerps between adjacent
        IK solutions.

        T (waypoint count) is computed from path length and target speed
        to ensure constant EE velocity.

        When ``safe_z=True``, routes through clearance height:
        start → (start_xy, z_clear) → (goal_xy, z_clear) → goal

        When ``staging=True`` (implies safe_z), also routes through the
        staging XY to avoid the center pole:
        start → (start_xy, z_clear) → (staging_xy, z_clear) → (goal_xy, z_clear) → goal

        When ``target_rot`` (3x3 matrix) is given, IK includes orientation.
        """
        if staging:
            safe_z = True
        phase_count[0] += 1
        p_s = _fk_pos(q_s)
        p_g = _fk_pos(q_g)
        t_start = time.perf_counter()
        ee_speed = speed if speed is not None else CARTESIAN_SPEED

        # Build Cartesian via-points and compute path length
        if safe_z:
            z_clear = max(p_s[2], p_g[2]) + 0.08
            if staging:
                via_points = np.array([
                    p_s,
                    [p_s[0], p_s[1], z_clear],
                    [STAGING_XY[0], STAGING_XY[1], z_clear],
                    [p_g[0], p_g[1], z_clear],
                    p_g,
                ])
            else:
                via_points = np.array([
                    p_s,
                    [p_s[0], p_s[1], z_clear],
                    [p_g[0], p_g[1], z_clear],
                    p_g,
                ])
            seg_dists = np.linalg.norm(np.diff(via_points, axis=0), axis=1)
            path_length = float(np.sum(seg_dists))
        else:
            via_points = None
            path_length = float(np.linalg.norm(p_g - p_s))

        # Compute T and keypoints from path length (constant EE velocity)
        T = max(int(path_length / ee_speed * PLAYBACK_HZ), 10)
        n_keypoints = max(int(path_length / 0.001), 5)  # 1 keypoint per 1mm
        n_keypoints = min(n_keypoints, T // 3)  # at least 3 waypoints per keypoint for smoothing
        mode = "cartesian-safe" if safe_z else "cartesian"

        if joint_limits is None or device is None:
            print(f"  WARNING: no joint_limits/device, falling back to joint lerp")
            alphas = torch.linspace(0, 1, T).unsqueeze(1)
            traj = q_s.cpu() * (1 - alphas) + q_g.cpu() * alphas
        else:
            lower = torch.tensor(joint_limits[0], dtype=torch.float32, device=device)
            upper = torch.tensor(joint_limits[1], dtype=torch.float32, device=device)

            # Generate keypoints along the Cartesian path
            if via_points is not None:
                cum = np.concatenate([[0], np.cumsum(seg_dists)])
                total_len = cum[-1]
                cart_targets = []
                for i in range(n_keypoints):
                    s = i / (n_keypoints - 1) * total_len
                    seg_idx = np.searchsorted(cum[1:], s, side='right')
                    seg_idx = min(seg_idx, len(seg_dists) - 1)
                    if seg_dists[seg_idx] > 0:
                        t_local = (s - cum[seg_idx]) / seg_dists[seg_idx]
                    else:
                        t_local = 0.0
                    pt = via_points[seg_idx] * (1 - t_local) + via_points[seg_idx + 1] * t_local
                    cart_targets.append(pt)
            else:
                # Straight-line Cartesian interpolation
                cart_targets = []
                for i in range(n_keypoints):
                    alpha = i / (n_keypoints - 1)
                    cart_targets.append(p_s * (1 - alpha) + p_g * alpha)

            # Solve IK at each Cartesian keypoint (pinocchio: ~0.1ms each)
            # First and last keypoints are pinned to q_s/q_g exactly to
            # avoid joint discontinuities at phase boundaries.
            # If target_rot is set, slerp rotation from start to target.
            start_rot = None
            if target_rot is not None and pin_ik is not None:
                q_full = pin.neutral(pin_ik.model)
                q_s_np = q_s.cpu().numpy()
                for ji_idx, ji in enumerate(pin_ik.joint_indices):
                    q_full[ji] = float(q_s_np[ji_idx])
                pin.forwardKinematics(pin_ik.model, pin_ik.data, q_full)
                pin.updateFramePlacements(pin_ik.model, pin_ik.data)
                start_rot = pin_ik.data.oMf[pin_ik.frame_id].rotation.copy()
                rots = ScipyRotation.from_matrix(np.stack([start_rot, target_rot]))
                rot_slerp = Slerp([0.0, 1.0], rots)

            key_qs = [q_s.clone()]  # start is already known
            q_prev_np = q_s.cpu().numpy()
            failed = 0
            for i in range(1, n_keypoints):
                # Last keypoint → use q_g exactly (no IK solve)
                if i == n_keypoints - 1:
                    key_qs.append(q_g.clone())
                    break
                target_xyz = cart_targets[i]
                # Interpolated rotation at this keypoint
                kp_rot = None
                if target_rot is not None and start_rot is not None:
                    alpha_r = i / (n_keypoints - 1)
                    kp_rot = rot_slerp(alpha_r).as_matrix()
                if pin_ik is not None:
                    q_np, err = pin_ik.solve(
                        np.array(target_xyz, dtype=np.float64), q_init=q_prev_np,
                        target_rot=kp_rot,
                    )
                    max_jump = float(np.max(np.abs(q_np - q_prev_np)))
                    # Cross-validate with pytorch-kinematics FK
                    cross_ok = True
                    if err <= 0.05 and max_jump <= np.pi / 2:
                        q_t = torch.tensor(q_np, dtype=torch.float32, device=device)
                        ee_check = _fk_pos(q_t)
                        cross_err = float(np.linalg.norm(ee_check - target_xyz))
                        if cross_err > 0.01:
                            cross_ok = False
                    else:
                        cross_ok = False
                    if not cross_ok:
                        # Fallback: Adam IK for this keypoint (safe, no table collision)
                        target_pos = torch.tensor(target_xyz, dtype=torch.float32, device=device)
                        q_prev_dev = torch.tensor(q_prev_np, dtype=torch.float32, device=device)
                        q_adam, adam_err = _ik_batch_adam(
                            ik_chain, target_pos, None, q_prev_dev,
                            lower, upper, device,
                            num_seeds=2, max_iters=150, rot_weight=0.0,
                        )
                        if adam_err <= 0.01:
                            key_qs.append(q_adam)
                            q_prev_np = q_adam.detach().cpu().numpy()
                        else:
                            failed += 1
                            alpha_j = i / (n_keypoints - 1)
                            q_fallback = q_s.cpu().numpy() * (1 - alpha_j) + q_g.cpu().numpy() * alpha_j
                            key_qs.append(torch.tensor(q_fallback, dtype=torch.float32, device=device))
                    else:
                        key_qs.append(q_t)
                        q_prev_np = q_np
                else:
                    # Fallback to Adam if no pinocchio
                    target_pos = torch.tensor(target_xyz, dtype=torch.float32, device=device)
                    q_prev_dev = torch.tensor(q_prev_np, dtype=torch.float32, device=device)
                    q_ik, err = _ik_batch_adam(
                        ik_chain, target_pos, None, q_prev_dev,
                        lower, upper, device,
                        num_seeds=2, max_iters=150, rot_weight=0.0,
                    )
                    if err > 0.05:
                        failed += 1
                        alpha_j = torch.tensor(i / (n_keypoints - 1), dtype=torch.float32)
                        q_fallback = q_s.to(device) * (1 - alpha_j) + q_g.to(device) * alpha_j
                        key_qs.append(q_fallback)
                    else:
                        key_qs.append(q_ik)
                        q_prev_np = q_ik.detach().cpu().numpy()

            if failed > 0:
                print(f"[motion-planner] {label}: {failed}/{n_keypoints-1} keypoint IK failures")

            # Distribute T waypoints across keypoint intervals
            n_intervals = len(key_qs) - 1
            wp_per_interval = max(T // n_intervals, 2)
            segments_inner = []
            total_wp = 0
            for i in range(n_intervals):
                qs = key_qs[i].cpu()
                qg = key_qs[i + 1].cpu()
                if i < n_intervals - 1:
                    n_wp = wp_per_interval
                else:
                    n_wp = max(T - total_wp, 2)
                alphas = torch.linspace(0, 1, n_wp).unsqueeze(1)
                seg = qs * (1 - alphas) + qg * alphas
                if i > 0:
                    seg = seg[1:]
                segments_inner.append(seg)
                total_wp += seg.shape[0]
            traj = torch.cat(segments_inner, dim=0)

            phase_labels.append((sum(s.shape[0] for s in segments), label, traj.shape[0]))
        return traj

    # Helper: append segment, threading actual last waypoint for continuity
    def _append(seg, skip_first=True):
        if skip_first and len(segments) > 0:
            segments.append(seg[1:])
        else:
            segments.append(seg)
        return seg[-1]  # return actual last waypoint

    # Phase 1: start → pre-grasp (via staging to avoid pole)
    seg1 = _cartesian_segment("Phase 1: start -> pre-grasp", q_start, q_pregrasp, staging=True)
    q_prev = _append(seg1, skip_first=False)

    # Phase 1b: rotate at pre-grasp to match grasp orientation
    if pin_ik is not None:
        p_pregrasp = _fk_pos(q_prev)
        # Get grasp EE rotation via pinocchio FK
        q_full_grasp = pin.neutral(pin_ik.model)
        q_grasp_np = q_grasp.cpu().numpy()
        for i, ji in enumerate(pin_ik.joint_indices):
            q_full_grasp[ji] = float(q_grasp_np[i])
        pin.forwardKinematics(pin_ik.model, pin_ik.data, q_full_grasp)
        pin.updateFramePlacements(pin_ik.model, pin_ik.data)
        grasp_rot = pin_ik.data.oMf[pin_ik.frame_id].rotation.copy()

        q_pre_rot_np, rot_err = pin_ik.solve(
            p_pregrasp, q_init=q_prev.cpu().numpy(),
            target_rot=grasp_rot, rot_weight=0.5,
        )
        if rot_err < 0.05:
            q_pre_rotated = torch.tensor(q_pre_rot_np, dtype=torch.float32, device=device)
            seg1b = _cartesian_segment("Phase 1b: rotate at pre-grasp", q_prev, q_pre_rotated, target_rot=grasp_rot)
            q_prev = _append(seg1b)

    # Phase 2: pre-grasp → grasp (short descent)
    seg2 = _cartesian_segment("Phase 2: pre-grasp -> grasp", q_prev, q_grasp)
    q_prev = _append(seg2)

    # Dwell at grasp (hold for gripper close)
    grasp_dwell_start = sum(s.shape[0] for s in segments)
    grasp_dwell = q_prev.unsqueeze(0).expand(DWELL_STEPS, -1).cpu()
    phase_labels.append((sum(s.shape[0] for s in segments), "Dwell: grasp (gripper close)", DWELL_STEPS))
    segments.append(grasp_dwell)

    if q_place is not None:
        # Phase 3: grasp -> pre-place (direct transit via staging)
        seg3 = _cartesian_segment("Phase 3: grasp -> pre-place", q_prev, q_preplace, safe_z=True)
        q_prev = _append(seg3)
    else:
        # Grasp-only: lift back to pre-grasp
        seg3 = _cartesian_segment("Phase 3: grasp -> pre-grasp", q_prev, q_pregrasp, safe_z=True)
        q_prev = _append(seg3)

    if q_place is not None:

        # Phase 3b: rotate at pre-place to match grasp orientation
        if pin_ik is None:
            print("[motion-planner] Skipping pre-place rotation (no pin_ik)")
        if pin_ik is not None:
            p_preplace = _fk_pos(q_prev)
            q_full_grasp = pin.neutral(pin_ik.model)
            q_grasp_np = q_grasp.cpu().numpy()
            for i, ji in enumerate(pin_ik.joint_indices):
                q_full_grasp[ji] = float(q_grasp_np[i])
            pin.forwardKinematics(pin_ik.model, pin_ik.data, q_full_grasp)
            pin.updateFramePlacements(pin_ik.model, pin_ik.data)
            grasp_rot_place = pin_ik.data.oMf[pin_ik.frame_id].rotation.copy()

            q_preplace_rot_np, rot_err = pin_ik.solve(
                p_preplace, q_init=q_prev.cpu().numpy(),
                target_rot=grasp_rot_place, rot_weight=0.5,
            )
            print(f"[motion-planner] Pre-place rotation IK: err={rot_err:.4f} (threshold=0.05)")
            if rot_err < 0.05:
                q_preplace_rotated = torch.tensor(q_preplace_rot_np, dtype=torch.float32, device=device)
                seg3b = _cartesian_segment("Phase 3b: rotate at pre-place", q_prev, q_preplace_rotated, target_rot=grasp_rot_place)
                q_prev = _append(seg3b)
            else:
                print(f"[motion-planner] Skipping pre-place rotation (err too high)")

        # Recompute q_place seeded from current q_prev (which has correct orientation after 3b)
        if place_xyzrpy is not None:
            q_place_new = solve_ik(
                ik_chain, place_xyzrpy, q_prev, joint_limits, device,
                num_seeds=NUM_SEEDS,
            )
            if q_place_new is not None:
                q_place = q_place_new

        # Phase 4: pre-place → place (descent)
        seg4 = _cartesian_segment("Phase 4: pre-place -> place", q_prev, q_place, target_rot=grasp_rot_place if pin_ik is not None else None)
        q_prev = _append(seg4)

        # Dwell at place (hold for gripper open)
        place_dwell_start = sum(s.shape[0] for s in segments)
        place_dwell = q_prev.unsqueeze(0).expand(DWELL_STEPS, -1).cpu()
        phase_labels.append((sum(s.shape[0] for s in segments), "Dwell: place (gripper open)", DWELL_STEPS))
        segments.append(place_dwell)

    # Phase 5: place → above-home (staging transit)
    p_home = _fk_pos(q_home)
    p_prev = _fk_pos(q_prev)
    z_above = max(p_prev[2], p_home[2]) + 0.08

    # IK for above-home: position-only from q_prev seed
    above_home_pos = np.array([p_home[0], p_home[1], z_above])
    q_ah_np, ah_err = pin_ik.solve(above_home_pos, q_init=q_prev.cpu().numpy())
    if ah_err >= 0.05:
        # Fallback: single staging phase to home
        seg5 = _cartesian_segment("Phase 5: place -> home", q_prev, q_home, staging=True)
        _append(seg5)
    else:
        q_above_home = torch.tensor(q_ah_np, dtype=torch.float32, device=device)
        seg5 = _cartesian_segment("Phase 5: place -> above-home", q_prev, q_above_home, staging=True)
        q_prev = _append(seg5)

        # Phase 5b: rotate at above-home to match home orientation
        q_full_home = pin.neutral(pin_ik.model)
        q_home_np = q_home.cpu().numpy()
        for i, ji in enumerate(pin_ik.joint_indices):
            q_full_home[ji] = float(q_home_np[i])
        pin.forwardKinematics(pin_ik.model, pin_ik.data, q_full_home)
        pin.updateFramePlacements(pin_ik.model, pin_ik.data)
        home_rot = pin_ik.data.oMf[pin_ik.frame_id].rotation.copy()

        q_ah_rot_np, rot_err = pin_ik.solve(
            above_home_pos, q_init=q_prev.cpu().numpy(),
            target_rot=home_rot, rot_weight=0.5,
        )
        if rot_err < 0.05:
            q_above_rotated = torch.tensor(q_ah_rot_np, dtype=torch.float32, device=device)
            seg5b = _cartesian_segment("Phase 5b: rotate at above-home", q_prev, q_above_rotated, target_rot=home_rot)
            q_prev = _append(seg5b)

        # Phase 6: above-home → home (descent)
        seg6 = _cartesian_segment("Phase 6: descend to home", q_prev, q_home)
        _append(seg6)

    full_traj = torch.cat(segments, dim=0)
    total_waypoints = full_traj.shape[0]
    t_plan = time.perf_counter() - t0

    print(f"[motion-planner] Trajectory: {total_waypoints} wp ({t_plan:.1f}s)")
    for ps, plbl, pn in phase_labels:
        print(f"[motion-planner]   wp {ps:4d}-{ps+pn:4d}: {plbl}")

    # Post-optimization collision validation
    if capsule_model is not None:
        _log_collision_check(
            validate_trajectory(
                ik_chain, capsule_model, full_traj, pc_tensor, SAFETY_MARGIN,
                table_plane=table_plane, table_polygon=table_polygon,
            ),
            total_waypoints,
        )

    close_rad = compute_gripper_close_rad(object_width)
    gripper_actions = [
        {"waypoint": 0, "rad": float(GRIPPER_OPEN_RAD)},  # open before descent
        {"waypoint": grasp_dwell_start, "rad": float(close_rad)},
    ]
    if q_place is not None:
        gripper_actions.append(
            {"waypoint": place_dwell_start, "rad": float(GRIPPER_OPEN_RAD)}
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
        "place_pose": place_xyzrpy.tolist() if place_xyzrpy is not None else None,
        "gripper_actions": json.dumps(gripper_actions),
        "phases": json.dumps([(s, lbl, n) for s, lbl, n in phase_labels]),
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
    return pc_tensor, pc_robot


def _setup_table_plane(optimizer, pc_robot, cam_t, cam_rot, latest_intrinsics,
                       latest_image_size, device):
    """Compute table plane from camera frustum, filter below-table points, and set on optimizer.

    Returns ``(table_plane, table_polygon, pc_filtered, pc_tensor_filtered)``
    where the last two are the point cloud and GPU tensor with below-table
    points removed.
    """
    if pc_robot is None or latest_intrinsics is None:
        return None, None, pc_robot, None
    fx, fy, cx, cy = latest_intrinsics
    w, h = latest_image_size
    table_plane, table_polygon = compute_table_plane(
        cam_t, cam_rot, fx, fy, cx, cy, w, h, pc_robot, device=device,
    )
    optimizer.set_table(table_plane, table_polygon)
    plane_z, bounds = table_plane
    # Filter below-table points (floor, arm reflections, table surface)
    pc_filtered, _mask = filter_below_table(pc_robot, plane_z)
    pc_tensor_filtered = pointcloud_to_tensor(pc_filtered, device) if len(pc_filtered) > 0 else None
    return table_plane, table_polygon, pc_filtered, pc_tensor_filtered


def _gripper_actions_to_array(gripper_actions, num_waypoints, ramp_steps=30):
    """Convert discrete gripper actions to continuous (T,) array for trajectory JSON.

    Values are normalized 0-1: 0 = fully open, 1 = fully closed.
    Ramps linearly over ``ramp_steps`` waypoints (~1s at 30Hz) between actions.
    """
    if not gripper_actions:
        return None
    arr = np.zeros(num_waypoints, dtype=np.float32)
    sorted_actions = sorted(gripper_actions, key=lambda a: a["waypoint"])

    def _rad_to_norm(rad):
        v = (rad - GRIPPER_OPEN_RAD) / (GRIPPER_CLOSED_RAD - GRIPPER_OPEN_RAD)
        return max(0.0, min(1.0, v))

    current = 0.0  # start fully open
    action_idx = 0
    ramp_start_val = 0.0
    ramp_target_val = 0.0
    ramp_start_wp = -ramp_steps  # no active ramp initially
    for i in range(num_waypoints):
        while action_idx < len(sorted_actions) and sorted_actions[action_idx]["waypoint"] <= i:
            ramp_start_val = current
            ramp_target_val = _rad_to_norm(float(sorted_actions[action_idx]["rad"]))
            ramp_start_wp = sorted_actions[action_idx]["waypoint"]
            action_idx += 1
        steps_in = i - ramp_start_wp
        if steps_in < ramp_steps:
            alpha = (steps_in + 1) / ramp_steps
            current = ramp_start_val + alpha * (ramp_target_val - ramp_start_val)
        else:
            current = ramp_target_val
        arr[i] = current
    return arr


def _set_playback(state, traj_np, traj_meta, node=None, pc_tensor=None):
    """Store a newly planned trajectory for playback and optionally export.

    If *node* is provided and playback mode is not "confirm", sends the
    trajectory as JSON on the ``trajectory_json`` output for external playback.
    In confirm mode, the JSON is stored in ``state["trajectory_doc"]`` and
    sent when the ``execute`` signal arrives.
    """
    state["trajectory"] = traj_np                        # (T, J)
    state["step"] = 0
    state["playing"] = PLAYBACK_MODE != "confirm"        # confirm mode waits for execute
    state["internal_done"] = False                       # internal tick playback finished
    state["play_arm"] = traj_meta.get("arm", "left")
    state["play_start"] = None                           # set on first tick
    state["play_dt"] = float(traj_meta.get("dt", 0.1))

    # Parse gripper actions for playback
    ga_str = traj_meta.get("gripper_actions", "")
    if ga_str:
        try:
            state["gripper_actions"] = json.loads(ga_str) if isinstance(ga_str, str) else ga_str
        except (json.JSONDecodeError, TypeError):
            state["gripper_actions"] = []
    else:
        state["gripper_actions"] = []
    state["gripper_fired"] = set()

    # Build trajectory JSON (v3) with gripper frames for external playback
    extra = {}
    for k in ("grasp_pose", "pregrasp_waypoint", "place_pose", "gripper_actions", "encoding"):
        if k in traj_meta:
            extra[k] = traj_meta[k]
    gripper_array = _gripper_actions_to_array(state["gripper_actions"], traj_np.shape[0])
    doc = build_trajectory_json(
        traj_np,
        arm=traj_meta.get("arm", "left"),
        dt=state["play_dt"],
        gripper=gripper_array,
        extra_metadata=extra,
    )
    state["trajectory_doc"] = doc

    export_path = EXPORT_PATH
    if not export_path and EXPORT_DIR:
        os.makedirs(EXPORT_DIR, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        arm_name = traj_meta.get("arm", "unknown")
        export_path = os.path.join(EXPORT_DIR, f"{ts}_{arm_name}_trajectory.json")
    if export_path:
        with open(export_path, "w") as f:
            json.dump(doc, f, indent=2)
        # Save point cloud for offline collision analysis
        if pc_tensor is not None:
            pc_path = export_path.replace('_trajectory.json', '_pointcloud.npy')
            np.save(pc_path, pc_tensor.cpu().numpy() if hasattr(pc_tensor, 'cpu') else pc_tensor)
            print(f'[motion-planner] Saved point cloud ({len(pc_tensor)} pts) -> {pc_path}')
        print(f"[motion-planner] Exported trajectory → {EXPORT_PATH}")

    # Send trajectory JSON for external playback (non-confirm mode)
    if node is not None and PLAYBACK_MODE != "confirm":
        json_str = json.dumps(doc)
        node.send_output("trajectory_json", pa.array([json_str]))


def main():
    if DEVICE == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif DEVICE not in ("cuda", "mps"):
        device = torch.device(DEVICE)
    else:
        # MPS (Apple Metal) is slower than CPU for trajectory optimization
        # due to kernel launch overhead on hundreds of small sequential ops.
        # CUDA gives real speedups; MPS does not.  Default to CPU.
        if DEVICE == "mps":
            print("[motion-planner] MPS requested but CPU is faster for this workload, using CPU")
        device = torch.device("cpu")
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
        "gripper_actions": [],  # list of {"waypoint": N, "rad": float}
        "gripper_fired": set(),
    }


    node = Node()
    print(
        f"[motion-planner] Ready ({', '.join(arms.keys())} arm(s)), "
        f"playback={PLAYBACK_MODE}, "
        f"export={'on → ' + (EXPORT_PATH or EXPORT_DIR) if (EXPORT_PATH or EXPORT_DIR) else 'off'}"
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

                # Fire gripper commands — ramp over GRIPPER_RAMP_STEPS for smooth motion
                GRIPPER_RAMP_STEPS = 30  # ~1s at 30Hz
                for action in playback["gripper_actions"]:
                    wp = action["waypoint"]
                    target_rad = float(action["rad"])
                    if step >= wp and step < wp + GRIPPER_RAMP_STEPS:
                        # First step: log and record start rad
                        if wp not in playback["gripper_fired"]:
                            playback["gripper_fired"].add(wp)
                            playback["gripper_ramp_start"] = playback.get("gripper_current_rad", GRIPPER_OPEN_RAD)
                        # Interpolate
                        alpha = (step - wp + 1) / GRIPPER_RAMP_STEPS
                        start_rad = playback.get("gripper_ramp_start", GRIPPER_OPEN_RAD)
                        current_rad = start_rad + alpha * (target_rad - start_rad)
                        playback["gripper_current_rad"] = current_rad
                        node.send_output(
                            "gripper_command",
                            pa.array([current_rad], type=pa.float32()),
                            metadata={"arm": playback["play_arm"]},
                        )

                if step >= T - 1 and not playback.get("internal_done"):
                    playback["internal_done"] = True
                    node.send_output("trajectory_status", pa.array([json.dumps({"status": "done"})]))

            # --- Arm trajectory status (external playback done) ---
            elif event_id in ("left_trajectory_status", "right_trajectory_status"):
                raw = event["value"][0].as_py()
                # Arm sends plain string "done" or JSON {"status": "done"}
                is_done = False
                if raw == "done":
                    is_done = True
                else:
                    try:
                        is_done = json.loads(raw).get("status") == "done"
                    except (json.JSONDecodeError, TypeError, IndexError):
                        pass
                if is_done and playback["playing"]:
                    playback["playing"] = False

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
                if playback["playing"]:
                    print("[motion-planner] Busy playing trajectory, ignoring target_pose")
                    node.send_output("trajectory_status", pa.array([json.dumps({
                        "status": "busy",
                    })]))
                    continue
                encoding = metadata.get("encoding", "jointstate")
                target_data = event["value"].to_numpy().astype(np.float32)

                # Determine target XY for automatic arm selection
                target_x, target_y = None, None
                if encoding in ("xyzrpy", "xyzquat"):
                    target_x = float(target_data[0])
                    target_y = float(target_data[1])
                elif encoding == "jaw_pixels":
                    result = _estimate_target_y(
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
                    if result is not None:
                        target_x, target_y = result

                arm_name = select_arm(metadata, target_y, target_x)
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

                # Compute table plane from camera frustum and filter
                table_plane, table_polygon, pc_robot, pc_tensor = _setup_table_plane(
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
                        pin_ik=arm.pin_ik,
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
                            pin_ik=arm.pin_ik,
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
                    _set_playback(playback, result[0], result[1], node=node, pc_tensor=pc_tensor)

                    traj_meta = result[1]
                    node.send_output("trajectory_status", pa.array([json.dumps({
                        "status": "ready",
                        "waypoints": result[0].shape[0],
                        "duration": round(result[0].shape[0] * float(traj_meta.get("dt", 0.1)), 1),
                        "arm": traj_meta.get("arm", "left"),
                    })]))
                elif encoding is not None:
                    node.send_output("trajectory_status", pa.array([json.dumps({
                        "status": "failed",
                        "reason": "Trajectory optimization failed",
                    })]))

            # --- Grasp result ---
            elif event_id == "grasp_result":
                if playback["playing"]:
                    print("[motion-planner] Busy playing trajectory, ignoring grasp_result")
                    node.send_output("trajectory_status", pa.array([json.dumps({
                        "status": "busy",
                    })]))
                    continue
                raw_text = event["value"][0].as_py()
                try:
                    data = json.loads(raw_text)
                except (json.JSONDecodeError, TypeError):
                    print(
                        f"[motion-planner] grasp_result: failed to parse JSON: {raw_text}"
                    )
                    continue

                # Forward failure status from selector
                if data.get("status") == "failed":
                    reason = data.get("reason", "unknown error")
                    print(f"[motion-planner] grasp_result: failed — {reason}")
                    node.send_output("trajectory_status", pa.array([json.dumps({
                        "status": "failed",
                        "reason": reason,
                    })]))
                    continue

                p1 = data.get("p1")
                p2 = data.get("p2")
                if not p1 or not p2 or len(p1) < 2 or len(p2) < 2:
                    print(f"[motion-planner] grasp_result: missing p1/p2: {data}")
                    continue

                u1 = float(p1[0])
                v1 = float(p1[1])
                u2 = float(p2[0])
                v2 = float(p2[1])
                print(
                    f"[motion-planner] grasp_result: jaw1=({u1:.0f},{v1:.0f}) "
                    f"jaw2=({u2:.0f},{v2:.0f})"
                )

                # Parse optional place target
                place_uv = None
                place_data = data.get("place_px")
                if place_data and len(place_data) >= 2:
                    place_u = float(place_data[0])
                    place_v = float(place_data[1])
                    place_uv = (place_u, place_v)
                    print(
                        f"[motion-planner] grasp_result: place=({place_u:.0f},{place_v:.0f})"
                    )

                # Arm selection: left side of image = left arm, right = right arm
                mid_u = (u1 + u2) / 2
                w, _ = latest_image_size
                preferred_arm = metadata.get("arm", "")
                if preferred_arm not in ("left", "right"):
                    preferred_arm = "left" if mid_u < w / 2 else "right"
                print(f"[motion-planner] Arm selection: pixel u={mid_u:.0f}/{w}, choosing {preferred_arm}")
                # Try preferred arm first, then the other
                arm_order = [preferred_arm]
                other_arm = "right" if preferred_arm == "left" else "left"
                if other_arm in arms:
                    arm_order.append(other_arm)

                pc_tensor, pc_robot = build_pointcloud(
                    latest_depth,
                    latest_intrinsics,
                    latest_image_size,
                    cam_t,
                    cam_rot,
                    device,
                )

                result = None
                fail_reason = None
                for arm_name in arm_order:
                    arm = arms.get(arm_name)
                    if arm is None:
                        continue
                    print(f"[motion-planner] Trying {arm_name} arm for grasp")

                    # Compute table plane (may differ per arm optimizer)
                    table_plane, table_polygon, pc_robot_filtered, pc_tensor_filtered = _setup_table_plane(
                        arm.optimizer, pc_robot, cam_t, cam_rot,
                        latest_intrinsics, latest_image_size, device,
                    )

                    result, fail_reason = plan_grasp_from_pixels(
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
                        pc_tensor_filtered,
                        arm.num_joints,
                        arm=arm_name,
                        table_plane=table_plane,
                        table_polygon=table_polygon,
                        place_uv=place_uv,
                        pin_ik=arm.pin_ik,
                    )
                    if result is not None:
                        break
                    print(f"[motion-planner] {arm_name} arm failed: {fail_reason}")

                if result is not None:
                    _set_playback(playback, result[0], result[1], node=node, pc_tensor=pc_tensor)

                    traj_meta = result[1]
                    node.send_output("trajectory_status", pa.array([json.dumps({
                        "status": "ready",
                        "waypoints": result[0].shape[0],
                        "duration": round(result[0].shape[0] * float(traj_meta.get("dt", 0.1)), 1),
                        "arm": traj_meta.get("arm", "left"),
                    })]))
                else:
                    node.send_output("trajectory_status", pa.array([json.dumps({
                        "status": "failed",
                        "reason": fail_reason or "Both arms failed",
                    })]))

            # --- Execute: start playback (confirm mode) ---
            elif event_id == "execute":
                if playback["trajectory"] is not None and not playback["playing"]:
                    playback["playing"] = True
                    playback["play_start"] = None
                    print("[motion-planner] Execution triggered")
                    # Send trajectory JSON for external playback (confirm mode)
                    doc = playback.get("trajectory_doc")
                    if doc is not None:
                        json_str = json.dumps(doc)
                        node.send_output("trajectory_json", pa.array([json_str]))
                
        elif event["type"] == "STOP":
            break


if __name__ == "__main__":
    main()
