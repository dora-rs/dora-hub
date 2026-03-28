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
    EXPORT_PATH:          If set, auto-save every planned trajectory to this JSON path
                          (overwrites on each plan).
    EXPORT_DIR:           If set, auto-save every trajectory to a unique timestamped
                          file in this directory (e.g. 20260317_143022_left_grasp.json).
    MAX_JOINT_STEP:       Max per-step joint angle change in radians (default 0.05).
    PLAYBACK:             "true"/"false"/"confirm" — enable built-in playback on tick.
                          "confirm" plans but waits for an ``execute`` input to start.
"""

import json
import os
import tempfile
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

try:
    from trac_ik import trac_ik as _trac_ik_mod
    _HAS_TRAC_IK = True
except ImportError:
    _HAS_TRAC_IK = False

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
from .grasp_utils import grasp_pose_from_jaw_pixels, place_pose_from_pixel, flip_rpy_around_jaw
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
PLACE_DEPTH_OFFSET = float(os.getenv("PLACE_DEPTH_OFFSET", "0.01"))
FLOOR_HEIGHT = float(os.getenv("FLOOR_HEIGHT", "0.005"))
APPROACH_MARGIN = float(os.getenv("APPROACH_MARGIN", "0.10"))
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
PLAYBACK_HZ = 36  # trajectory playback rate

# Gripper motor angle constants (imported from trajectory_json)
GRIPPER_TRAVEL_MM = 44.0      # jaw travel range in mm


def compute_gripper_close_rad(object_width_m):
    """Compute gripper close angle based on object width.

    Starts from fully closed and opens just enough to leave a gap
    slightly smaller than the object (5mm squeeze) for a firm hold.
    """
    gap_mm = max(0, object_width_m * 1000 - 25)  # 25mm squeeze margin
    # Fraction of travel to open from closed: 0 = fully closed, 1 = fully open
    open_frac = min(1.0, gap_mm / GRIPPER_TRAVEL_MM)
    return GRIPPER_CLOSED_RAD + open_frac * (GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD)


def compute_gripper_approach_rad(object_width_m):
    """Compute gripper approach opening — just wider than the object.

    Opens to object width + 10mm margin, instead of fully open.
    This avoids knocking nearby objects during approach.
    """
    gap_mm = object_width_m * 1000 + 7  # 7mm clearance
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
              target_rot=None, rot_weight=0.5, nullspace_weight=1.0,
              horiz_grip=False, w_grip=0.3):
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
            if horiz_grip:
                Ry = oMf.rotation[:, 1]  # TCP Y-axis
                world_up = np.array([0.0, 0.0, 1.0])
                grip_err = np.array([-w_grip * np.dot(Ry, world_up)])
                err = np.concatenate([err, grip_err])
                J_full = pin.computeFrameJacobian(
                    self.model, self.data, q, self.frame_id, pin.LOCAL_WORLD_ALIGNED
                )
                J_ang = J_full[3:, :]  # angular part (3×N)
                skew_y = np.array([
                    [0, -Ry[2], Ry[1]],
                    [Ry[2], 0, -Ry[0]],
                    [-Ry[1], Ry[0], 0],
                ])
                J_grip = (world_up @ skew_y @ J_ang).reshape(1, -1)
                J = np.vstack([J, w_grip * J_grip])
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


# ---------------------------------------------------------------------------
# TRAC-IK solver (fast 6-DOF IK with SQP + KDL fallback)
# ---------------------------------------------------------------------------
class TracIKSolver:
    """Wrapper around pytracik for OpenArm IK."""

    def __init__(self, urdf_path, base_link, ee_link, timeout=0.005, epsilon=1e-5,
                 solver_type="Speed"):
        # TracIK's URDF parser chokes on empty <collision> geometry tags —
        # strip them the same way build_chain() does for pytorch_kinematics.
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        for link in root.findall(".//link"):
            for tag in ("visual", "collision", "inertial"):
                for elem in link.findall(tag):
                    link.remove(elem)
        tmp = tempfile.NamedTemporaryFile(suffix=".urdf", delete=False, mode="w")
        tmp.write(ET.tostring(root, encoding="unicode"))
        tmp.close()
        self._solver = _trac_ik_mod.TracIK(
            base_link, ee_link, tmp.name,
            timeout=timeout, epsilon=epsilon, solver_type=solver_type,
        )
        self.dof = self._solver.dof

    def solve(self, target_pos, q_init, target_rot=None):
        """Solve 6-DOF IK.  Returns (q_arm, pos_err) like PinIK."""
        q_seed = np.asarray(q_init, dtype=np.float64)
        pos = np.asarray(target_pos, dtype=np.float64)
        rot = np.asarray(target_rot, dtype=np.float64) if target_rot is not None else np.eye(3)
        result = self._solver.ik(pos, rot, q_seed)
        if result is None:
            return None, float("inf")
        q = np.asarray(result, dtype=np.float32)
        fk_pos, _ = self._solver.fk(q.astype(np.float64))
        return q, float(np.linalg.norm(fk_pos - pos))


def _build_trac_ik(urdf_path, ee_link_name):
    """Build a TracIK solver for one arm (returns None if pytracik not installed)."""
    if not _HAS_TRAC_IK:
        return None
    try:
        solver = TracIKSolver(urdf_path, "world", ee_link_name,
                              timeout=0.005, solver_type="Speed")
        return solver
    except Exception as e:
        print(f"[motion-planner] TracIK init failed: {e} — will use PinIK + Adam")
        return None


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
    trac_ik: TracIKSolver = None
    trac_ik_dist: TracIKSolver = None  # Distance mode for keypoints


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

    # Build TracIK solvers: Speed for main IK, Distance for keypoints
    trac_solver = _build_trac_ik(urdf_path, ee_link)
    trac_dist = None
    if trac_solver is not None:
        print(f"[motion-planner] {name} arm: TracIK ready ({trac_solver.dof}J, Speed mode)")
        try:
            trac_dist = TracIKSolver(urdf_path, "world", ee_link,
                                     timeout=0.005, solver_type="Distance")
            print(f"[motion-planner] {name} arm: TracIK Distance ready")
        except Exception as e:
            print(f"[motion-planner] TracIK Distance init failed: {e}")

    return ArmConfig(name, ik_chain, optimizer, joint_limits, num_joints, current_joints,
                     pin_ik=pin_solver, trac_ik=trac_solver, trac_ik_dist=trac_dist)


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
    pt_robot = cam_rot.as_matrix() @ np.array([x_cam, y_cam, z]) + cam_t
    return float(pt_robot[1])


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
    horiz_grip=False,
    w_grip=0.15,
):
    """Batched Adam IK: all seeds optimised in parallel.

    Returns (best_q, best_pos_err) or (None, err) if no seed converges.
    """
    nj = len(lower)
    if torch.is_tensor(current_joints):
        current_joints = current_joints.to(device)
    else:
        current_joints = torch.tensor(current_joints, dtype=torch.float32, device=device)
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

        if horiz_grip:
            fk_mat = fk.get_matrix()  # (num_seeds, 4, 4)
            grip_z = fk_mat[:, 2, 1]  # TCP Y-axis Z-component
            loss = loss + w_grip * (grip_z ** 2).sum()

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
    num_seeds=4,
    max_iters=500,
    pos_threshold=0.005,
    position_only=False,
    pin_ik=None,
    horiz_grip=False,
    trac_ik_solver=None,
):
    """Solve IK with orientation awareness.

    Strategy:
      1. PinIK position-only → seed for TracIK
      2. TracIK 6-DOF refinement from PinIK seed → exact orientation
      3. Adam 6-DOF (rot_weight=0.1) → approximate orientation
      4. If none achieve orientation: return None (fail explicitly)

    When ``position_only=True``, skips orientation and returns first
    position solution found.

    Returns q_goal tensor or None.
    """
    t_ik_start = time.perf_counter()
    target_xyz = target_xyzrpy[:3]
    q_init_np = current_joints.cpu().numpy() if torch.is_tensor(current_joints) else np.asarray(current_joints)

    # ---- Position-only fast path (TracIK with current rotation) ----
    if position_only:
        if trac_ik_solver is not None:
            # Use current EE rotation as target — just move position
            _, cur_rot = trac_ik_solver._solver.fk(q_init_np.astype(np.float64))
            q_np, err = trac_ik_solver.solve(
                np.array(target_xyz, dtype=np.float64),
                q_init=q_init_np, target_rot=cur_rot,
            )
            if q_np is not None and err <= pos_threshold:
                return torch.tensor(q_np, dtype=torch.float32, device=device)
        # PinIK fallback for position-only
        if pin_ik is not None:
            q_np, err = pin_ik.solve(np.array(target_xyz, dtype=np.float64), q_init=q_init_np,
                                     horiz_grip=horiz_grip)
            if err <= pos_threshold:
                q_t = torch.tensor(q_np, dtype=torch.float32, device=device)
                ee_check = _fk_pos_chain(ik_chain, q_t)
                cross_err = float(np.linalg.norm(ee_check - target_xyz))
                if cross_err <= pos_threshold:
                    return q_t
        print(f"[IK] Position-only failed")
        return None

    # ---- 6-DOF path: orientation matters ----
    target_rot_np = ScipyRotation.from_euler("XYZ", target_xyzrpy[3:6]).as_matrix()

    # Step 1: Try TracIK directly from current_joints (fast, ~0.1ms)
    if trac_ik_solver is not None:
        q_refined, ref_err = trac_ik_solver.solve(
            np.array(target_xyz, dtype=np.float64),
            q_init=q_init_np,
            target_rot=target_rot_np,
        )
        if q_refined is not None and ref_err <= pos_threshold:
            q_t = torch.tensor(q_refined, dtype=torch.float32, device=device)
            ee_check = _fk_pos_chain(ik_chain, q_t)
            cross_err = float(np.linalg.norm(ee_check - target_xyz))
            if cross_err <= pos_threshold:
                t_ik = time.perf_counter() - t_ik_start
                print(f"[IK] TracIK 6-DOF OK (direct): err={ref_err*1000:.2f}mm ({t_ik*1000:.0f}ms)")
                return q_t

    # Step 2: TracIK failed — get position seed via PinIK, then retry TracIK
    q_seed_np = q_init_np
    if pin_ik is not None:
        q_np, err = pin_ik.solve(np.array(target_xyz, dtype=np.float64), q_init=q_init_np,
                                 horiz_grip=horiz_grip)
        if err <= pos_threshold * 5:  # looser threshold — just need a nearby seed
            q_seed_np = q_np

    if trac_ik_solver is not None:
        q_refined, ref_err = trac_ik_solver.solve(
            np.array(target_xyz, dtype=np.float64),
            q_init=q_seed_np,
            target_rot=target_rot_np,
        )
        if q_refined is not None and ref_err <= pos_threshold:
            q_t = torch.tensor(q_refined, dtype=torch.float32, device=device)
            ee_check = _fk_pos_chain(ik_chain, q_t)
            cross_err = float(np.linalg.norm(ee_check - target_xyz))
            if cross_err <= pos_threshold:
                t_ik = time.perf_counter() - t_ik_start
                print(f"[IK] TracIK 6-DOF OK (Adam seed): err={ref_err*1000:.2f}mm ({t_ik:.2f}s)")
                return q_t

    # Step 3: TracIK failed even with Adam seed — orientation is unreachable
    t_ik = time.perf_counter() - t_ik_start
    print(f"[IK] FAILED: target orientation unreachable at "
          f"xyz=[{target_xyz[0]:.3f},{target_xyz[1]:.3f},{target_xyz[2]:.3f}] "
          f"({t_ik:.1f}s)")
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
    """Log collision validation results, split by source."""
    if not collisions:
        print(f"[motion-planner] Collision: 0/{total_waypoints} wp — clear")
        return

    pole_col = [(t, n, d) for t, n, d in collisions if "[body]" in n]
    table_col = [(t, n, d) for t, n, d in collisions if "[table]" in n]
    pc_col = [(t, n, d) for t, n, d in collisions if "[body]" not in n and "[table]" not in n]

    parts = []
    if pole_col:
        pole_wps = len(set(c[0] for c in pole_col))
        pole_worst = min(pole_col, key=lambda c: c[2])
        parts.append(f"pole: {pole_wps}wp/{-pole_worst[2]*1000:.1f}mm")
    if pc_col:
        pc_wps = len(set(c[0] for c in pc_col))
        pc_worst = min(pc_col, key=lambda c: c[2])
        parts.append(f"pointcloud: {pc_wps}wp/{-pc_worst[2]*1000:.1f}mm")
    if table_col:
        tbl_wps = len(set(c[0] for c in table_col))
        tbl_worst = min(table_col, key=lambda c: c[2])
        parts.append(f"table: {tbl_wps}wp/{-tbl_worst[2]*1000:.1f}mm")

    total_wps = len(set(c[0] for c in collisions))
    print(f"[motion-planner] Collision: {total_wps}/{total_waypoints} wp — {', '.join(parts)}")


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
    dt = 1.0 / PLAYBACK_HZ
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
    place_mask_bbox=None,
    place_mask=None,
    pick_mask=None,
    pin_ik=None,
    trac_ik_solver=None,
    trac_ik_dist=None,
    latest_image=None,
    action="",
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

    # --- Heading sweep: try multiple approach headings until IK succeeds ---
    if arm == "left":
        heading_candidates = [-135, -120, -105, -90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90, -150, -165, -180]
    else:
        heading_candidates = [-135, -150, -165, -180, -195, -210, -225, -240, -255, -270, -120, -105, -90]

    grasp_approach_angle = 70.0 if action == "pour" else APPROACH_ANGLE_DEG

    # Per-arm calibration offset (metres), calibrated at -135° heading.
    # Rotate the XY component when using a different heading so the
    # correction stays aligned with the gripper frame.
    CAL_OFFSET_XY = np.array([0.03, -0.03])  # at reference heading -135°
    CAL_REF_HEADING = -135.0

    grasp_xyzrpy = None
    pregrasp_xyzrpy = None
    object_top_z = None
    object_width = None
    q_pregrasp = None
    q_grasp = None
    q_pregrasp_travel = None
    chosen_heading = None

    for heading in heading_candidates:
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
            approach_angle_deg=grasp_approach_angle,
            approach_heading_deg=float(heading),
            pick_mask=pick_mask,
        )
        if result is None:
            if heading == heading_candidates[0]:
                print("[motion-planner] grasp: invalid depth at jaw points, skipping")
                return None, "Invalid depth at grasp point"
            continue

        g_xyzrpy, pg_xyzrpy, obj_top_z, obj_w = result
        # Rotate calibration offset XY by heading difference from reference
        delta_rad = np.radians(float(heading) - CAL_REF_HEADING)
        cos_d, sin_d = np.cos(delta_rad), np.sin(delta_rad)
        rotated_xy = np.array([
            cos_d * CAL_OFFSET_XY[0] - sin_d * CAL_OFFSET_XY[1],
            sin_d * CAL_OFFSET_XY[0] + cos_d * CAL_OFFSET_XY[1],
        ])
        cal_offset = np.array([rotated_xy[0], rotated_xy[1], 0.0, 0.0, 0.0, 0.0])
        g_xyzrpy = g_xyzrpy + cal_offset
        pg_xyzrpy = pg_xyzrpy + cal_offset

        q_pg = solve_ik(
            ik_chain, pg_xyzrpy, current_joints, joint_limits, device,
            num_seeds=NUM_SEEDS, horiz_grip=True, trac_ik_solver=trac_ik_solver,
        )
        if q_pg is None:
            print(f"[motion-planner] heading {heading}° pre-grasp IK failed, trying next")
            continue

        # Baby-step IK from pre-grasp down to grasp (~2mm per step)
        descent_mm = np.linalg.norm(g_xyzrpy[:3] - pg_xyzrpy[:3]) * 1000
        n_baby = max(int(descent_mm / 2.0), 5)
        q_prev_baby = q_pg
        q_g = None
        last_good_alpha = 0.0
        for bi in range(1, n_baby + 1):
            alpha = bi / n_baby
            baby_xyzrpy = pg_xyzrpy * (1 - alpha) + g_xyzrpy * alpha
            q_baby = solve_ik(ik_chain, baby_xyzrpy, q_prev_baby, joint_limits, device,
                              num_seeds=NUM_SEEDS, horiz_grip=True, trac_ik_solver=trac_ik_solver)
            if q_baby is None:
                break
            q_prev_baby = q_baby
            last_good_alpha = alpha
            if bi == n_baby:
                q_g = q_baby
        if q_g is None:
            remaining_mm = descent_mm * (1.0 - last_good_alpha)
            print(f"[motion-planner] heading {heading}° grasp IK failed at {remaining_mm:.0f}mm from target ({n_baby} steps, {descent_mm:.0f}mm descent), trying next")
            continue

        # Both IK succeeded
        grasp_xyzrpy = g_xyzrpy
        pregrasp_xyzrpy = pg_xyzrpy
        object_top_z = obj_top_z
        object_width = obj_w
        q_pregrasp = q_pg
        q_grasp = q_g
        chosen_heading = heading
        if heading != heading_candidates[0]:
            print(f"[motion-planner] heading {heading}° succeeded (default {heading_candidates[0]}° failed)")
        break

    if q_grasp is None:
        print(f"[motion-planner] grasp: IK failed for all headings {heading_candidates}, skipping")
        return None, "IK failed for grasp pose (all headings unreachable)"

    # Solve a travel pre-grasp at 70° orientation for smooth transit from home.
    # Uses the same heading that succeeded for the grasp.
    if APPROACH_ANGLE_DEG < 60.0:
        travel_angle = 70.0
        heading_rad = np.radians(float(chosen_heading))
        horiz = np.array([np.cos(heading_rad), np.sin(heading_rad), 0.0])
        angle_rad = np.radians(travel_angle)
        travel_approach = np.sin(angle_rad) * horiz + np.cos(angle_rad) * np.array([0.0, 0.0, -1.0])
        travel_approach = travel_approach / np.linalg.norm(travel_approach)
        travel_z_ee = travel_approach
        up = np.array([0.0, 0.0, 1.0])
        travel_y = np.cross(travel_z_ee, up)
        travel_y = travel_y / np.linalg.norm(travel_y)
        travel_x = np.cross(travel_y, travel_z_ee)
        travel_x = travel_x / np.linalg.norm(travel_x)
        travel_y = np.cross(travel_z_ee, travel_x)
        from scipy.spatial.transform import Rotation as R_scipy
        travel_rot = np.stack([travel_x, travel_y, travel_z_ee], axis=1)
        travel_rpy = R_scipy.from_matrix(travel_rot).as_euler("XYZ")
        travel_pregrasp = pregrasp_xyzrpy.copy()
        travel_pregrasp[3:6] = travel_rpy
        q_pregrasp_travel = solve_ik(
            ik_chain, travel_pregrasp, current_joints, joint_limits, device,
            num_seeds=NUM_SEEDS, horiz_grip=True, trac_ik_solver=trac_ik_solver,
        )
        if q_pregrasp_travel is not None:
            print(f"[motion-planner] travel pre-grasp at 70° OK")
        else:
            print(f"[motion-planner] travel pre-grasp at 70° failed, using direct")

    # FK verification: check where the IK solution actually places the EE
    if trac_ik_solver is not None:
        fk_pos, fk_rot = trac_ik_solver._solver.fk(q_grasp.cpu().numpy().astype(np.float64))
        fk_err = np.linalg.norm(fk_pos - grasp_xyzrpy[:3]) * 1000
        print(f"[motion-planner] FK verify grasp: target={np.round(grasp_xyzrpy[:3], 4)} "
              f"fk={np.round(fk_pos, 4)} err={fk_err:.1f}mm")
        fk_pregrasp_pos, _ = trac_ik_solver._solver.fk(q_pregrasp.cpu().numpy().astype(np.float64))
        fk_pre_err = np.linalg.norm(fk_pregrasp_pos - pregrasp_xyzrpy[:3]) * 1000
        print(f"[motion-planner] FK verify pregrasp: target={np.round(pregrasp_xyzrpy[:3], 4)} "
              f"fk={np.round(fk_pregrasp_pos, 4)} err={fk_pre_err:.1f}mm")

    # --- Pick-and-place: solve IK for place/preplace poses ---
    q_place = None
    q_preplace = None
    place_xyzrpy = None

    # Pre-compute flipped RPY: Rz_local(90°) = roll around approach/finger axis
    # Done at a closer position for workspace, then move back to place.
    flipped_rpy = None
    if action in ("flip", "pour") and q_grasp is not None:
        from scipy.spatial.transform import Rotation as R_flip_rot
        R_grasp_mat = R_flip_rot.from_euler("XYZ", grasp_xyzrpy[3:6]).as_matrix()
        # Start with 90° roll for validation
        Rz_90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        R_rolled = R_grasp_mat @ Rz_90
        flipped_rpy = R_flip_rot.from_matrix(R_rolled).as_euler("XYZ").astype(np.float32)
        print(f"[motion-planner] flip: grasp_rpy={np.round(np.degrees(grasp_xyzrpy[3:6]), 1)} "
              f"-> place_rpy={np.round(np.degrees(flipped_rpy), 1)} (Rz 90° roll)")

    if place_uv is not None:
        place_u, place_v = place_uv
        # For flip/pour: use FK-computed flipped orientation (J6 at opposite limit)
        if action in ("flip", "pour") and flipped_rpy is not None:
            place_rpy = flipped_rpy
            print(f"[motion-planner] flip: place_rpy={np.round(np.degrees(place_rpy), 1)} (from FK J6 flip)")
            place_approach_margin = APPROACH_MARGIN
        else:
            place_rpy = grasp_xyzrpy[3:6]
            place_approach_margin = APPROACH_MARGIN
        place_result = place_pose_from_pixel(
            place_u, place_v,
            latest_depth, fx, fy, cx, cy,
            cam_t, cam_rot,
            grasp_rpy=place_rpy,
            width=w, height=h,
            place_depth_offset=PLACE_DEPTH_OFFSET,
            floor_height=FLOOR_HEIGHT,
            approach_margin=place_approach_margin,
            jaw_contact_depth=JAW_CONTACT_DEPTH,
            mask_bbox=None if action in ("flip", "pour") else place_mask_bbox,
            place_mask=None if action in ("flip", "pour") else place_mask,
            pick_mask=pick_mask,
        )
        if place_result is not None:
            place_xyzrpy, preplace_xyzrpy = place_result
            place_xyzrpy = place_xyzrpy + cal_offset
            preplace_xyzrpy = preplace_xyzrpy + cal_offset
            if action not in ("flip", "pour"):
                q_preplace = solve_ik(
                    ik_chain, preplace_xyzrpy, q_grasp, joint_limits, device,
                    num_seeds=NUM_SEEDS, horiz_grip=True, trac_ik_solver=trac_ik_solver,
                )
                if q_preplace is not None:
                    q_place = solve_ik(
                        ik_chain, place_xyzrpy, q_preplace, joint_limits, device,
                        num_seeds=NUM_SEEDS, horiz_grip=True, trac_ik_solver=trac_ik_solver,
                    )
                if q_place is None:
                    print("[motion-planner] grasp: Place IK failed, rejecting")
                    return None, "Place target unreachable (IK failed)"
        else:
            print("[motion-planner] grasp: Place depth invalid, rejecting")
            return None, "Place target has invalid depth"


    # Flip/pour waypoints: move to closer position, do the roll, move back
    q_flip_waypoints = []
    descent_waypoints = []
    if action in ("flip", "pour") and place_xyzrpy is not None and flipped_rpy is not None:
        from scipy.spatial.transform import Rotation as R_flip_rot
        # Build travel orientation (90° = fully horizontal at -135° heading)
        heading_rad = np.radians(-135.0)
        horiz = np.array([np.cos(heading_rad), np.sin(heading_rad), 0.0])
        angle_rad = np.radians(90.0)
        t_z = np.sin(angle_rad) * horiz + np.cos(angle_rad) * np.array([0, 0, -1.0])
        t_z /= np.linalg.norm(t_z)
        up = np.array([0.0, 0.0, 1.0])
        t_y = np.cross(t_z, up); t_y /= np.linalg.norm(t_y)
        t_x = np.cross(t_y, t_z); t_x /= np.linalg.norm(t_x)
        t_y = np.cross(t_z, t_x)
        R_travel = np.stack([t_x, t_y, t_z], axis=1)
        travel_rpy = R_flip_rot.from_matrix(R_travel).as_euler("XYZ").astype(np.float32)

        import torch as _torch

        # Strategy: try (angle, height_extra) combinations until one works
        # Prefer: target angle at low height, then go higher
        if action == "pour":
            strategies = [
                (130, 0.06), (130, 0.10), (130, 0.14),
            ]
        else:
            # Flip: 180° at increasing heights
            strategies = [
                (180, 0.04), (180, 0.08), (180, 0.12),
            ]

        q_close = None
        q_rolled = None
        rolled_rpy = None
        for roll_deg, height_extra in strategies:
            # Pour: tilt above the pour target (place), not the pick position
            if action == "pour" and place_xyzrpy is not None:
                close_pos = place_xyzrpy[:3].copy()
                # Offset pour position slightly toward the arm for better reach
                if arm == "left":
                    close_pos[0] += 0.01   # slightly forward
                    close_pos[1] -= 0.01   # slightly left (toward left arm)
                else:
                    close_pos[0] += 0.01
                    close_pos[1] += 0.01   # slightly right (toward right arm)
                # Use pregrasp Z as base (higher than place surface)
                close_pos[2] = max(pregrasp_xyzrpy[2], place_xyzrpy[2]) + height_extra
            else:
                close_pos = pregrasp_xyzrpy[:3].copy()
                close_pos[2] += height_extra

            # Solve travel orientation IK at this height
            # close_pos already includes cal_offset from pregrasp_xyzrpy
            close_xyzrpy = np.concatenate([close_pos, travel_rpy + cal_offset[3:6]])
            q_close_try = solve_ik(ik_chain, close_xyzrpy, q_grasp, joint_limits, device,
                                   num_seeds=NUM_SEEDS * 2, horiz_grip=False, trac_ik_solver=trac_ik_solver)
            if q_close_try is None:
                print(f"[motion-planner] flip: travel IK failed at +{height_extra:.2f}m, skipping")
                continue

            # Compute rolled orientation (right arm rolls opposite direction)
            roll_sign = -1.0 if arm == "right" else 1.0
            c, s = np.cos(np.radians(roll_sign * roll_deg)), np.sin(np.radians(roll_sign * roll_deg))
            Rz_roll = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
            R_rolled = R_travel @ Rz_roll
            rolled_rpy_try = R_flip_rot.from_matrix(R_rolled).as_euler("XYZ").astype(np.float32)
            pos_target = close_pos.astype(np.float64)

            print(f"[motion-planner] flip: trying {roll_deg}° at +{height_extra:.2f}m "
                  f"(z={close_pos[2]:.3f}m) arm={arm}...")

            # Baby-step TracIK: incrementally rotate from 0° to target in 10° steps
            # Save ALL intermediate solutions as waypoints for smooth trajectory
            step_deg = 1.0
            n_steps = int(abs(roll_sign * roll_deg) / step_deg)
            q_prev_step = q_close_try
            baby_step_waypoints = []
            q_rolled_try = None
            for s in range(1, n_steps + 1):
                angle_s = roll_sign * step_deg * s
                c_s, s_s = np.cos(np.radians(angle_s)), np.sin(np.radians(angle_s))
                Rz_s = np.array([[c_s, -s_s, 0], [s_s, c_s, 0], [0, 0, 1]], dtype=np.float64)
                R_s = R_travel @ Rz_s
                rpy_s = R_flip_rot.from_matrix(R_s).as_euler("XYZ").astype(np.float32)
                xyzrpy_s = np.concatenate([close_pos, rpy_s + cal_offset[3:6]])
                q_s = solve_ik(ik_chain, xyzrpy_s, q_prev_step, joint_limits, device,
                               num_seeds=NUM_SEEDS, horiz_grip=False, trac_ik_solver=trac_ik_solver)
                if q_s is not None:
                    # Reject solutions with large joint jumps (config switch)
                    q_prev_np = q_prev_step.cpu().numpy() if _torch.is_tensor(q_prev_step) else np.asarray(q_prev_step)
                    q_s_np = q_s.cpu().numpy() if _torch.is_tensor(q_s) else np.asarray(q_s)
                    max_jump = float(np.max(np.abs(q_s_np - q_prev_np)))
                    if max_jump > 0.1:  # ~6° max per 1° EE step
                        print(f"[motion-planner] flip: baby-step {angle_s:.0f}° rejected "
                              f"(joint jump {np.degrees(max_jump):.1f}°), stopping")
                        break
                    q_prev_step = q_s
                    baby_step_waypoints.append(q_s)
                else:
                    print(f"[motion-planner] flip: TracIK baby-step failed at {angle_s:.0f}°")
                    break
            else:
                q_rolled_try = q_prev_step
                print(f"[motion-planner] flip: TracIK baby-step solved {roll_deg}° in {n_steps} steps!")

            if q_rolled_try is None and baby_step_waypoints:
                # Use whatever angle TracIK reached (partial flip is better than nothing)
                q_rolled_try = baby_step_waypoints[-1]
                actual_deg = len(baby_step_waypoints) * step_deg
                print(f"[motion-planner] flip: TracIK reached {actual_deg:.0f}° (target was {roll_deg}°)")
            if q_rolled_try is not None:
                q_close = q_close_try
                q_rolled = q_rolled_try
                rolled_rpy = rolled_rpy_try
                print(f"[motion-planner] flip: {roll_deg}° at +{height_extra:.2f}m OK!")
                break
            else:
                print(f"[motion-planner] flip: {roll_deg}° at +{height_extra:.2f}m failed")

        if q_rolled is not None:
            if baby_step_waypoints:
                q_flip_waypoints = [q_close] + baby_step_waypoints
                print(f"[motion-planner] flip: using {len(baby_step_waypoints)} baby-step waypoints")
            else:
                q_flip_waypoints = [q_close, q_rolled]

        else:
            return None, f"{'Pour' if action == 'pour' else 'Flip'} failed: no angle/height combination worked"

        # Descent: TracIK baby steps, each seeded from previous.
        # On joint jump or IK failure, subdivide into smaller steps before giving up.
        # Pour stays at height (no descent needed).
        q_rolled_wp = q_flip_waypoints[-1]
        q_preplace = q_rolled_wp
        q_place = q_rolled_wp  # default: drop from height
        descent_waypoints = []
        q_prev_descent = q_rolled_wp
        target_z = grasp_xyzrpy[2] + 0.02
        current_z = close_pos[2]
        base_step_z = 0.01
        stop_descent = False
        while action != "pour" and current_z - base_step_z >= target_z and not stop_descent:
            next_z = current_z - base_step_z
            # Try the full step first, then subdivide on failure
            for n_sub in [1, 2, 4]:
                sub_step = base_step_z / n_sub
                sub_wps = []
                q_sub_prev = q_prev_descent
                sub_ok = True
                for si in range(n_sub):
                    sub_z = current_z - sub_step * (si + 1)
                    descent_pos = close_pos.copy()
                    descent_pos[2] = sub_z
                    descent_xyzrpy = np.concatenate([descent_pos, rolled_rpy + cal_offset[3:6]])
                    q_step = solve_ik(ik_chain, descent_xyzrpy, q_sub_prev, joint_limits, device,
                                      num_seeds=NUM_SEEDS, horiz_grip=False, trac_ik_solver=trac_ik_solver)
                    if q_step is not None:
                        q_step_np = q_step.cpu().numpy() if torch.is_tensor(q_step) else np.asarray(q_step)
                        q_prev_np = q_sub_prev.cpu().numpy() if torch.is_tensor(q_sub_prev) else np.asarray(q_sub_prev)
                        if float(np.max(np.abs(q_step_np - q_prev_np))) > 0.1:
                            sub_ok = False
                            break
                        sub_wps.append(q_step)
                        q_sub_prev = q_step
                    else:
                        sub_ok = False
                        break
                if sub_ok:
                    descent_waypoints.extend(sub_wps)
                    q_prev_descent = sub_wps[-1]
                    current_z = next_z
                    break
            else:
                print(f"[motion-planner] flip: descent stopped at z={current_z:.3f}m (subdivide failed)")
                stop_descent = True
        if descent_waypoints:
            q_place = descent_waypoints[-1]
            print(f"[motion-planner] flip: descended {len(descent_waypoints)}cm "
                  f"to z={current_z:.3f}m")
        else:
            print(f"[motion-planner] flip: no descent possible, dropping from height")

    q_start = current_joints.clone()
    t_plan_start = time.perf_counter()

    result = _build_pick_place_trajectory(
        node, optimizer, ik_chain, capsule_model,
        q_start, q_pregrasp, q_grasp, q_preplace, q_place,
        pc_tensor, num_joints, arm, object_width,
        grasp_xyzrpy, place_xyzrpy,
        q_pregrasp_travel=q_pregrasp_travel,
        joint_limits=joint_limits, device=device,
        table_plane=table_plane, table_polygon=table_polygon,
        pin_ik=pin_ik,
        trac_ik_solver=trac_ik_solver,
        trac_ik_dist=trac_ik_dist,
        q_flip_waypoints=q_flip_waypoints,
        descent_waypoints=descent_waypoints if action == "flip" else [],
        action=action,
        chosen_heading=chosen_heading,
    )
    if result is None:
        return None, "Trajectory planning failed"

    # Save debug image with pick/place targets drawn on RGB (or depth fallback)
    if EXPORT_DIR:
        try:
            import cv2
            if latest_image is not None:
                img_flat = latest_image
                if img_flat.size == w * h * 3:
                    debug_img = img_flat.reshape(h, w, 3).copy()
                    debug_img = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)
                else:
                    debug_img = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                depth_2d = latest_depth.reshape(h, w) if latest_depth.ndim == 1 else latest_depth
                depth_vis = np.clip(depth_2d.astype(np.float32) / 1000.0, 0, 2.0)
                depth_vis = (depth_vis / 2.0 * 255).astype(np.uint8)
                debug_img = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
            # Draw pick (jaw) points — green
            iu1, iv1 = int(round(u1)), int(round(v1))
            iu2, iv2 = int(round(u2)), int(round(v2))
            cv2.circle(debug_img, (iu1, iv1), 8, (0, 255, 0), 2)
            cv2.circle(debug_img, (iu2, iv2), 8, (0, 255, 0), 2)
            cv2.line(debug_img, (iu1, iv1), (iu2, iv2), (0, 255, 0), 2)
            mid_u, mid_v = (iu1 + iu2) // 2, (iv1 + iv2) // 2
            cv2.putText(debug_img, "PICK", (mid_u - 20, mid_v - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # Draw free/occupied space within container and corrected place target
            if place_mask is not None and place_xyzrpy is not None:
                from .grasp_utils import find_free_place_target
                fp_result = find_free_place_target(
                    place_mask, latest_depth, fx, fy, cx, cy,
                    cam_t, cam_rot, pick_mask=pick_mask, width=w, height=h,
                )
                if fp_result is not None:
                    pm_2d = place_mask if place_mask.ndim == 2 else place_mask.reshape(h, w)
                    place_bin = (pm_2d > 0).astype(np.uint8)
                    depth_2d_dbg = latest_depth.reshape(h, w).astype(np.float32)
                    valid_dbg = (depth_2d_dbg > 100) & (depth_2d_dbg < 5000) & (place_bin > 0)
                    depths_dbg = depth_2d_dbg[valid_dbg]
                    if len(depths_dbg) > 0:
                        floor_d = np.percentile(depths_dbg, 85)
                        free_px = (np.abs(depth_2d_dbg - floor_d) <= 15) & valid_dbg
                        occupied_px = valid_dbg & ~free_px
                        # Green overlay for free space
                        green_overlay = np.zeros_like(debug_img)
                        green_overlay[:, :] = (0, 200, 0)
                        debug_img[free_px] = cv2.addWeighted(
                            debug_img, 0.6, green_overlay, 0.4, 0)[free_px]
                        # Red overlay for occupied space
                        red_overlay = np.zeros_like(debug_img)
                        red_overlay[:, :] = (0, 0, 200)
                        debug_img[occupied_px] = cv2.addWeighted(
                            debug_img, 0.6, red_overlay, 0.4, 0)[occupied_px]
                    fp_u, fp_v, _, fits = fp_result
                    # Stamp pick mask outline at the chosen place position
                    if pick_mask is not None and fits:
                        pk_2d = pick_mask if pick_mask.ndim == 2 else pick_mask.reshape(h, w)
                        pk_bin = (pk_2d > 0).astype(np.uint8)
                        pk_ys, pk_xs = np.where(pk_bin > 0)
                        if len(pk_ys) > 0:
                            pk_cy, pk_cx = int(pk_ys.mean()), int(pk_xs.mean())
                            dy, dx = fp_v - pk_cy, fp_u - pk_cx
                            # Create shifted mask and draw its contour
                            M = np.float32([[1, 0, dx], [0, 1, dy]])
                            shifted_mask = cv2.warpAffine(pk_bin, M, (w, h))
                            cnts_pk, _ = cv2.findContours(shifted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(debug_img, cnts_pk, -1, (0, 255, 255), 3)
                    # Draw corrected place target crosshair
                    label = "PLACE(free)" if fits else "PLACE(no-fit)"
                    sz = 18
                    cv2.line(debug_img, (fp_u - sz, fp_v), (fp_u + sz, fp_v), (0, 200, 255), 3)
                    cv2.line(debug_img, (fp_u, fp_v - sz), (fp_u, fp_v + sz), (0, 200, 255), 3)
                    cv2.circle(debug_img, (fp_u, fp_v), 14, (0, 200, 255), 2)
                    cv2.putText(debug_img, label, (fp_u - 50, fp_v - 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            # Draw original place point — blue
            if place_uv is not None:
                pu, pv = int(round(place_uv[0])), int(round(place_uv[1]))
                cv2.circle(debug_img, (pu, pv), 12, (255, 100, 0), 2)
                cv2.putText(debug_img, "PLACE(orig)", (pu - 40, pv - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)
            # Draw place mask bbox — cyan
            if place_mask_bbox is not None:
                x0, y0, x1, y1 = [int(v) for v in place_mask_bbox]
                cv2.rectangle(debug_img, (x0, y0), (x1, y1), (255, 255, 0), 1)
            cv2.putText(debug_img, f"arm={arm}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            debug_path = str(Path(EXPORT_DIR) / "pick_place_targets.png")
            cv2.imwrite(debug_path, debug_img)
            print(f"[motion-planner] Debug image: {debug_path}")
        except Exception as e:
            print(f"[motion-planner] Debug image failed: {e}")

    return result, None


def _smooth_dwell_boundaries(traj, max_step, ramp_waypoints=8):
    """Add deceleration/acceleration ramps at dwell boundaries.

    Detects dwell regions (consecutive identical waypoints) and replaces the
    last *ramp_waypoints* before each dwell and first *ramp_waypoints* after
    each dwell with cosine-eased ramps.  This prevents the velocity
    discontinuity where the arm goes from full speed to zero in one timestep.

    The dwell position is preserved exactly — only the approach/departure
    waypoints are modified to create smooth velocity profiles.
    """
    traj_np = traj.numpy().copy() if hasattr(traj, 'numpy') else np.array(traj, copy=True)
    T = traj_np.shape[0]
    if T < 3:
        return traj

    # Find dwell regions: runs of ≥2 identical waypoints
    diffs = np.max(np.abs(np.diff(traj_np, axis=0)), axis=1)  # (T-1,)
    is_static = diffs < 1e-8

    # Collect dwell intervals as (first_static_idx, last_static_idx+1)
    dwells = []
    t = 0
    while t < T - 1:
        if is_static[t]:
            start = t
            while t < T - 1 and is_static[t]:
                t += 1
            # is_static[start..t-1] are True, meaning traj[start]==...==traj[t]
            dwells.append((start, t + 1))  # half-open: traj[start:t+1] are all equal
        else:
            t += 1

    original = traj_np.copy()
    for dwell_start, dwell_end in dwells:
        dwell_pos = original[dwell_start]

        # Entry ramp: ease the preceding motion segment into the dwell position.
        # Waypoints [ramp_s, dwell_start) get cosine-interpolated between their
        # original position at ramp_s and dwell_pos at dwell_start.
        ramp_s = max(0, dwell_start - ramp_waypoints)
        if ramp_s < dwell_start and ramp_s > 0:
            anchor = original[ramp_s]  # unmodified position to lerp from
            n = dwell_start - ramp_s
            for i in range(n):
                # alpha goes 0→1 across the ramp; cosine ease-in (decelerating)
                alpha = (i + 1) / n
                w = 0.5 * (1 - np.cos(np.pi * alpha))
                traj_np[ramp_s + i] = anchor * (1 - w) + dwell_pos * w

        # Exit ramp: ease from dwell position into the following motion segment.
        ramp_e = min(T, dwell_end + ramp_waypoints)
        if dwell_end < T and ramp_e > dwell_end:
            anchor = original[ramp_e - 1] if ramp_e <= T else original[-1]
            n = ramp_e - dwell_end
            for i in range(n):
                alpha = (i + 1) / n
                w = 0.5 * (1 - np.cos(np.pi * alpha))
                traj_np[dwell_end + i] = dwell_pos * (1 - w) + anchor * w

    n_smoothed = sum(
        min(ramp_waypoints, ds) + min(ramp_waypoints, T - de)
        for ds, de in dwells
    )
    if dwells:
        print(f"[motion-planner] Smoothed {n_smoothed} waypoints at {len(dwells)} dwell boundaries")

    return torch.tensor(traj_np, dtype=traj.dtype)


def _build_pick_place_trajectory(
    node, optimizer, ik_chain, capsule_model,
    q_start, q_pregrasp, q_grasp, q_preplace, q_place,
    pc_tensor, num_joints, arm, object_width,
    grasp_xyzrpy, place_xyzrpy,
    joint_limits=None, device=None,
    table_plane=None, table_polygon=None,
    pin_ik=None,
    q_pregrasp_travel=None,
    trac_ik_solver=None,
    trac_ik_dist=None,
    q_flip_waypoints=None,
    descent_waypoints=None,
    action="",
    chosen_heading=-135.0,
):
    """Build a full pick-and-place trajectory with dwell waypoints and gripper actions.

    Phases:
      1. start → pre-grasp          (long, collision-aware)
      2. pre-grasp → grasp           (short, approach)
      dwell: hold at grasp           (gripper closes)
      3. grasp → [flip waypoints →] pre-place  (long, collision-aware)
      4. pre-place → place           (short, descent)
      dwell: hold at place           (gripper opens)
      5. place → pre-place           (short, retreat)
      6. pre-place → home            (long, collision-aware)

    Returns ``(traj_np, metadata)`` on success, or ``None`` on failure.
    """
    t0 = time.perf_counter()
    q_home = torch.zeros_like(q_start)

    segments = []
    phase_count = [0]

    def _fk_pos(q):
        return _fk_pos_chain(ik_chain, q)

    # Staging XY to route around the center pole between arms
    if arm == "left":
        STAGING_XY = (-0.3, -0.3)
    else:
        STAGING_XY = (-0.3, 0.1)

    def _cartesian_segment(label, q_s, q_g, safe_z=False, staging=False, target_rot=None, speed=None, horiz_grip=False, position_only=False):
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
            z_clear = max(p_s[2], p_g[2]) + APPROACH_MARGIN
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
        n_keypoints = max(int(path_length / 0.0025), 5)  # 1 keypoint per 2.5mm
        # Ensure enough waypoints for all keypoints (1:1 mapping)
        T = max(T, n_keypoints)
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

            # Solve IK at each Cartesian keypoint using TracIK, seeded from
            # previous keypoint.  Slerp rotation from start to goal EE frames.

            # Get start/goal EE rotations from TracIK FK.
            # Only slerp when target_rot is explicitly given (e.g. rotation phases).
            # For transit segments, keep start rotation constant — slerping between
            # very different orientations creates unreachable intermediate targets.
            start_rot_kp = None
            rot_slerp = None
            if trac_ik_solver is not None:
                _, start_rot_kp = trac_ik_solver._solver.fk(q_s.cpu().numpy().astype(np.float64))
                if target_rot is not None:
                    rots = ScipyRotation.from_matrix(np.stack([start_rot_kp, target_rot]))
                    rot_slerp = Slerp([0.0, 1.0], rots)

            key_qs = [q_s.clone()]
            q_prev_np = q_s.cpu().numpy()
            # Track last successful rotation for fallback
            _, prev_rot_np = trac_ik_solver._solver.fk(q_prev_np.astype(np.float64)) if trac_ik_solver is not None else (None, None)
            failed = 0
            IK_SKIP = 3  # solve every Nth keypoint, joint-lerp between
            for i in range(1, n_keypoints):
                if i == n_keypoints - 1:
                    key_qs.append(q_g.clone())
                    break
                # Skip intermediate keypoints — joint-lerp will fill them
                if i % IK_SKIP != 0:
                    continue
                target_xyz = cart_targets[i]
                alpha_r = i / (n_keypoints - 1)
                if position_only:
                    # Use FK rotation from joint-space lerp — guaranteed achievable
                    q_lerp = q_s.cpu().numpy() * (1 - alpha_r) + q_g.cpu().numpy() * alpha_r
                    if trac_ik_solver is not None:
                        _, kp_rot = trac_ik_solver._solver.fk(q_lerp.astype(np.float64))
                    else:
                        kp_rot = prev_rot_np
                elif rot_slerp is not None:
                    kp_rot = rot_slerp(alpha_r).as_matrix()
                elif start_rot_kp is not None:
                    kp_rot = start_rot_kp
                else:
                    kp_rot = prev_rot_np

                solved = False
                # Use Distance mode — minimizes joint distance from seed (safer)
                kp_solver = trac_ik_dist if trac_ik_dist is not None else trac_ik_solver
                if kp_solver is not None and kp_rot is not None:
                    q_np, err = kp_solver.solve(
                        np.array(target_xyz, dtype=np.float64),
                        q_init=q_prev_np,
                        target_rot=kp_rot,
                    )
                    if q_np is not None and err <= 0.005:
                        key_qs.append(torch.tensor(q_np, dtype=torch.float32, device=device))
                        q_prev_np = q_np
                        prev_rot_np = kp_rot
                        solved = True

                    # Fallback: keep previous rotation
                    if not solved and prev_rot_np is not None and not np.array_equal(kp_rot, prev_rot_np):
                        q_np, err = kp_solver.solve(
                            np.array(target_xyz, dtype=np.float64),
                            q_init=q_prev_np,
                            target_rot=prev_rot_np,
                        )
                        if q_np is not None and err <= 0.005:
                            key_qs.append(torch.tensor(q_np, dtype=torch.float32, device=device))
                            q_prev_np = q_np
                            solved = True

                if not solved:
                    failed += 1
                    # Skip — interpolation between adjacent successful
                    # keypoints will cover the gap smoothly.

            if failed > 0:
                print(f"[motion-planner] {label}: {failed}/{n_keypoints-1} keypoint IK failures")

            # Distribute waypoints proportionally to joint-space distance
            n_intervals = len(key_qs) - 1
            interval_dists = []
            for i in range(n_intervals):
                d = float(torch.norm(key_qs[i + 1].cpu() - key_qs[i].cpu()))
                interval_dists.append(max(d, 1e-6))
            total_dist = sum(interval_dists)

            segments_inner = []
            total_wp = 0
            for i in range(n_intervals):
                qs = key_qs[i].cpu()
                qg = key_qs[i + 1].cpu()
                # More waypoints for longer intervals
                frac = interval_dists[i] / total_dist if total_dist > 0 else 1.0 / n_intervals
                n_wp = max(int(round(T * frac)), 2)
                alphas = torch.linspace(0, 1, n_wp).unsqueeze(1)
                seg = qs * (1 - alphas) + qg * alphas
                if i > 0:
                    seg = seg[1:]
                segments_inner.append(seg)
                total_wp += seg.shape[0]
            traj = torch.cat(segments_inner, dim=0)

        return traj

    # Helper: append segment, threading actual last waypoint for continuity
    def _append(seg, skip_first=True):
        if skip_first and len(segments) > 0:
            segments.append(seg[1:])
        else:
            segments.append(seg)
        return seg[-1]  # return actual last waypoint

    # Phase 1: start → pre-grasp
    if q_pregrasp_travel is not None:
        # Travel at 70° orientation (reliable), then interpolate to actual pre-grasp
        seg1 = _cartesian_segment("Phase 1: start -> pre-grasp (travel)", q_start, q_pregrasp_travel, safe_z=True, staging=True, position_only=True)
        q_prev = _append(seg1, skip_first=False)
        # Phase 1b: simultaneous rotation + descent from travel to pre-grasp.
        # Slerp orientation and interpolate XYZ in one pass (65 baby-steps).
        N_ROTATE = 100
        travel_pos_1b = _fk_pos(q_pregrasp_travel)
        pregrasp_pos_1b = _fk_pos(q_pregrasp)
        _, rot_start = trac_ik_solver._solver.fk(q_pregrasp_travel.cpu().numpy().astype(np.float64))
        _, rot_end = trac_ik_solver._solver.fk(q_pregrasp.cpu().numpy().astype(np.float64))
        rots_1b = Slerp([0.0, 1.0], ScipyRotation.from_matrix(np.stack([rot_start, rot_end])))
        seg1b_wps = []
        q_prev_1b = q_pregrasp_travel.cpu().numpy().astype(np.float64)
        for si in range(1, N_ROTATE + 1):
            alpha = si / N_ROTATE
            rot_i = rots_1b(alpha).as_matrix()
            pos_i = travel_pos_1b * (1 - alpha) + pregrasp_pos_1b * alpha
            q_i, err = trac_ik_solver.solve(
                np.array(pos_i, dtype=np.float64),
                q_init=q_prev_1b, target_rot=rot_i,
            )
            if q_i is not None and err < 0.005:
                max_jump = float(np.max(np.abs(q_i - q_prev_1b)))
                if max_jump > 0.3:
                    alpha_t = torch.tensor([alpha], dtype=torch.float32)
                    q_lerp = q_pregrasp_travel.cpu() * (1 - alpha_t) + q_pregrasp.cpu() * alpha_t
                    seg1b_wps.append(q_lerp.squeeze(0))
                else:
                    seg1b_wps.append(torch.tensor(q_i, dtype=torch.float32))
                    q_prev_1b = q_i
            else:
                alpha_t = torch.tensor([alpha], dtype=torch.float32)
                q_lerp = q_pregrasp_travel.cpu() * (1 - alpha_t) + q_pregrasp.cpu() * alpha_t
                seg1b_wps.append(q_lerp.squeeze(0))
        seg1b = torch.stack(seg1b_wps)
        drop_mm = (travel_pos_1b[2] - pregrasp_pos_1b[2]) * 1000
        print(f"  Phase 1b: rotate + descend to pre-grasp ({N_ROTATE} steps, {drop_mm:.0f}mm drop)")
        q_prev = _append(seg1b)
    else:
        seg1 = _cartesian_segment("Phase 1: start -> pre-grasp", q_start, q_pregrasp, safe_z=True, staging=True, position_only=True)
        q_prev = _append(seg1, skip_first=False)

    # Phase 2: pre-grasp → grasp (short descent, slow)
    seg2 = _cartesian_segment("Phase 2: pre-grasp -> grasp", q_prev, q_grasp, horiz_grip=True, speed=CARTESIAN_SPEED / 3.0)
    q_prev = _append(seg2)

    # Dwell at grasp (hold for gripper close)
    grasp_dwell_start = sum(s.shape[0] for s in segments)
    grasp_dwell = q_prev.unsqueeze(0).expand(DWELL_STEPS, -1).cpu()
    segments.append(grasp_dwell)

    if q_place is not None:
        # Phase 3: grasp → [flip/pour waypoints →] pre-place
        if q_flip_waypoints and action == "pour":
            # === POUR trajectory ===
            # Phase 3a: transfer from grasp to above pour target (q_close)
            q_pour_travel = q_flip_waypoints[0]  # q_close at pour target height
            seg_transfer = _cartesian_segment("Phase 3 pour: transfer to pour target",
                                              q_prev, q_pour_travel, safe_z=True, horiz_grip=True, speed=CARTESIAN_SPEED / 3.0)
            q_prev = _append(seg_transfer)

            # Phase 3b: tilt waypoints (baby-step roll 130°)
            for i, q_wp in enumerate(q_flip_waypoints[1:], start=1):
                N_STEPS = 2
                alphas = torch.linspace(0, 1, N_STEPS + 1)[1:].unsqueeze(1)
                seg = (q_prev.cpu() * (1 - alphas) + q_wp.cpu() * alphas)
                print(f"  Phase 3 pour tilt {i}/{len(q_flip_waypoints)-1}: "
                      f"joint interp ({N_STEPS} steps)")
                q_prev = _append(seg)

            # Pour dwell: hold at tilted position for ~2s (keep gripper closed!)
            POUR_DWELL_STEPS = 60  # ~2s at 30Hz
            pour_dwell = q_prev.unsqueeze(0).expand(POUR_DWELL_STEPS, -1).cpu()
            segments.append(pour_dwell)
            print(f"  Phase 3 pour dwell: {POUR_DWELL_STEPS} steps (~{POUR_DWELL_STEPS / PLAYBACK_HZ:.1f}s)")

            # Un-tilt: reverse through baby-step waypoints back to q_close
            for i, q_wp in enumerate(reversed(q_flip_waypoints[1:])):
                N_STEPS = 2
                alphas = torch.linspace(0, 1, N_STEPS + 1)[1:].unsqueeze(1)
                seg = (q_prev.cpu() * (1 - alphas) + q_wp.cpu() * alphas)
                q_prev = _append(seg)
            # Back to q_close (travel orientation above pour target)
            N_STEPS = 2
            alphas = torch.linspace(0, 1, N_STEPS + 1)[1:].unsqueeze(1)
            seg = (q_prev.cpu() * (1 - alphas) + q_pour_travel.cpu() * alphas)
            q_prev = _append(seg)
            print(f"  Phase 3 pour un-tilt: {len(q_flip_waypoints)-1} waypoints")

            # Return cup to original pick position: move back to pre-grasp, then descend to grasp
            seg_return = _cartesian_segment("Phase 4 pour: return to pre-grasp",
                                            q_prev, q_pregrasp, safe_z=True, horiz_grip=True)
            q_prev = _append(seg_return)

            seg_descend = _cartesian_segment("Phase 4 pour: descend to grasp (place cup)",
                                             q_prev, q_grasp, horiz_grip=True, speed=CARTESIAN_SPEED / 3.0)
            q_prev = _append(seg_descend)

            # Place dwell: open gripper to release cup at original position
            place_dwell_start = sum(s.shape[0] for s in segments)
            place_dwell = q_prev.unsqueeze(0).expand(DWELL_STEPS, -1).cpu()
            segments.append(place_dwell)

            # Lift back to pre-grasp
            seg_lift = _cartesian_segment("Phase 4 pour: lift from place", q_prev, q_pregrasp, horiz_grip=True)
            q_prev = _append(seg_lift)

        elif q_flip_waypoints:
            # === FLIP trajectory ===
            # Flip waypoints: joint interp (for the roll)
            for i, q_wp in enumerate(q_flip_waypoints):
                N_STEPS = 15 if i == 0 else 2  # first step (grasp→travel) needs more time
                alphas = torch.linspace(0, 1, N_STEPS + 1)[1:].unsqueeze(1)
                seg = (q_prev.cpu() * (1 - alphas) + q_wp.cpu() * alphas)
                print(f"  Phase 3 flip {i+1}/{len(q_flip_waypoints)}: "
                      f"joint interp ({N_STEPS} steps)")
                q_prev = _append(seg)

            # Descent: joint interp through baby-step IK waypoints
            if descent_waypoints is not None and len(descent_waypoints) > 0:
                for i, q_dw in enumerate(descent_waypoints):
                    N_D = 10  # small steps between close configs
                    alphas = torch.linspace(0, 1, N_D + 1)[1:].unsqueeze(1)
                    seg_d = (q_prev.cpu() * (1 - alphas) + q_dw.cpu() * alphas)
                    q_prev = _append(seg_d)
                print(f"  Phase 3 flip descent: {len(descent_waypoints)} steps")

            # Dwell at place (hold for gripper open)
            place_dwell_start = sum(s.shape[0] for s in segments)
            place_dwell = q_prev.unsqueeze(0).expand(DWELL_STEPS, -1).cpu()
            segments.append(place_dwell)

            # Lift back up: reverse descent waypoints (same amount we descended)
            q_rolled_wp = q_flip_waypoints[-1]
            if descent_waypoints is not None and len(descent_waypoints) > 0:
                for q_dw in reversed(descent_waypoints):
                    N_D = 10
                    alphas = torch.linspace(0, 1, N_D + 1)[1:].unsqueeze(1)
                    seg_d = (q_prev.cpu() * (1 - alphas) + q_dw.cpu() * alphas)
                    q_prev = _append(seg_d)
                # Back to rolled config (same height as where we started the roll)
                N_D = 15
                alphas = torch.linspace(0, 1, N_D + 1)[1:].unsqueeze(1)
                seg_d = (q_prev.cpu() * (1 - alphas) + q_rolled_wp.cpu() * alphas)
                q_prev = _append(seg_d)
                print(f"  Phase 4a (flip): lift back {len(descent_waypoints)} steps")

            # Roll back to travel: reverse through baby-step waypoints
            for i, q_wp in enumerate(reversed(q_flip_waypoints)):
                N_STEPS = 2
                alphas = torch.linspace(0, 1, N_STEPS + 1)[1:].unsqueeze(1)
                seg = (q_prev.cpu() * (1 - alphas) + q_wp.cpu() * alphas)
                q_prev = _append(seg)
            print(f"  Phase 4b (flip): roll back through {len(q_flip_waypoints)} waypoints")
        else:
            seg3 = _cartesian_segment("Phase 3: grasp -> pre-place", q_prev, q_preplace, safe_z=True, staging=True, horiz_grip=True)
            q_prev = _append(seg3)

            # Phase 4: pre-place → place (descent, slow)
            seg4 = _cartesian_segment("Phase 4: pre-place -> place", q_prev, q_place, horiz_grip=True, speed=CARTESIAN_SPEED / 3.0)
            q_prev = _append(seg4)

            # Dwell at place (hold for gripper open)
            place_dwell_start = sum(s.shape[0] for s in segments)
            place_dwell = q_prev.unsqueeze(0).expand(DWELL_STEPS, -1).cpu()
            segments.append(place_dwell)

            # Phase 4a: lift back to pre-place height
            seg4a = _cartesian_segment("Phase 4a: place -> pre-place", q_prev, q_preplace, horiz_grip=True)
            q_prev = _append(seg4a)
    else:
        # Grasp-only: lift back to pre-grasp
        seg3a = _cartesian_segment("Phase 3a: grasp -> pre-grasp", q_prev, q_pregrasp, safe_z=True, horiz_grip=True)
        q_prev = _append(seg3a)

    # Phase 5: place → above-home (staging transit)
    p_home = _fk_pos(q_home)
    p_prev = _fk_pos(q_prev)
    z_above = max(p_prev[2], p_home[2]) + 0.08

    # IK for above-home from q_prev seed (TracIK with current EE rotation)
    above_home_pos = np.array([p_home[0], p_home[1], z_above])
    q_prev_np = q_prev.cpu().numpy()
    ah_err = float("inf")
    q_ah_np = None
    if trac_ik_solver is not None:
        _, prev_rot = trac_ik_solver._solver.fk(q_prev_np.astype(np.float64))
        q_ah_np, ah_err = trac_ik_solver.solve(above_home_pos, q_init=q_prev_np, target_rot=prev_rot)
    # If we used a travel orientation, lift higher first then rotate back (skip for flip)
    if q_pregrasp_travel is not None and not q_flip_waypoints:
        # Lift to clearance height before rotating orientation
        p_cur = _fk_pos(q_prev)
        z_clear_rot = max(p_cur[2] + 0.06, z_above)
        lift_pos = np.array([p_cur[0], p_cur[1], z_clear_rot])
        if trac_ik_solver is not None:
            _, cur_rot_lift = trac_ik_solver._solver.fk(q_prev.cpu().numpy().astype(np.float64))
            q_lift, lift_err = trac_ik_solver.solve(lift_pos, q_init=q_prev.cpu().numpy().astype(np.float64), target_rot=cur_rot_lift)
            if q_lift is not None and lift_err < 0.005:
                q_lift_t = torch.tensor(q_lift, dtype=torch.float32, device=device)
                seg_lift = _cartesian_segment("Phase 4b-lift: lift before rotation", q_prev, q_lift_t, horiz_grip=True)
                q_prev = _append(seg_lift)
                print(f"  Phase 4b-lift: raised {(z_clear_rot - p_cur[2])*1000:.0f}mm before rotation")

        # Solve IK for 70° travel orientation at the lifted position
        p_cur = _fk_pos(q_prev)
        travel_angle = 70.0
        heading_rad = np.radians(float(chosen_heading))
        horiz = np.array([np.cos(heading_rad), np.sin(heading_rad), 0.0])
        angle_rad = np.radians(travel_angle)
        travel_approach = np.sin(angle_rad) * horiz + np.cos(angle_rad) * np.array([0.0, 0.0, -1.0])
        travel_approach = travel_approach / np.linalg.norm(travel_approach)
        travel_z_ee = travel_approach
        up = np.array([0.0, 0.0, 1.0])
        travel_y = np.cross(travel_z_ee, up)
        travel_y = travel_y / np.linalg.norm(travel_y)
        travel_x = np.cross(travel_y, travel_z_ee)
        travel_x = travel_x / np.linalg.norm(travel_x)
        travel_y = np.cross(travel_z_ee, travel_x)
        from scipy.spatial.transform import Rotation as R_scipy
        travel_rot = np.stack([travel_x, travel_y, travel_z_ee], axis=1)
        travel_rpy = R_scipy.from_matrix(travel_rot).as_euler("XYZ")
        p_cur_np = p_cur.cpu().numpy() if hasattr(p_cur, 'cpu') else np.asarray(p_cur)
        cur_travel_pose = np.concatenate([p_cur_np, travel_rpy]).astype(np.float32)
        q_cur_travel = solve_ik(
            ik_chain, cur_travel_pose, q_prev, joint_limits, device,
            num_seeds=NUM_SEEDS, horiz_grip=True, trac_ik_solver=trac_ik_solver,
        )
        if q_cur_travel is not None:
            N_ROTATE = 45
            alphas = torch.linspace(0, 1, N_ROTATE + 1)[1:].unsqueeze(1)
            seg4b = (q_prev.cpu() * (1 - alphas) + q_cur_travel.cpu() * alphas)
            print(f"  Phase 4b: rotate back to travel orientation ({N_ROTATE} steps)")
            q_prev = _append(seg4b)
        else:
            print(f"  Phase 4b: travel orientation IK failed at current pos, skipping rotation")
    # Go home (always Cartesian with safe_z to avoid table collisions)
    home_phase_start = sum(s.shape[0] for s in segments)
    seg5 = _cartesian_segment("Phase 5: place -> home", q_prev, q_home, staging=True, position_only=True)
    _append(seg5)

    full_traj = torch.cat(segments, dim=0)
    full_traj = _smooth_dwell_boundaries(full_traj, MAX_JOINT_STEP)
    # General discontinuity smoothing: apply a moving-average filter to
    # waypoints with large joint jumps, progressively relaxing the threshold.
    traj_np = full_traj.numpy().copy()
    total_fixed = 0
    for threshold in [0.12, 0.10, 0.08, 0.06, 0.04]:
        for _pass in range(5):
            diffs = np.abs(np.diff(traj_np, axis=0))
            n_fixed = 0
            for t in range(2, traj_np.shape[0] - 2):
                max_step = max(np.max(diffs[t - 1]), np.max(diffs[t]) if t < diffs.shape[0] else 0)
                if max_step > threshold:
                    # Weighted average: 25% each neighbor, 50% current (gentler)
                    traj_np[t] = 0.25 * traj_np[t - 1] + 0.5 * traj_np[t] + 0.25 * traj_np[t + 1]
                    n_fixed += 1
            total_fixed += n_fixed
            if n_fixed == 0:
                break
    if total_fixed > 0:
        print(f"[motion-planner] Smoothed {total_fixed} discontinuities (multi-pass)")

    # Gaussian smoothing: low-pass filter to remove jerk spikes at phase boundaries.
    # Uses a small kernel to smooth velocity transitions without distorting the path.
    # Skip dwell regions (consecutive identical waypoints) to preserve hold positions.
    from scipy.ndimage import uniform_filter1d
    SMOOTH_WINDOW = 7  # ~0.2s at 36Hz — enough to smooth phase transitions
    smoothed = uniform_filter1d(traj_np, size=SMOOTH_WINDOW, axis=0, mode='nearest')
    # Protect dwell regions: don't smooth where waypoints are static
    diffs_smooth = np.max(np.abs(np.diff(traj_np, axis=0)), axis=1)
    is_moving = np.ones(traj_np.shape[0], dtype=bool)
    for t in range(1, traj_np.shape[0]):
        if t - 1 < len(diffs_smooth) and diffs_smooth[t - 1] < 1e-7:
            is_moving[t] = False
    # Also protect first and last 3 waypoints
    is_moving[:3] = False
    is_moving[-3:] = False
    n_smoothed_gauss = int(np.sum(is_moving))
    traj_np[is_moving] = smoothed[is_moving]
    print(f"[motion-planner] Gaussian smoothed {n_smoothed_gauss}/{traj_np.shape[0]} moving waypoints (window={SMOOTH_WINDOW})")

    full_traj = torch.tensor(traj_np, dtype=full_traj.dtype)
    total_waypoints = full_traj.shape[0]
    t_plan = time.perf_counter() - t0

    print(f"[motion-planner] Trajectory: {total_waypoints} wp ({t_plan:.1f}s)")

    # FK Z analysis for home return phase
    if trac_ik_solver is not None and home_phase_start < total_waypoints:
        z_vals = []
        for i in range(home_phase_start, total_waypoints, max(1, (total_waypoints - home_phase_start) // 20)):
            fk_pos, _ = trac_ik_solver._solver.fk(full_traj[i].cpu().numpy().astype(np.float64))
            z_vals.append((i, fk_pos[0], fk_pos[1], fk_pos[2]))
        # Also last waypoint
        fk_pos, _ = trac_ik_solver._solver.fk(full_traj[-1].cpu().numpy().astype(np.float64))
        z_vals.append((total_waypoints - 1, fk_pos[0], fk_pos[1], fk_pos[2]))
        z_arr = [v[3] for v in z_vals]
        min_z_idx = int(np.argmin(z_arr))
        print(f"[motion-planner] === Home phase Z profile (wp {home_phase_start}-{total_waypoints}) ===")
        for i, x, y, z in z_vals:
            marker = " <-- MIN" if z == z_arr[min_z_idx] else ""
            print(f"  wp {i:5d}: X={x:+.3f} Y={y:+.3f} Z={z:.3f}{marker}")
        print(f"  Z range: {min(z_arr)*1000:.0f}mm - {max(z_arr)*1000:.0f}mm (dip={( max(z_arr)-min(z_arr))*1000:.0f}mm)")

    # Post-optimization collision validation
    if capsule_model is not None:
        _log_collision_check(
            validate_trajectory(
                ik_chain, capsule_model, full_traj, pc_tensor, SAFETY_MARGIN,
                table_plane=table_plane, table_polygon=table_polygon,
            ),
            total_waypoints,
        )

    # Build abort trajectory: from check waypoint → home (for grasp failure)
    # Uses the same _cartesian_segment with safe_z for table clearance.
    abort_traj_np = None
    grasp_check_waypoint_raw = grasp_dwell_start + DWELL_STEPS + 30 + 30 + 15
    if grasp_check_waypoint_raw < total_waypoints:
        q_abort_start = full_traj[grasp_check_waypoint_raw]
        abort_seg = _cartesian_segment("Abort: check -> home", q_abort_start, q_home, safe_z=True, position_only=True)
        if abort_seg is not None and abort_seg.shape[0] > 0:
            abort_traj_np = abort_seg.numpy().astype(np.float32)
            print(f"[motion-planner] Abort path: {abort_traj_np.shape[0]} waypoints (check wp {grasp_check_waypoint_raw} → home)")

    close_rad = compute_gripper_close_rad(object_width)
    approach_rad = compute_gripper_approach_rad(object_width)
    # Pour: open jaw fully for easier cup grab
    if action == "pour":
        approach_rad = GRIPPER_OPEN_RAD
    gripper_actions = [
        {"waypoint": 0, "rad": float(approach_rad)},  # open just wider than object
        {"waypoint": grasp_dwell_start, "rad": float(close_rad)},
    ]
    if q_place is not None:
        gripper_actions.append(
            {"waypoint": place_dwell_start, "rad": float(approach_rad)}
        )

    # After the close ramp, command a full squeeze (0.0 rad) so the motor
    # either stalls on the object or closes fully — giving a clear signal.
    # Close ramp ends at grasp_dwell_start + 30, squeeze ramp starts there
    # and runs another 30 steps, then allow ~15 steps for motor to settle.
    # Pour: skip full squeeze — just hold at close_rad to avoid crushing the cup.
    squeeze_waypoint = grasp_dwell_start + 30
    if action == "pour":
        # Gentle squeeze: halfway between close_rad and fully closed
        pour_squeeze_rad = close_rad * 0.5
        gripper_actions.append(
            {"waypoint": squeeze_waypoint, "rad": float(pour_squeeze_rad)}
        )
    else:
        gripper_actions.append(
            {"waypoint": squeeze_waypoint, "rad": float(GRIPPER_CLOSED_RAD)}
        )
    grasp_check_waypoint = squeeze_waypoint + 30 + 15  # after squeeze ramp + settle

    traj_np = full_traj.numpy().astype(np.float32)
    dt = 1.0 / PLAYBACK_HZ
    out_metadata = {
        "num_waypoints": total_waypoints,
        "num_joints": num_joints,
        "dt": dt,
        "encoding": "trajectory",
        "arm": arm,
        "grasp_pose": grasp_xyzrpy.tolist(),
        "place_pose": place_xyzrpy.tolist() if place_xyzrpy is not None else None,
        "gripper_actions": json.dumps(gripper_actions),
        "grasp_check_waypoint": grasp_check_waypoint,
        "grasp_dwell_start": grasp_dwell_start,
        "grasp_dwell_end": grasp_dwell_start + DWELL_STEPS + 30,  # after close ramp
        "home_phase_start": home_phase_start,
        "abort_trajectory": abort_traj_np,
        "expected_close_rad": float(close_rad),
        "approach_rad": float(approach_rad),
        "action": action,
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


def _auto_save_trajectory(doc, arm, traj_type):
    """Save trajectory to EXPORT_DIR with a unique timestamped filename."""
    if not EXPORT_DIR:
        return
    from datetime import datetime
    dir_path = Path(EXPORT_DIR)
    dir_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{arm}_{traj_type}.json"
    path = dir_path / filename
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"[motion-planner] Auto-saved trajectory → {path}")


def _run_live_analysis(traj_np, traj_meta):
    """Run trajectory analysis and print report to logs."""
    from dora_motion_planner.analyze_trajectory import summarize, format_report
    dt = float(traj_meta.get("dt", 0.1))
    results = summarize(traj_np, dt, metadata=traj_meta, max_step=MAX_JOINT_STEP)
    arm = traj_meta.get("arm", "left")
    report = format_report(f"{arm} trajectory", results)
    for line in report.split("\n"):
        print(f"[motion-planner] {line}")


def _build_home_trajectory(current_joints_np, arm_cfg, device, hz=PLAYBACK_HZ):
    """Build a safe home-return trajectory: lift up, then Cartesian to home.

    Uses IK to plan a lift-then-home path that avoids table collisions.
    Falls back to joint interpolation if IK fails.
    Returns ``(traj_np, metadata)`` suitable for ``_set_playback``.
    """
    num_joints = arm_cfg.num_joints
    q_start = current_joints_np[:num_joints].astype(np.float32)
    q_home = np.zeros(num_joints, dtype=np.float32)

    ik_chain = arm_cfg.ik_chain
    trac_ik = arm_cfg.trac_ik

    segments = []

    # 1. Lift straight up by 80mm from current position
    q_start_t = torch.tensor(q_start, dtype=torch.float32, device=device)
    fk = ik_chain.forward_kinematics(q_start_t.unsqueeze(0))
    cur_pos = fk.get_matrix()[0, :3, 3].cpu().numpy()
    lift_pos = cur_pos.copy()
    lift_pos[2] += 0.08  # 80mm up

    q_lift = None
    if trac_ik is not None:
        _, cur_rot = trac_ik._solver.fk(q_start.astype(np.float64))
        q_lift_np, err = trac_ik.solve(lift_pos, q_init=q_start.astype(np.float64), target_rot=cur_rot)
        if q_lift_np is not None and err < 0.005:
            q_lift = q_lift_np.astype(np.float32)

    if q_lift is not None:
        # Interpolate to lift position
        N_LIFT = 30
        alphas = np.linspace(0, 1, N_LIFT).reshape(-1, 1).astype(np.float32)
        seg_lift = q_start * (1 - alphas) + q_lift * alphas
        segments.append(seg_lift)
        q_prev = q_lift
    else:
        q_prev = q_start

    # 2. Interpolate from lifted position to home
    max_delta = float(np.max(np.abs(q_home - q_prev)))
    duration = max(1.0, max_delta / 2.0)
    N_HOME = max(int(duration * hz), 20)
    alphas = np.linspace(0, 1, N_HOME).reshape(-1, 1).astype(np.float32)
    seg_home = q_prev * (1 - alphas) + q_home * alphas
    segments.append(seg_home)

    traj_np = np.concatenate(segments, axis=0)
    meta = {
        "num_waypoints": traj_np.shape[0],
        "num_joints": num_joints,
        "dt": 1.0 / hz,
        "encoding": "home_return",
    }
    return traj_np, meta


def _set_playback(state, traj_np, traj_meta, node=None):
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

    # Grasp verification: waypoint to check + expected gripper rad
    state["expected_close_rad"] = traj_meta.get("expected_close_rad")
    state["approach_rad"] = traj_meta.get("approach_rad")
    state["grasp_check_waypoint"] = traj_meta.get("grasp_check_waypoint")
    state["grasp_checked"] = False
    if state["grasp_check_waypoint"] is not None:
        print(f"[motion-planner] Grasp check armed at waypoint {state['grasp_check_waypoint']} "
              f"(expected_close={state['expected_close_rad']})")

    # Build trajectory JSON (v3) with gripper frames for external playback
    extra = {}
    for k in ("grasp_pose", "pregrasp_waypoint", "place_pose", "gripper_actions", "encoding",
              "grasp_check_waypoint", "grasp_dwell_start", "grasp_dwell_end", "home_phase_start"):
        if k in traj_meta:
            extra[k] = traj_meta[k]
    if "grasp_input" in traj_meta:
        extra["grasp_input"] = traj_meta["grasp_input"]
    gripper_array = _gripper_actions_to_array(state["gripper_actions"], traj_np.shape[0])
    doc = build_trajectory_json(
        traj_np,
        arm=traj_meta.get("arm", "left"),
        dt=state["play_dt"],
        gripper=gripper_array,
        extra_metadata=extra,
        gripper_kp_scale=1.0,
    )
    # Build abort trajectory commands (for grasp failure → home)
    abort_traj = traj_meta.get("abort_trajectory")
    if abort_traj is not None:
        abort_doc = build_trajectory_json(
            abort_traj,
            arm=traj_meta.get("arm", "left"),
            dt=state["play_dt"],
            gripper=None,  # gripper open during abort (no gripper frames = open)
        )
        doc["abort_commands"] = abort_doc["commands"]

    state["trajectory_doc"] = doc

    if EXPORT_PATH:
        with open(EXPORT_PATH, "w") as f:
            json.dump(doc, f, indent=2)
        print(f"[motion-planner] Exported trajectory → {EXPORT_PATH}")

    # Auto-save to EXPORT_DIR with unique filename
    traj_type = traj_meta.get("encoding", "trajectory")
    _auto_save_trajectory(doc, traj_meta.get("arm", "left"), traj_type)

    # Live analysis
    _run_live_analysis(traj_np, traj_meta)

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
    latest_image = None
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
        "play_dt": 1.0 / PLAYBACK_HZ,
        "gripper_actions": [],  # list of {"waypoint": N, "rad": float}
        "gripper_fired": set(),
    }

    # Latest joint states per arm (for grasp verification)
    latest_joint_states = {"left": None, "right": None}


    node = Node()
    print(
        f"[motion-planner] Ready ({', '.join(arms.keys())} arm(s)), "
        f"playback={PLAYBACK_MODE}, "
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
                    print(f"[motion-planner] Playback streaming done, waiting for arm to finish...")

            # --- Arm trajectory status (external playback done) ---
            elif event_id in ("left_trajectory_status", "right_trajectory_status"):
                raw = event["value"][0].as_py()
                # Parse status: "done", "grasp_failed", or JSON
                status_str = None
                status_obj = {}
                if raw in ("done", "playing"):
                    status_str = raw
                else:
                    try:
                        status_obj = json.loads(raw)
                        status_str = status_obj.get("status")
                    except (json.JSONDecodeError, TypeError, IndexError):
                        pass

                if status_str == "grasp_failed" and playback["playing"]:
                    # Playback node already went home — just signal retry
                    playback["playing"] = False
                    tau = status_obj.get("gripper_max_torque", "?")
                    print(f"[motion-planner] Grasp failed (close_max={tau}Nm), signaling retry")
                    node.send_output("trajectory_status", pa.array([json.dumps({
                        "status": "failed",
                        "reason": "Grasp failed (gripper closed without object)",
                        "retry": True,
                    })]))

                elif status_str == "done" and playback["playing"]:
                    playback["playing"] = False
                    tau = status_obj.get("gripper_max_torque")
                    if tau is not None:
                        print(f"[motion-planner] Grasp OK (close_max={tau:.2f}Nm)")
                    print(f"[motion-planner] Arm reports done, signaling trajectory complete")
                    node.send_output("trajectory_status", pa.array([json.dumps({"status": "done"})]))

            # --- Image (RGB for debug visualization) ---
            elif event_id == "image":
                latest_image = event["value"].to_numpy().astype(np.uint8)

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
                    latest_joint_states["left"] = js

            elif event_id == "right_joint_state":
                arm = arms.get("right")
                if arm:
                    js = event["value"].to_numpy().astype(np.float32)
                    arm.current_joints = torch.tensor(
                        js[: arm.num_joints], dtype=torch.float32, device=device
                    )
                    latest_joint_states["right"] = js

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
                        trac_ik_solver=arm.trac_ik,
                        trac_ik_dist=arm.trac_ik_dist,
                        latest_image=latest_image,
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
                            trac_ik_solver=arm.trac_ik,
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
                    _set_playback(playback, result[0], result[1], node=node)
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
                # KPI timing
                command_ts = data.get("command_ts", 0)
                t_received = time.time()
                if command_ts:
                    print(f"[KPI] command->planner: {t_received - command_ts:.1f}s")
                print(
                    f"[motion-planner] grasp_result: jaw1=({u1:.0f},{v1:.0f}) "
                    f"jaw2=({u2:.0f},{v2:.0f})"
                )

                # Parse optional place target
                place_uv = None
                place_mask_bbox = None
                place_data = data.get("place_px")
                if place_data and len(place_data) >= 2:
                    place_u = float(place_data[0])
                    place_v = float(place_data[1])
                    place_uv = (place_u, place_v)
                    print(
                        f"[motion-planner] grasp_result: place=({place_u:.0f},{place_v:.0f})"
                    )
                place_bbox_data = data.get("place_mask_bbox")
                if place_bbox_data and len(place_bbox_data) >= 4:
                    place_mask_bbox = [int(x) for x in place_bbox_data]
                    print(f"[motion-planner] grasp_result: place_mask_bbox={place_mask_bbox}")

                grasp_action = data.get("action", "")
                if grasp_action:
                    print(f"[motion-planner] grasp_result: action={grasp_action}")

                # Load SAM3 masks from disk for free-space place targeting
                loaded_place_mask = None
                loaded_pick_mask = None
                if place_uv is not None and EXPORT_DIR:
                    import glob as _g_mask
                    search_dirs = [Path(EXPORT_DIR).parent, Path(EXPORT_DIR).parent.resolve()]
                    # Load place mask
                    place_mask_files = []
                    for sd in search_dirs:
                        place_mask_files.extend(sorted(_g_mask.glob(str(sd / "critic_place_mask_*.png"))))
                    if place_mask_files:
                        import cv2 as _cv2_mask
                        pm = _cv2_mask.imread(str(place_mask_files[-1]), _cv2_mask.IMREAD_GRAYSCALE)
                        if pm is not None:
                            w_img, h_img = latest_image_size
                            if pm.shape[:2] != (h_img, w_img):
                                pm = _cv2_mask.resize(pm, (w_img, h_img), interpolation=_cv2_mask.INTER_NEAREST)
                            loaded_place_mask = pm
                            print(f"[motion-planner] Loaded place mask: {place_mask_files[-1]}")
                    # Load pick mask
                    pick_mask_files = []
                    for sd in search_dirs:
                        pick_mask_files.extend(sorted(_g_mask.glob(str(sd / "critic_mask_*.png"))))
                    if pick_mask_files:
                        import cv2 as _cv2_mask
                        pkm = _cv2_mask.imread(str(pick_mask_files[-1]), _cv2_mask.IMREAD_GRAYSCALE)
                        if pkm is not None:
                            w_img, h_img = latest_image_size
                            if pkm.shape[:2] != (h_img, w_img):
                                pkm = _cv2_mask.resize(pkm, (w_img, h_img), interpolation=_cv2_mask.INTER_NEAREST)
                            loaded_pick_mask = pkm
                            print(f"[motion-planner] Loaded pick mask: {pick_mask_files[-1]}")

                # Save debug image with pick/place targets (before IK, always saved)
                if EXPORT_DIR and latest_image is not None:
                    try:
                        import cv2
                        w, h = latest_image_size
                        img_flat = latest_image
                        if img_flat.size == w * h * 3:
                            debug_img = img_flat.reshape(h, w, 3).copy()
                            debug_img = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)
                        else:
                            debug_img = np.zeros((h, w, 3), dtype=np.uint8)
                        iu1, iv1 = int(round(u1)), int(round(v1))
                        iu2, iv2 = int(round(u2)), int(round(v2))
                        cv2.circle(debug_img, (iu1, iv1), 8, (0, 255, 0), 2)
                        cv2.circle(debug_img, (iu2, iv2), 8, (0, 255, 0), 2)
                        cv2.line(debug_img, (iu1, iv1), (iu2, iv2), (0, 255, 0), 2)
                        mid_u, mid_v = (iu1 + iu2) // 2, (iv1 + iv2) // 2
                        cv2.putText(debug_img, "PICK", (mid_u - 20, mid_v - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        # Draw pick mask bbox — green
                        pick_bbox = data.get("mask_bbox")
                        if pick_bbox and len(pick_bbox) >= 4:
                            x0, y0, x1, y1 = [int(v) for v in pick_bbox]
                            cv2.rectangle(debug_img, (x0, y0), (x1, y1), (0, 255, 0), 1)
                        # Overlay SAM3 pick mask as semi-transparent green
                        # Selector saves masks in its working dir (parent of EXPORT_DIR)
                        import glob as _g
                        # Search relative to EXPORT_DIR parent and also absolute
                        search_dirs = [Path(EXPORT_DIR).parent, Path(EXPORT_DIR).parent.resolve()]
                        mask_files = []
                        for sd in search_dirs:
                            mask_files.extend(sorted(_g.glob(str(sd / "critic_mask_*.png"))))
                        pick_mask_path = str(mask_files[-1]) if mask_files else ""
                        if not mask_files:
                            print(f"[debug] No mask files found in {search_dirs}")
                        if pick_mask_path and os.path.exists(pick_mask_path):
                            print(f"[debug] Loading mask: {pick_mask_path}")
                            mask_img = cv2.imread(pick_mask_path, cv2.IMREAD_GRAYSCALE)
                            if mask_img is not None:
                                if mask_img.shape[:2] != debug_img.shape[:2]:
                                    mask_img = cv2.resize(mask_img, (debug_img.shape[1], debug_img.shape[0]),
                                                          interpolation=cv2.INTER_NEAREST)
                                green = np.zeros_like(debug_img)
                                green[:, :] = (0, 255, 0)
                                debug_img[mask_img > 0] = cv2.addWeighted(
                                    debug_img, 0.75, green, 0.25, 0)[mask_img > 0]
                                cnts, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                cv2.drawContours(debug_img, cnts, -1, (0, 255, 0), 2)
                        # Draw original place point
                        if place_uv is not None:
                            pu, pv = int(round(place_uv[0])), int(round(place_uv[1]))
                            cv2.circle(debug_img, (pu, pv), 12, (255, 100, 0), 2)
                            cv2.putText(debug_img, "PLACE(orig)", (pu - 45, pv - 18),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)
                        # Draw place mask bbox
                        if place_mask_bbox is not None:
                            x0, y0, x1, y1 = [int(v) for v in place_mask_bbox]
                            cv2.rectangle(debug_img, (x0, y0), (x1, y1), (255, 255, 0), 1)
                        # Free-space visualization within container
                        if loaded_place_mask is not None and place_uv is not None and latest_depth is not None:
                            from .grasp_utils import find_free_place_target
                            w_d, h_d = latest_image_size
                            fx_d, fy_d, cx_d, cy_d = latest_intrinsics
                            fp_dbg = find_free_place_target(
                                loaded_place_mask, latest_depth, fx_d, fy_d, cx_d, cy_d,
                                cam_t, cam_rot, pick_mask=loaded_pick_mask,
                                width=w_d, height=h_d,
                            )
                            if fp_dbg is not None:
                                pm_2d = loaded_place_mask
                                place_bin = (pm_2d > 0).astype(np.uint8)
                                depth_2d_dbg = latest_depth.reshape(h_d, w_d).astype(np.float32)
                                valid_dbg = (depth_2d_dbg > 100) & (depth_2d_dbg < 5000) & (place_bin > 0)
                                depths_dbg = depth_2d_dbg[valid_dbg]
                                if len(depths_dbg) > 0:
                                    floor_d = np.percentile(depths_dbg, 85)
                                    free_px = (np.abs(depth_2d_dbg - floor_d) <= 15) & valid_dbg
                                    occupied_px = valid_dbg & ~free_px
                                    # Green overlay for free space
                                    green_ov = np.zeros_like(debug_img)
                                    green_ov[:, :] = (0, 200, 0)
                                    debug_img[free_px] = cv2.addWeighted(
                                        debug_img, 0.6, green_ov, 0.4, 0)[free_px]
                                    # Red overlay for occupied space
                                    red_ov = np.zeros_like(debug_img)
                                    red_ov[:, :] = (0, 0, 200)
                                    debug_img[occupied_px] = cv2.addWeighted(
                                        debug_img, 0.6, red_ov, 0.4, 0)[occupied_px]
                                # Draw container contour
                                cnts_c, _ = cv2.findContours(place_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                cv2.drawContours(debug_img, cnts_c, -1, (255, 150, 0), 2)
                                fp_u, fp_v, _, fits = fp_dbg
                                # Stamp pick mask outline at placement position
                                if loaded_pick_mask is not None and fits:
                                    pk_bin = (loaded_pick_mask > 0).astype(np.uint8)
                                    pk_ys, pk_xs = np.where(pk_bin > 0)
                                    if len(pk_ys) > 0:
                                        pk_cy, pk_cx = int(pk_ys.mean()), int(pk_xs.mean())
                                        dy, dx = fp_v - pk_cy, fp_u - pk_cx
                                        M = np.float32([[1, 0, dx], [0, 1, dy]])
                                        shifted_m = cv2.warpAffine(pk_bin, M, (w_d, h_d))
                                        cnts_pk, _ = cv2.findContours(shifted_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                        cv2.drawContours(debug_img, cnts_pk, -1, (0, 255, 255), 3)
                                # Crosshair at corrected place target
                                label = "PLACE(free)" if fits else "PLACE(no-fit)"
                                sz = 18
                                cv2.line(debug_img, (fp_u - sz, fp_v), (fp_u + sz, fp_v), (0, 200, 255), 3)
                                cv2.line(debug_img, (fp_u, fp_v - sz), (fp_u, fp_v + sz), (0, 200, 255), 3)
                                cv2.circle(debug_img, (fp_u, fp_v), 14, (0, 200, 255), 2)
                                cv2.putText(debug_img, label, (fp_u - 50, fp_v - 22),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                        elif loaded_place_mask is not None:
                            # Fallback: just draw place mask contour
                            place_bin_f = (loaded_place_mask > 0).astype(np.uint8)
                            cnts_f, _ = cv2.findContours(place_bin_f, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(debug_img, cnts_f, -1, (255, 150, 0), 2)
                        # Arm label
                        cv2.putText(debug_img, f"arm={metadata.get('arm', '?')}", (10, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        import datetime as _dt
                        ts = _dt.datetime.now().strftime("%H%M%S")
                        debug_path = str(Path(EXPORT_DIR) / f"pick_place_targets_{ts}.png")
                        cv2.imwrite(debug_path, debug_img)
                        # Also save as latest for easy access
                        latest_path = str(Path(EXPORT_DIR) / "pick_place_targets.png")
                        cv2.imwrite(latest_path, debug_img)
                        print(f"[motion-planner] Debug image: {debug_path}")
                        # Send debug image to MoQ publisher (BGR→RGB)
                        debug_rgb = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
                        node.send_output("debug_image", pa.array(debug_rgb.ravel()))
                    except Exception as e:
                        print(f"[motion-planner] Debug image failed: {e}")

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
                preferred_arm = select_arm(metadata, target_y)
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
                        place_mask_bbox=place_mask_bbox,
                        place_mask=loaded_place_mask,
                        pick_mask=loaded_pick_mask,
                        pin_ik=arm.pin_ik,
                        trac_ik_solver=arm.trac_ik,
                        trac_ik_dist=arm.trac_ik_dist,
                        latest_image=latest_image,
                        action=grasp_action,
                    )
                    if result is not None:
                        break
                    print(f"[motion-planner] {arm_name} arm failed: {fail_reason}")

                if result is not None:
                    grasp_input = {"p1": [u1, v1], "p2": [u2, v2]}
                    if place_uv:
                        grasp_input["place"] = list(place_uv)
                    result[1]["grasp_input"] = grasp_input
                    _set_playback(playback, result[0], result[1], node=node)
                    traj_meta = result[1]
                    t_planned = time.time()
                    duration = round(result[0].shape[0] * float(traj_meta.get("dt", 0.1)), 1)
                    if command_ts:
                        print(f"[KPI] command->planned: {t_planned - command_ts:.1f}s  "
                              f"(planner: {t_planned - t_received:.1f}s)  "
                              f"trajectory: {result[0].shape[0]}wp, {duration}s")
                    node.send_output("trajectory_status", pa.array([json.dumps({
                        "status": "ready",
                        "waypoints": result[0].shape[0],
                        "duration": duration,
                        "arm": traj_meta.get("arm", "left"),
                    })]))
                else:
                    node.send_output("trajectory_status", pa.array([json.dumps({
                        "status": "failed",
                        "reason": fail_reason or "Both arms failed",
                        "retry": True,
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
