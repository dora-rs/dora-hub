"""Dora node: vertical lift then forward reach demo for OpenArm.

Pre-computes a smooth joint trajectory using pytorch_kinematics:
  1. Lift up along z-axis (gripper pointing down)
  2. Tilt gripper to point forward (horizontal)
  3. Reach forward
  4. Return to start (reverse)

Uses URDF-based FK/IK (PseudoInverseIK) with per-step Cartesian IK
along minimum-jerk profiles for straight-line motion.

Env vars:
    STEPS_PER_PHASE: frames per phase at 10Hz (default 60 = 6s per phase)
    LIFT_HEIGHT: meters to lift (default 0.20)
    FORWARD_DIST: meters to reach forward (default 0.15)
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytorch_kinematics as pk
import torch
from dora import Node
from pytorch_kinematics.transforms.rotation_conversions import matrix_to_quaternion

URDF_PATH = str(Path(__file__).parent / "openarm_v10.urdf")
END_EFFECTOR_LINK = "openarm_left_link8"


def build_chain():
    with open(URDF_PATH, "r") as f:
        urdf_str = f.read()
    root = ET.fromstring(urdf_str)
    for link in root.findall(".//link"):
        for tag in ("visual", "collision", "inertial"):
            for elem in link.findall(tag):
                link.remove(elem)
    clean_urdf = ET.tostring(root, encoding="unicode")
    return pk.build_serial_chain_from_urdf(clean_urdf, END_EFFECTOR_LINK)


def fk_matrix(chain, q):
    """FK: numpy (N,) -> 4x4 numpy matrix."""
    q_t = torch.tensor(q, dtype=torch.float32).unsqueeze(0)
    tf = chain.forward_kinematics(q_t, end_only=True)
    return tf.get_matrix()[0].detach().numpy()


def make_target_tf(pos, rot_matrix):
    """Create Transform3d from position (3,) and rotation matrix (3,3)."""
    pos_t = torch.tensor(pos, dtype=torch.float32)
    rot_t = torch.tensor(rot_matrix, dtype=torch.float32).unsqueeze(0)
    quat = matrix_to_quaternion(rot_t).squeeze(0)
    return pk.Transform3d(pos=pos_t, rot=quat)


def solve_ik(chain, target_tf, q_init, limits_tensor, lower, upper,
             num_retries=20, pos_tol=0.003, rot_tol=0.03):
    """Solve IK with multiple retries around q_init."""
    num_joints = len(chain.get_joint_parameter_names(exclude_fixed=True))

    seeds = [q_init.copy()]
    for _ in range(num_retries - 1):
        noise = np.random.randn(num_joints) * 0.5
        seeds.append(np.clip(q_init + noise, lower, upper))

    retry_configs = torch.tensor(
        np.array(seeds), dtype=torch.float32
    ).requires_grad_(True)

    ik_solver = pk.PseudoInverseIK(
        chain,
        max_iterations=100_000,
        retry_configs=retry_configs,
        joint_limits=limits_tensor,
        early_stopping_any_converged=True,
        debug=False,
        lr=0.05,
        pos_tolerance=pos_tol,
        rot_tolerance=rot_tol,
    )
    result = ik_solver.solve(target_tf)

    sols = result.solutions.detach()[0]
    errs = result.err_pos.detach()[0]
    best_idx = torch.argmin(errs).item()
    q_sol = np.clip(sols[best_idx].numpy(), lower, upper)

    T_check = fk_matrix(chain, q_sol)
    target_mat = target_tf.get_matrix()[0].detach().numpy()
    pos_err = np.linalg.norm(T_check[:3, 3] - target_mat[:3, 3])
    rot_err = result.err_rot.detach()[0, best_idx].item()
    print(f"  IK: pos_err={pos_err:.6f}m, rot_err={rot_err:.6f}rad")
    return q_sol


def solve_ik_step(chain, target_tf, q_prev, limits_tensor, lower, upper,
                  max_joint_step=0.05):
    """Fast single-seed IK for trajectory steps.

    Clamps per-joint change to max_joint_step to prevent solution jumps.
    """
    q_seed = torch.tensor(q_prev, dtype=torch.float32).unsqueeze(0).requires_grad_(True)
    ik_solver = pk.PseudoInverseIK(
        chain,
        max_iterations=100_000,
        retry_configs=q_seed,
        joint_limits=limits_tensor,
        early_stopping_any_converged=True,
        debug=False,
        lr=0.05,
        pos_tolerance=0.003,
        rot_tolerance=0.03,
    )
    result = ik_solver.solve(target_tf)
    q_sol = result.solutions.detach()[0, 0].numpy()
    q_sol = np.clip(q_sol, lower, upper)
    # Limit per-joint change to prevent solution jumps
    delta = q_sol - q_prev
    delta = np.clip(delta, -max_joint_step, max_joint_step)
    return np.clip(q_prev + delta, lower, upper)


# ── Trajectory helpers ─────────────────────────────────────────────────


def min_jerk(t):
    """Minimum-jerk profile: zero vel/acc/jerk at endpoints."""
    return 10 * t**3 - 15 * t**4 + 6 * t**5



def compute_phase_cartesian(chain, p_start, p_end, R, q_start, steps, limits_tensor, lower, upper):
    """Per-step Cartesian IK along a straight line (constant orientation).

    Solves forward from q_start, seeding each step from previous solution.
    """
    traj = [q_start.copy()]
    q_prev = q_start.copy()

    for i in range(1, steps + 1):
        t = min_jerk(i / steps)
        target_pos = (1 - t) * p_start + t * p_end
        target_tf = make_target_tf(target_pos, R)
        q_sol = solve_ik_step(chain, target_tf, q_prev, limits_tensor, lower, upper)
        q_prev = q_sol
        traj.append(q_sol)

        if i % 10 == 0:
            T = fk_matrix(chain, q_sol)
            pos_err = np.linalg.norm(T[:3, 3] - target_pos)
            print(f"  step {i}/{steps}: pos_err={pos_err:.6f}m")

    return traj


def interpolate_joints(q_start, q_end, steps):
    """Minimum-jerk interpolation in joint space (guaranteed smooth)."""
    traj = []
    for i in range(steps + 1):
        s = min_jerk(i / steps)
        traj.append((1 - s) * q_start + s * q_end)
    return traj


# ── Main trajectory computation ───────────────────────────────────────


def compute_trajectory(chain, steps_per_phase, lift_height, forward_dist):
    num_joints = len(chain.get_joint_parameter_names(exclude_fixed=True))
    limits = chain.get_joint_limits()
    limits_tensor = torch.tensor(limits)
    lower = np.array(limits[0])
    upper = np.array(limits[1])
    # Small margin to avoid floating-point boundary rejections
    margin = 0.005
    lower_m = lower + margin
    upper_m = upper - margin
    q_home = np.clip(np.zeros(num_joints), lower_m, upper_m)

    T0 = fk_matrix(chain, q_home)
    p0 = T0[:3, 3].copy()
    R0 = T0[:3, :3].copy()

    print(f"Home position: {p0}")
    print(f"Home EE z-axis: {R0[:, 2]}")
    print(f"Joint limits lower: {lower}")
    print(f"Joint limits upper: {upper}")

    # Forward direction: perpendicular to arm extension, in the body's forward direction
    # The arm extends sideways from the body; rotate 90° CW in XY to get body forward
    arm_dir = p0.copy()
    arm_dir[2] = 0
    if np.linalg.norm(arm_dir) < 1e-6:
        arm_dir = np.array([1.0, 0.0, 0.0])
    arm_dir = arm_dir / np.linalg.norm(arm_dir)
    # 90° clockwise rotation in XY: [x,y] -> [y, -x]
    fwd = np.array([arm_dir[1], -arm_dir[0], 0.0])
    print(f"Arm extension dir: {arm_dir}")
    print(f"Forward direction: {fwd}")

    # --- Build trajectory ---
    # Phase 1: Lift up (Cartesian IK, constant orientation)
    p_up = p0.copy()
    p_up[2] += lift_height

    print(f"\n--- Phase 1: Lift to {p_up} ({steps_per_phase} steps, Cartesian IK) ---")
    traj_up = compute_phase_cartesian(chain, p0, p_up, R0, q_home, steps_per_phase, limits_tensor, lower_m, upper_m)

    # Phase 2: Reach forward (Cartesian IK, constant orientation)
    p_fwd = p_up + forward_dist * fwd

    print(f"\n--- Phase 2: Forward to {p_fwd} ({steps_per_phase} steps, Cartesian IK) ---")
    traj_fwd = compute_phase_cartesian(chain, p_up, p_fwd, R0, traj_up[-1], steps_per_phase, limits_tensor, lower_m, upper_m)

    # Combine and reverse
    traj_out = traj_up + traj_fwd[1:]
    traj_return = list(reversed(traj_out[:-1]))
    full_traj = np.array(traj_out + traj_return, dtype=np.float32)

    # Final safety clamp on entire trajectory (with margin)
    full_traj = np.clip(full_traj, lower_m, upper_m)

    print(f"\nTrajectory: {len(full_traj)} steps")
    print(f"  Out: {len(traj_out)} | Return: {len(traj_return)}")

    # Verify smoothness
    dt = 0.1
    dq = np.diff(full_traj, axis=0) / dt
    ddq = np.diff(dq, axis=0) / dt
    max_vel = np.max(np.abs(dq), axis=0)
    max_acc = np.max(np.abs(ddq), axis=0)
    print(f"Max vel (rad/s):  [{', '.join(f'{v:.3f}' for v in max_vel)}]")
    print(f"Max acc (rad/s2): [{', '.join(f'{a:.3f}' for a in max_acc)}]")

    for label, q in [
        ("Home", q_home),
        ("Top", traj_up[-1]),
        ("Forward", traj_fwd[-1]),
    ]:
        T = fk_matrix(chain, q)
        print(f"  {label}: pos={np.round(T[:3,3], 4)}, z={np.round(T[:3,2], 3)}")

    return full_traj


def main():
    steps_per_phase = int(os.getenv("STEPS_PER_PHASE", "60"))
    lift_height = float(os.getenv("LIFT_HEIGHT", "0.08"))
    forward_dist = float(os.getenv("FORWARD_DIST", "0.08"))

    chain = build_chain()
    print(f"Chain: {chain.get_joint_parameter_names(exclude_fixed=True)}")

    trajectory = compute_trajectory(chain, steps_per_phase, lift_height, forward_dist)
    num_steps = len(trajectory)

    node = Node()
    step = 0
    print(f"Playing trajectory ({num_steps} steps, looping)")

    for event in node:
        if event["type"] == "INPUT":
            if event["id"] == "tick":
                metadata = event["metadata"]
                metadata["encoding"] = "jointstate"
                node.send_output(
                    "joint_command",
                    pa.array(trajectory[step], type=pa.float32()),
                    metadata=metadata,
                )
                step = (step + 1) % num_steps
        elif event["type"] == "STOP":
            break


if __name__ == "__main__":
    main()
