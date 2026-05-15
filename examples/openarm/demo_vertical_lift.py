"""Dora node: vertical lift demo for OpenArm using numerical IK.

Pre-computes a smooth joint trajectory by solving IK at Cartesian waypoints
along a vertical line, then plays it back on tick.

Two IK backends:
  - Sequential (scipy): solves each waypoint independently with L-BFGS-B
  - Batch (pytorch-kinematics): optimizes all waypoints simultaneously with
    smoothness penalties via torch.optim.LBFGS and autograd

Env vars:
    STEPS_UP: frames per half-cycle at 10Hz (default 60 = 6s)
    LIFT_HEIGHT: meters to lift (default 0.12)
    USE_BATCH_IK: "true" (default) to use batch optimizer, "false" for scipy
"""

import os
from pathlib import Path

import numpy as np
import pyarrow as pa
from dora import Node
from scipy.optimize import minimize


# ── Forward Kinematics (from URDF) ──────────────────────────────────────

def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]])

def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]])

def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]])

def _trans(x, y, z):
    T = np.eye(4); T[0,3], T[1,3], T[2,3] = x, y, z; return T

def _rpy(r, p, y):
    return _rot_z(y) @ _rot_y(p) @ _rot_x(r)


def fk(q):
    """Forward kinematics for left arm. Returns 4x4 transform."""
    T = _trans(0, 0.031, 0.698) @ _rpy(-np.pi/2, 0, 0)
    T = T @ _trans(0, 0, 0.0625) @ _rot_z(q[0])
    T = T @ _trans(-0.0301, 0, 0.06) @ _rpy(-np.pi/2, 0, 0) @ _rot_x(-q[1])
    T = T @ _trans(0.0301, 0, 0.06625) @ _rot_z(q[2])
    T = T @ _trans(0, 0.0315, 0.15375) @ _rot_y(q[3])
    T = T @ _trans(0, -0.0315, 0.0955) @ _rot_z(q[4])
    T = T @ _trans(0.0375, 0, 0.1205) @ _rot_x(q[5])
    T = T @ _trans(-0.0375, 0, 0) @ _rot_y(-q[6])
    T = T @ _trans(1e-6, 0.0205, 0)
    return T


# ── Numerical IK ────────────────────────────────────────────────────────

# Joint limits from URDF (L_J1..L_J7)
JOINT_BOUNDS = [
    (-3.49, 1.40),   # J1
    (-3.32, 0.17),   # J2
    (-1.57, 1.57),   # J3
    (0.0, 2.44),     # J4
    (-1.57, 1.57),   # J5
    (-0.79, 0.79),   # J6
    (-1.57, 1.57),   # J7
]


def solve_ik(target_pos, target_z_axis, q_init, w_pos=100.0, w_ori=50.0, w_reg=0.01):
    """Solve IK: find joint angles that place end effector at target_pos
    with EE z-axis aligned to target_z_axis.

    w_reg penalizes deviation from q_init for smoothness.
    """
    def cost(q):
        T = fk(q)
        pos_err = T[:3, 3] - target_pos
        ori_err = T[:3, 2] - target_z_axis
        reg_err = q - q_init
        return (w_pos * np.dot(pos_err, pos_err) +
                w_ori * np.dot(ori_err, ori_err) +
                w_reg * np.dot(reg_err, reg_err))

    result = minimize(cost, q_init, method='L-BFGS-B', bounds=JOINT_BOUNDS,
                      options={'ftol': 1e-14, 'maxiter': 500})
    return result.x


# ── Trajectory generation ───────────────────────────────────────────────

def min_jerk(t):
    """Minimum-jerk profile: zero vel/acc/jerk at endpoints."""
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def compute_trajectory(steps_half, lift_height):
    """Pre-compute joint trajectory for vertical oscillation via IK.

    Solves IK at every step (no spline interpolation) using a minimum-jerk
    Cartesian profile for guaranteed smooth velocity and acceleration.
    """
    q_zero = np.zeros(7)
    T0 = fk(q_zero)
    p0 = T0[:3, 3].copy()
    z_down = np.array([0.0, 0.0, -1.0])  # gripper pointing down

    p_top = p0.copy()
    p_top[2] += lift_height

    # First solve the top position with a good seed
    q_seed_top = np.array([0.75, 0.0, 0.0, 1.5, 0.0, 0.0, 0.75])
    q_top = solve_ik(p_top, z_down, q_seed_top)
    T_top = fk(q_top)
    print(f"Top IK: pos_err={np.linalg.norm(T_top[:3,3]-p_top):.6f}m "
          f"ori_err={np.linalg.norm(T_top[:3,2]-z_down):.6f}")
    print(f"Top joints: [{', '.join(f'{v:.4f}' for v in q_top)}]")

    # Solve IK at every step from top→bottom using minimum-jerk profile.
    # Use q_zero as seed for the last few steps to guide IK toward zero.
    traj_rev = [q_top.copy()]
    q_prev = q_top.copy()

    print(f"Computing IK for {steps_half} steps (lift={lift_height:.3f}m)...")
    for i in range(1, steps_half + 1):
        t = min_jerk(i / steps_half)
        target = p_top + t * (p0 - p_top)
        # Near the end, increase regularization toward zero for smooth landing
        blend = max(0.0, (i - steps_half * 0.8) / (steps_half * 0.2))
        w_reg = 0.01 + blend * 5.0  # ramp from 0.01 to 5.01
        q_seed = (1 - blend) * q_prev + blend * q_zero
        q_sol = solve_ik(target, z_down, q_seed, w_reg=w_reg)
        q_prev = q_sol
        traj_rev.append(q_sol)

        if i % 10 == 0:
            T = fk(q_sol)
            pos_err = np.linalg.norm(T[:3, 3] - target)
            ori_err = np.linalg.norm(T[:3, 2] - z_down)
            print(f"  step {i}/{steps_half}: pos_err={pos_err:.6f}m ori_err={ori_err:.6f}")

    # Don't force zero — the IK with strong regularization converges naturally
    traj_up = np.array(list(reversed(traj_rev)), dtype=np.float32)

    # Down phase is just the reverse
    traj_down = traj_up[::-1].copy()

    # Full cycle: up then down
    full_traj = np.vstack([traj_up, traj_down[1:]])
    print(f"Trajectory: {len(full_traj)} steps ({steps_half} up + {steps_half} down)")

    # Verify smoothness
    dt = 0.1  # 10Hz
    dq = np.diff(full_traj, axis=0) / dt
    ddq = np.diff(dq, axis=0) / dt
    max_vel = np.max(np.abs(dq), axis=0)
    max_acc = np.max(np.abs(ddq), axis=0)
    print(f"Max joint velocities (rad/s): [{', '.join(f'{v:.3f}' for v in max_vel)}]")
    print(f"Max joint accel (rad/s²): [{', '.join(f'{v:.3f}' for v in max_acc)}]")

    return full_traj


def compute_trajectory_batch(steps_half, lift_height):
    """Batch-optimize the trajectory using pytorch-kinematics + LBFGS.

    Uses the sequential solution as a warm start, then jointly optimizes
    all waypoints for position/orientation tracking plus acceleration and
    jerk smoothness penalties.
    """
    import xml.etree.ElementTree as ET

    import pytorch_kinematics as pk
    import torch

    # --- Build kinematic chain from URDF ---
    # Strip visual/collision elements — pk's URDF parser chokes on empty <geometry/>
    urdf_path = Path(__file__).parent / "openarm_v10.urdf"
    tree = ET.parse(urdf_path)
    for link in tree.findall(".//link"):
        for tag in ("visual", "collision"):
            for elem in link.findall(tag):
                link.remove(elem)
    urdf_data = ET.tostring(tree.getroot(), encoding="unicode")
    chain = pk.build_serial_chain_from_urdf(urdf_data, "openarm_left_link8")

    # --- Sequential init (warm start) ---
    print("Computing sequential IK for warm start...")
    seq_traj = compute_trajectory(steps_half, lift_height)  # (N, 7)
    N = len(seq_traj)

    # --- Compute Cartesian targets in pk's frame (not the hand-coded FK frame) ---
    q_zero_t = torch.zeros(1, 7)
    tf0 = chain.forward_kinematics(q_zero_t, end_only=True)
    p0_pk = tf0.get_matrix()[0, :3, 3].detach().numpy().copy()
    z_down = np.array([0.0, 0.0, -1.0])

    p_top_pk = p0_pk.copy()
    p_top_pk[2] += lift_height

    # Up phase targets
    targets_up = []
    for i in range(steps_half + 1):
        t = min_jerk(i / steps_half)
        targets_up.append(p0_pk + t * (p_top_pk - p0_pk))
    # Down phase targets (reverse, skip first to avoid duplicate)
    targets_down = list(reversed(targets_up[:-1]))
    targets_all = targets_up + targets_down  # length = N

    targets_tensor = torch.tensor(np.array(targets_all), dtype=torch.float32)
    z_down_tensor = torch.tensor(z_down, dtype=torch.float32).unsqueeze(0).expand(N, -1)

    # Joint limits as tensors for clamping
    jl = torch.tensor(JOINT_BOUNDS, dtype=torch.float32)
    j_lo, j_hi = jl[:, 0], jl[:, 1]

    # --- Optimization with Adam (robust to projection/clamping) ---
    Q_init = torch.tensor(seq_traj, dtype=torch.float32)  # frozen reference
    Q = Q_init.clone().requires_grad_(True)

    w_pos, w_ori, w_acc, w_jerk, w_reg = 100.0, 50.0, 10.0, 1.0, 1.0

    optimizer = torch.optim.Adam([Q], lr=0.001)
    n_iters = 500

    print(f"Batch optimizing {N} waypoints ({N * 7} variables), {n_iters} iters...")
    for step in range(n_iters):
        optimizer.zero_grad()

        # Batch FK
        tf = chain.forward_kinematics(Q, end_only=True)
        mat = tf.get_matrix()  # (N, 4, 4)

        # Position tracking
        pos_err = mat[:, :3, 3] - targets_tensor
        cost = w_pos * (pos_err ** 2).sum()

        # Orientation tracking (EE z-axis → [0,0,-1])
        ori_err = mat[:, :3, 2] - z_down_tensor
        cost = cost + w_ori * (ori_err ** 2).sum()

        # Acceleration penalty (finite differences)
        acc = Q[2:] - 2 * Q[1:-1] + Q[:-2]
        cost = cost + w_acc * (acc ** 2).sum()

        # Jerk penalty
        jrk = Q[3:] - 3 * Q[2:-1] + 3 * Q[1:-2] - Q[:-3]
        cost = cost + w_jerk * (jrk ** 2).sum()

        # Regularization toward sequential solution (prevents config drift)
        cost = cost + w_reg * ((Q - Q_init) ** 2).sum()

        cost.backward()
        optimizer.step()

        # Project: clamp joint limits and pin endpoints
        with torch.no_grad():
            Q.clamp_(j_lo, j_hi)
            Q[0] = torch.zeros(7)

        if step % 100 == 0:
            print(f"  step {step}: loss={cost.item():.4f}")

    result = Q.detach().numpy().astype(np.float32)

    # Verify smoothness
    dt = 0.1
    dq = np.diff(result, axis=0) / dt
    ddq = np.diff(dq, axis=0) / dt
    dddq = np.diff(ddq, axis=0) / dt
    max_vel = np.max(np.abs(dq), axis=0)
    max_acc = np.max(np.abs(ddq), axis=0)
    max_jerk = np.max(np.abs(dddq), axis=0)
    print(f"[Batch] Max joint vel (rad/s): [{', '.join(f'{v:.3f}' for v in max_vel)}]")
    print(f"[Batch] Max joint acc (rad/s²): [{', '.join(f'{v:.3f}' for v in max_acc)}]")
    print(f"[Batch] Max joint jerk (rad/s³): [{', '.join(f'{v:.3f}' for v in max_jerk)}]")

    return result


def main():
    steps_half = int(os.getenv("STEPS_UP", "60"))
    lift_height = float(os.getenv("LIFT_HEIGHT", "0.12"))
    use_batch = os.getenv("USE_BATCH_IK", "true").lower() in ("1", "true", "yes")

    if use_batch:
        trajectory = compute_trajectory_batch(steps_half, lift_height)
    else:
        trajectory = compute_trajectory(steps_half, lift_height)
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
