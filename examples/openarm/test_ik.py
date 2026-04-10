#!/usr/bin/env python3
"""Standalone test to validate IK with the OpenArm URDF.

Run on remote:
    pip install pytorch-kinematics torch numpy
    python test_ik.py
"""

import os
from pathlib import Path

import numpy as np
import pytorch_kinematics as pk
import torch
from pytorch_kinematics.transforms.rotation_conversions import matrix_to_euler_angles

URDF_PATH = os.getenv("URDF_PATH", str(Path(__file__).parent / "openarm_v10.urdf"))
END_EFFECTOR_LINK = os.getenv("END_EFFECTOR_LINK", "openarm_left_link8")


def main():
    print(f"Loading URDF: {URDF_PATH}")
    print(f"End-effector link: {END_EFFECTOR_LINK}")

    with open(URDF_PATH, "rb") as f:
        urdf_data = f.read()

    chain = pk.build_serial_chain_from_urdf(urdf_data, END_EFFECTOR_LINK)
    joint_names = chain.get_joint_parameter_names(exclude_fixed=True)
    joint_limits = chain.get_joint_limits()
    num_joints = len(joint_names)

    print(f"\nKinematic chain: {num_joints} joints")
    print(f"Joint names: {joint_names}")
    print(f"Joint limits (lower): {joint_limits[0].tolist()}")
    print(f"Joint limits (upper): {joint_limits[1].tolist()}")

    # --- Test FK with zero joints ---
    print("\n--- FK at zero configuration ---")
    q_zero = torch.zeros(1, num_joints)
    tf_zero = chain.forward_kinematics(q_zero, end_only=True)
    mat = tf_zero.get_matrix()
    xyz = mat[0, :3, 3].detach().numpy()
    rpy = matrix_to_euler_angles(mat[0, :3, :3].unsqueeze(0), "XYZ").squeeze().detach().numpy()
    print(f"EE position (xyz): {xyz}")
    print(f"EE orientation (rpy): {rpy}")

    # --- Test FK at mid-range configuration ---
    print("\n--- FK at mid-range configuration ---")
    lower = joint_limits[0]
    upper = joint_limits[1]
    q_mid = ((lower + upper) / 2).unsqueeze(0)
    print(f"Mid-range joints: {q_mid.squeeze().tolist()}")
    tf_mid = chain.forward_kinematics(q_mid, end_only=True)
    mat = tf_mid.get_matrix()
    xyz = mat[0, :3, 3].detach().numpy()
    rpy = matrix_to_euler_angles(mat[0, :3, :3].unsqueeze(0), "XYZ").squeeze().detach().numpy()
    print(f"EE position (xyz): {xyz}")
    print(f"EE orientation (rpy): {rpy}")

    # --- Test IK round-trip ---
    print("\n--- IK round-trip test ---")
    # Use a known configuration to compute a target pose, then solve IK to recover it
    q_test = torch.zeros(1, num_joints)
    # Set some non-zero joint values within limits
    q_test[0, 0] = 0.3   # J1
    q_test[0, 1] = -0.5  # J2
    q_test[0, 3] = 1.0   # J4
    q_test[0, 4] = 0.2   # J5
    print(f"Test joints: {q_test.squeeze().tolist()}")

    # FK to get target pose
    target_tf = chain.forward_kinematics(q_test, end_only=True)
    target_mat = target_tf.get_matrix()
    target_xyz = target_mat[0, :3, 3].detach().numpy()
    target_rpy = matrix_to_euler_angles(
        target_mat[0, :3, :3].unsqueeze(0), "XYZ"
    ).squeeze().detach().numpy()
    print(f"Target EE position: {target_xyz}")
    print(f"Target EE orientation: {target_rpy}")

    # IK to recover joint angles
    ik_solver = pk.PseudoInverseIK(
        chain,
        max_iterations=100_000,
        retry_configs=torch.zeros(1, num_joints).requires_grad_(True),
        joint_limits=torch.tensor(chain.get_joint_limits()),
        early_stopping_any_converged=True,
        debug=False,
        lr=0.05,
        pos_tolerance=0.005,
        rot_tolerance=0.05,
    )
    solution = ik_solver.solve(target_tf)

    print(f"\nIK converged: pos_err={solution.err_pos:.6f}, rot_err={solution.err_rot:.6f}")
    q_solution = solution.solutions.detach().squeeze().numpy()
    print(f"IK solution joints: {q_solution.tolist()}")

    # Verify FK(IK(target)) ≈ target
    q_sol_tensor = solution.solutions.detach()
    verify_tf = chain.forward_kinematics(q_sol_tensor, end_only=True)
    verify_mat = verify_tf.get_matrix()
    verify_xyz = verify_mat[0, :3, 3].detach().numpy()
    verify_rpy = matrix_to_euler_angles(
        verify_mat[0, :3, :3].unsqueeze(0), "XYZ"
    ).squeeze().detach().numpy()

    pos_err = np.linalg.norm(verify_xyz - target_xyz)
    rot_err = np.linalg.norm(verify_rpy - target_rpy)
    print(f"\nVerification FK(IK(target)):")
    print(f"  Position: {verify_xyz} (error: {pos_err:.6f} m)")
    print(f"  Orientation: {verify_rpy} (error: {rot_err:.6f} rad)")

    if pos_err < 0.01 and rot_err < 0.1:
        print("\nIK round-trip: PASSED")
    else:
        print("\nIK round-trip: FAILED (errors too large)")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
