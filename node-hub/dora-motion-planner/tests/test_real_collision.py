"""Collision validation test using real RealSense depth data.

Uses a captured depth frame from the actual camera setup to verify that
the trajectory collision checker works with real point clouds and produces
meaningful diagnostic output.

Fixture: tests/fixtures/realsense_frame.npz
  Captured via: python tests/capture_realsense_fixture.py --path anon/realsense
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import torch
import pytorch_kinematics as pk

from dora_motion_planner.collision_model import (
    CapsuleCollisionModel,
    OPENARM_CAPSULES,
    capsule_points_distance,
)
from dora_motion_planner.grasp_utils import grasp_pose_from_jaw_pixels
from dora_motion_planner.main import validate_trajectory, solve_ik
from dora_motion_planner.pointcloud import (
    depth_to_pointcloud,
    parse_camera_transform,
    transform_points,
    pointcloud_to_tensor,
)
from dora_motion_planner.trajectory_optimizer import TrajectoryOptimizer

# --- Paths ---
URDF_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "examples"
    / "openarm"
    / "openarm_v10.urdf"
)
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "realsense_frame.npz"

# --- Config matching real setup ---
END_EFFECTOR_LINK = "openarm_left_hand_tcp"
DOWNSAMPLE_STRIDE = 8
SAFETY_MARGIN = 0.02
GRASP_DEPTH_OFFSET = -0.01
FLOOR_HEIGHT = 0.005
APPROACH_MARGIN = 0.03
DEVICE = "cpu"

# Trajectory params (smaller for faster tests)
NUM_WAYPOINTS = 40
NUM_SEEDS = 2
MAX_ITERS = 100


def build_chain(end_effector_link: str = END_EFFECTOR_LINK) -> pk.Chain:
    with open(URDF_PATH) as f:
        urdf_str = f.read()
    root = ET.fromstring(urdf_str)
    for link in root.findall(".//link"):
        for tag in ("visual", "collision", "inertial"):
            for elem in link.findall(tag):
                link.remove(elem)
    return pk.build_serial_chain_from_urdf(
        ET.tostring(root, encoding="unicode"), end_effector_link
    )


@pytest.fixture
def real_data():
    """Load captured RealSense frame fixture."""
    if not FIXTURE_PATH.exists():
        pytest.skip(f"Fixture not found: {FIXTURE_PATH}")
    data = np.load(FIXTURE_PATH, allow_pickle=True)
    return {
        "depth": data["depth"],
        "intrinsics": tuple(data["intrinsics"]),
        "image_size": tuple(data["image_size"]),
        "camera_transform_str": str(data["camera_transform_str"]),
    }


@pytest.fixture
def arm_setup():
    """Build chain, capsule model, optimizer for left arm."""
    device = torch.device(DEVICE)
    chain = build_chain()
    joint_limits = chain.get_joint_limits()
    num_joints = len(chain.get_joint_parameter_names(exclude_fixed=True))
    ik_chain = chain.to(dtype=torch.float32, device=str(device))
    capsule_model = CapsuleCollisionModel(OPENARM_CAPSULES, device)
    optimizer = TrajectoryOptimizer(
        chain=chain,
        capsule_model=capsule_model,
        joint_limits=joint_limits,
        device=device,
        safety_margin=SAFETY_MARGIN,
    )
    return {
        "chain": chain,
        "ik_chain": ik_chain,
        "capsule_model": capsule_model,
        "optimizer": optimizer,
        "joint_limits": joint_limits,
        "num_joints": num_joints,
        "device": device,
    }


@pytest.fixture
def point_cloud(real_data):
    """Build robot-frame point cloud from real depth data."""
    depth = real_data["depth"]
    fx, fy, cx, cy = real_data["intrinsics"]
    w, h = real_data["image_size"]
    cam_t, cam_rot = parse_camera_transform(real_data["camera_transform_str"])

    pc_cam = depth_to_pointcloud(
        depth, fx, fy, cx, cy, width=w, height=h, stride=DOWNSAMPLE_STRIDE
    )
    assert len(pc_cam) > 0, "No valid depth points"
    pc_robot = transform_points(pc_cam, cam_t, cam_rot)
    return pointcloud_to_tensor(pc_robot, torch.device(DEVICE))


class TestRealPointCloud:
    """Verify the real point cloud is reasonable."""

    def test_point_cloud_has_points(self, point_cloud):
        assert point_cloud.shape[0] > 100, "Point cloud too sparse"
        assert point_cloud.shape[1] == 3

    def test_point_cloud_in_workspace(self, point_cloud):
        """Points should be in front of the robot, within reach."""
        pc = point_cloud.numpy()
        # X should be roughly in [-0.5, 0.5] (left-right)
        assert pc[:, 0].min() > -2.0
        assert pc[:, 0].max() < 2.0
        # Z should be positive (above ground) and reasonable
        assert pc[:, 2].min() > -0.5
        assert pc[:, 2].max() < 2.0
        print(f"\n  Point cloud: {len(pc)} points")
        print(f"  X range: [{pc[:, 0].min():.3f}, {pc[:, 0].max():.3f}]")
        print(f"  Y range: [{pc[:, 1].min():.3f}, {pc[:, 1].max():.3f}]")
        print(f"  Z range: [{pc[:, 2].min():.3f}, {pc[:, 2].max():.3f}]")


class TestCapsuleFKKeyMatch:
    """Verify capsule link names match FK transform keys — the suspected bug."""

    def test_all_capsule_links_in_fk(self, arm_setup):
        """Every capsule link name must appear in FK transform keys."""
        ik_chain = arm_setup["ik_chain"]
        device = arm_setup["device"]
        num_joints = arm_setup["num_joints"]

        with torch.no_grad():
            transforms = ik_chain.forward_kinematics(
                torch.zeros(1, num_joints, device=device), end_only=False
            )

        fk_keys = set(transforms.keys())
        capsule_links = set(OPENARM_CAPSULES.keys())
        missing = capsule_links - fk_keys

        print(f"\n  FK keys: {sorted(fk_keys)}")
        print(f"  Capsule links: {sorted(capsule_links)}")
        if missing:
            print(f"  MISSING: {missing}")

        assert len(missing) == 0, (
            f"Capsule links not found in FK transforms: {missing}. "
            f"Collision avoidance is silently broken for these links!"
        )


class TestValidateTrajectoryWithRealData:
    """Run validate_trajectory with real point cloud data."""

    def test_home_to_home_no_collision(self, arm_setup, point_cloud):
        """A zero-motion trajectory at home should not collide (arm is above table)."""
        ik_chain = arm_setup["ik_chain"]
        capsule_model = arm_setup["capsule_model"]
        num_joints = arm_setup["num_joints"]

        # Trajectory: stay at home position
        home = torch.zeros(num_joints, dtype=torch.float32)
        traj = home.unsqueeze(0).expand(10, -1).clone()

        collisions = validate_trajectory(
            ik_chain, capsule_model, traj, point_cloud, SAFETY_MARGIN
        )
        colliding_wps = len(set(c[0] for c in collisions))
        if collisions:
            worst = min(collisions, key=lambda c: c[2])
            print(f"\n  Home trajectory: {colliding_wps}/10 waypoints in collision")
            print(f"  Worst: t={worst[0]} link={worst[1]} dist={worst[2]:.4f}m")
        else:
            print("\n  Home trajectory: clear (no collisions)")

    def test_straight_line_trajectory(self, arm_setup, point_cloud):
        """A linear interpolation to a random goal — check collision report."""
        ik_chain = arm_setup["ik_chain"]
        capsule_model = arm_setup["capsule_model"]
        optimizer = arm_setup["optimizer"]
        num_joints = arm_setup["num_joints"]
        device = arm_setup["device"]

        q_start = torch.zeros(num_joints, dtype=torch.float32, device=device)
        # Some non-zero goal within joint limits
        lower = torch.tensor(arm_setup["joint_limits"][0], dtype=torch.float32)
        upper = torch.tensor(arm_setup["joint_limits"][1], dtype=torch.float32)
        q_goal = (lower + upper) / 2  # midpoint of joint limits

        # Linear interpolation (no optimization — worst case for collisions)
        t = torch.linspace(0, 1, NUM_WAYPOINTS).unsqueeze(1)
        traj = q_start.cpu() + t * (q_goal.cpu() - q_start.cpu())

        collisions = validate_trajectory(
            ik_chain, capsule_model, traj, point_cloud, SAFETY_MARGIN
        )
        colliding_wps = len(set(c[0] for c in collisions))
        print(f"\n  Straight-line trajectory: {colliding_wps}/{NUM_WAYPOINTS} waypoints in collision")
        if collisions:
            worst = min(collisions, key=lambda c: c[2])
            print(f"  Worst: t={worst[0]} link={worst[1]} dist={worst[2]:.4f}m "
                  f"(penetration={-worst[2]*1000:.1f}mm)")
            # Show unique colliding links
            colliding_links = set(c[1] for c in collisions)
            print(f"  Colliding links: {sorted(colliding_links)}")

    def test_optimized_trajectory_fewer_collisions(self, arm_setup, point_cloud):
        """An optimized trajectory should have fewer collisions than straight-line."""
        ik_chain = arm_setup["ik_chain"]
        capsule_model = arm_setup["capsule_model"]
        optimizer = arm_setup["optimizer"]
        num_joints = arm_setup["num_joints"]
        device = arm_setup["device"]

        q_start = torch.zeros(num_joints, dtype=torch.float32, device=device)
        lower = torch.tensor(arm_setup["joint_limits"][0], dtype=torch.float32, device=device)
        upper = torch.tensor(arm_setup["joint_limits"][1], dtype=torch.float32, device=device)
        q_goal = (lower + upper) / 2

        # Straight-line baseline
        t = torch.linspace(0, 1, NUM_WAYPOINTS).unsqueeze(1)
        traj_linear = q_start.cpu() + t * (q_goal.cpu() - q_start.cpu())
        collisions_linear = validate_trajectory(
            ik_chain, capsule_model, traj_linear, point_cloud, SAFETY_MARGIN
        )

        # Optimized trajectory
        best_traj, best_cost = optimizer.optimize(
            q_start=q_start,
            q_goal=q_goal,
            point_cloud=point_cloud,
            T=NUM_WAYPOINTS,
            num_seeds=NUM_SEEDS,
            max_iters=MAX_ITERS,
        )
        collisions_opt = validate_trajectory(
            ik_chain, capsule_model, best_traj, point_cloud, SAFETY_MARGIN
        )

        linear_count = len(set(c[0] for c in collisions_linear))
        opt_count = len(set(c[0] for c in collisions_opt))
        print(f"\n  Straight-line collisions: {linear_count}/{NUM_WAYPOINTS}")
        print(f"  Optimized collisions: {opt_count}/{NUM_WAYPOINTS}")
        print(f"  Optimizer cost: {best_cost:.4f}")

        if collisions_opt:
            worst = min(collisions_opt, key=lambda c: c[2])
            print(f"  Worst optimized: t={worst[0]} link={worst[1]} "
                  f"penetration={-worst[2]*1000:.1f}mm")

        # The optimizer should do at least as well as straight-line
        # (it starts from linear interpolation)
        assert opt_count <= linear_count or linear_count == 0, (
            f"Optimizer made collisions worse: {opt_count} vs {linear_count}"
        )

    def test_grasp_toward_table_surface(self, arm_setup, point_cloud, real_data):
        """Plan a grasp trajectory toward the table surface (into point cloud).

        This is the real-world scenario: arm reaches down to grab something
        on the table. The trajectory SHOULD collide if collision avoidance
        isn't working, because the EE needs to go to the table surface.
        """
        ik_chain = arm_setup["ik_chain"]
        capsule_model = arm_setup["capsule_model"]
        optimizer = arm_setup["optimizer"]
        num_joints = arm_setup["num_joints"]
        device = arm_setup["device"]
        joint_limits = arm_setup["joint_limits"]

        # Target: center of the point cloud (on the table surface)
        pc_np = point_cloud.numpy()
        target_pos = pc_np.mean(axis=0)  # center of table surface
        # Top-down orientation: look straight down
        target_xyzrpy = np.array([
            target_pos[0], target_pos[1], target_pos[2],
            np.pi, 0.0, 0.0,  # top-down (180° roll = Z pointing down)
        ], dtype=np.float32)

        print(f"\n  Target (table center): [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")

        q_goal = solve_ik(
            ik_chain, target_xyzrpy, torch.zeros(num_joints, device=device),
            joint_limits, device, num_seeds=16, max_iters=1000,
        )
        if q_goal is None:
            # Try position above table (approach height)
            target_xyzrpy[2] += 0.05
            print(f"  IK failed at table, trying z={target_xyzrpy[2]:.3f}")
            q_goal = solve_ik(
                ik_chain, target_xyzrpy, torch.zeros(num_joints, device=device),
                joint_limits, device, num_seeds=16, max_iters=1000,
            )
        if q_goal is None:
            pytest.skip("IK failed to reach table surface")

        # Verify EE position
        with torch.no_grad():
            fk = ik_chain.forward_kinematics(q_goal.unsqueeze(0))
            ee_pos = fk.get_matrix()[0, :3, 3].cpu().numpy()
        print(f"  IK solution EE: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")

        q_start = torch.zeros(num_joints, dtype=torch.float32, device=device)

        # 1. Straight-line (no optimization)
        t = torch.linspace(0, 1, NUM_WAYPOINTS).unsqueeze(1)
        traj_linear = q_start.cpu() + t * (q_goal.cpu() - q_start.cpu())
        collisions_linear = validate_trajectory(
            ik_chain, capsule_model, traj_linear, point_cloud, SAFETY_MARGIN
        )
        linear_wps = len(set(c[0] for c in collisions_linear))

        # 2. Optimized trajectory
        best_traj, best_cost = optimizer.optimize(
            q_start=q_start, q_goal=q_goal, point_cloud=point_cloud,
            T=NUM_WAYPOINTS, num_seeds=NUM_SEEDS, max_iters=MAX_ITERS,
        )
        collisions_opt = validate_trajectory(
            ik_chain, capsule_model, best_traj, point_cloud, SAFETY_MARGIN
        )
        opt_wps = len(set(c[0] for c in collisions_opt))

        print(f"  Straight-line: {linear_wps}/{NUM_WAYPOINTS} waypoints in collision")
        print(f"  Optimized:     {opt_wps}/{NUM_WAYPOINTS} waypoints in collision (cost={best_cost:.4f})")

        if collisions_linear:
            worst = min(collisions_linear, key=lambda c: c[2])
            links = sorted(set(c[1] for c in collisions_linear))
            print(f"  Linear worst: t={worst[0]} link={worst[1]} penetration={-worst[2]*1000:.1f}mm")
            print(f"  Linear colliding links: {links}")

        if collisions_opt:
            worst = min(collisions_opt, key=lambda c: c[2])
            links = sorted(set(c[1] for c in collisions_opt))
            print(f"  Optimized worst: t={worst[0]} link={worst[1]} penetration={-worst[2]*1000:.1f}mm")
            print(f"  Optimized colliding links: {links}")

        # The goal itself is at/below table, so some collisions near the end
        # are expected. The key diagnostic: does the optimizer reduce them?
        if linear_wps > 0:
            print(f"  Collision reduction: {linear_wps} -> {opt_wps} "
                  f"({100*(1 - opt_wps/linear_wps):.0f}% fewer)")

    def test_validate_with_no_pointcloud(self, arm_setup):
        """validate_trajectory should return empty list when no point cloud."""
        ik_chain = arm_setup["ik_chain"]
        capsule_model = arm_setup["capsule_model"]
        num_joints = arm_setup["num_joints"]

        traj = torch.zeros(10, num_joints, dtype=torch.float32)
        collisions = validate_trajectory(ik_chain, capsule_model, traj, None, SAFETY_MARGIN)
        assert collisions == []
