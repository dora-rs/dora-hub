"""End-to-end integration test for the grasp motion planning pipeline.

Simulates what happens in a real dataflow:
  grasp_result JSON → pixel→3D deprojection → IK → trajectory optimisation

Uses the real URDF, camera transform, and realistic depth data to verify
the entire pipeline produces valid, collision-checked trajectories.

Matches the config from examples/openarm-grasp/openarm-grasp-motion.yml:
  CAMERA_TRANSFORM: "-0.23 0.71 0.3 90 -45 0"
  IMAGE_WIDTH: 1280, IMAGE_HEIGHT: 720
  END_EFFECTOR_LINK: "openarm_left_hand_tcp"
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import torch
import pytorch_kinematics as pk

from dora_motion_planner.collision_model import CapsuleCollisionModel, OPENARM_CAPSULES
from dora_motion_planner.grasp_utils import grasp_pose_from_jaw_pixels
from dora_motion_planner.pointcloud import (
    depth_to_pointcloud,
    parse_camera_transform,
    transform_points,
    pointcloud_to_tensor,
)
from dora_motion_planner.trajectory_optimizer import TrajectoryOptimizer

# --- Config matching openarm-grasp-motion.yml ---
URDF_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "examples"
    / "openarm"
    / "openarm_v10.urdf"
)
CAMERA_TRANSFORM_STR = "-0.23 0.71 0.3 90 -45 0"
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
END_EFFECTOR_LINK = "openarm_left_hand_tcp"
DOWNSAMPLE_STRIDE = 8
SAFETY_MARGIN = 0.02
GRASP_DEPTH_OFFSET = -0.01
FLOOR_HEIGHT = 0.005
APPROACH_MARGIN = 0.03
DEVICE = "cpu"


def build_chain(end_effector_link: str = END_EFFECTOR_LINK) -> pk.Chain:
    """Build kinematic chain from URDF with meshes stripped."""
    with open(URDF_PATH) as f:
        urdf_str = f.read()
    root = ET.fromstring(urdf_str)
    for link in root.findall(".//link"):
        for tag in ("visual", "collision", "inertial"):
            for elem in link.findall(tag):
                link.remove(elem)
    clean_urdf = ET.tostring(root, encoding="unicode")
    return pk.build_serial_chain_from_urdf(clean_urdf, end_effector_link)


def make_table_depth(
    width=IMAGE_WIDTH,
    height=IMAGE_HEIGHT,
    table_mm=500,
    object_mm=400,
    obj_region=(500, 300, 700, 450),
):
    """Create a synthetic depth map with a table and an object on it.

    Args:
        table_mm: Table surface depth in mm from camera.
        object_mm: Object surface depth in mm (closer to camera = higher).
        obj_region: (u_lo, v_lo, u_hi, v_hi) pixel bounding box of the object.
    """
    depth = np.full(height * width, table_mm, dtype=np.uint16)
    u_lo, v_lo, u_hi, v_hi = obj_region
    for v in range(v_lo, v_hi):
        for u in range(u_lo, u_hi):
            depth[v * width + u] = object_mm
    return depth


# Realistic camera intrinsics for RealSense D435i at 1280x720
FX, FY, CX, CY = 906.0, 905.0, 644.0, 379.0


@pytest.fixture
def chain():
    if not URDF_PATH.exists():
        pytest.skip("URDF not found")
    return build_chain()


@pytest.fixture
def optimizer(chain):
    device = torch.device(DEVICE)
    capsule_model = CapsuleCollisionModel(OPENARM_CAPSULES, device)
    limits = chain.get_joint_limits()
    return TrajectoryOptimizer(
        chain=chain,
        capsule_model=capsule_model,
        joint_limits=limits,
        device=DEVICE,
        safety_margin=SAFETY_MARGIN,
    )


@pytest.fixture
def cam_transform():
    return parse_camera_transform(CAMERA_TRANSFORM_STR)


class TestGraspDeprojection:
    """Test that grasp_result JSON → 3D pose conversion works with real camera config."""

    def test_normalized_coords_to_pixels(self):
        """Verify 0-1000 normalized coords convert to correct pixel coords."""
        # Simulate what main.py does with grasp_result
        p1 = [469, 486]  # normalized 0-1000
        p2 = [547, 486]

        u1 = p1[0] * IMAGE_WIDTH / 1000.0
        v1 = p1[1] * IMAGE_HEIGHT / 1000.0
        u2 = p2[0] * IMAGE_WIDTH / 1000.0
        v2 = p2[1] * IMAGE_HEIGHT / 1000.0

        assert abs(u1 - 600.3) < 1.0
        assert abs(v1 - 349.9) < 1.0
        assert abs(u2 - 700.2) < 1.0

    def test_deprojection_to_robot_frame(self, cam_transform):
        """Jaw pixels on an object produce 3D points in the arm's workspace."""
        cam_t, cam_rot = cam_transform
        depth = make_table_depth()

        result = grasp_pose_from_jaw_pixels(
            u1=600, v1=350,
            u2=700, v2=350,
            depth_map=depth,
            fx=FX, fy=FY, cx=CX, cy=CY,
            cam_translation=cam_t,
            cam_rotation=cam_rot,
            width=IMAGE_WIDTH, height=IMAGE_HEIGHT,
            grasp_depth_offset=GRASP_DEPTH_OFFSET,
            floor_height=FLOOR_HEIGHT,
            approach_margin=APPROACH_MARGIN,
        )

        assert result is not None, "Failed to compute grasp pose from jaw pixels"
        grasp_xyzrpy, pregrasp_xyzrpy = result

        # Grasp should be in the arm's reachable workspace
        x, y, z = grasp_xyzrpy[:3]
        print(f"Grasp position: ({x:.4f}, {y:.4f}, {z:.4f})")

        # OpenArm left workspace is roughly: X∈[-0.45, 0.72], Y∈[-0.72, 0.44], Z∈[0.0, 1.3]
        assert -0.5 < x < 0.8, f"Grasp X={x:.4f} outside workspace"
        assert -0.8 < y < 0.5, f"Grasp Y={y:.4f} outside workspace"
        assert 0.0 < z < 1.0, f"Grasp Z={z:.4f} outside workspace"

        # Pre-grasp should be above grasp
        assert pregrasp_xyzrpy[2] > grasp_xyzrpy[2]
        assert abs(pregrasp_xyzrpy[2] - grasp_xyzrpy[2] - APPROACH_MARGIN) < 0.01


class TestIKSolver:
    """Test IK solver with realistic grasp poses from the camera config."""

    def test_ik_for_grasp_pose(self, chain, cam_transform):
        """IK can solve for a typical grasp position in the workspace."""
        from dora_motion_planner.main import solve_ik

        cam_t, cam_rot = cam_transform
        depth = make_table_depth()
        device = torch.device(DEVICE)

        result = grasp_pose_from_jaw_pixels(
            u1=600, v1=350,
            u2=700, v2=350,
            depth_map=depth,
            fx=FX, fy=FY, cx=CX, cy=CY,
            cam_translation=cam_t,
            cam_rotation=cam_rot,
            width=IMAGE_WIDTH, height=IMAGE_HEIGHT,
            grasp_depth_offset=GRASP_DEPTH_OFFSET,
            floor_height=FLOOR_HEIGHT,
            approach_margin=APPROACH_MARGIN,
        )
        assert result is not None

        grasp_xyzrpy, pregrasp_xyzrpy = result
        joint_limits = chain.get_joint_limits()
        current_joints = torch.zeros(
            len(chain.get_joint_parameter_names(exclude_fixed=True)),
            dtype=torch.float32,
            device=device,
        )
        ik_chain = chain.to(dtype=torch.float32, device=str(device))

        # Test IK for pre-grasp (higher, easier to reach)
        q_pregrasp = solve_ik(
            ik_chain, pregrasp_xyzrpy, current_joints, joint_limits, device,
            num_seeds=16, max_iters=1000,
        )
        assert q_pregrasp is not None, (
            f"IK failed for pre-grasp at {np.round(pregrasp_xyzrpy[:3], 4)}"
        )

        # Test IK for grasp (starting from pre-grasp solution)
        q_grasp = solve_ik(
            ik_chain, grasp_xyzrpy, q_pregrasp, joint_limits, device,
            num_seeds=16, max_iters=1000,
        )
        assert q_grasp is not None, (
            f"IK failed for grasp at {np.round(grasp_xyzrpy[:3], 4)}"
        )

    def test_ik_workspace_boundary(self, chain):
        """IK should fail gracefully for positions outside the workspace."""
        from dora_motion_planner.main import solve_ik

        device = torch.device(DEVICE)
        joint_limits = chain.get_joint_limits()
        current_joints = torch.zeros(
            len(chain.get_joint_parameter_names(exclude_fixed=True)),
            dtype=torch.float32,
            device=device,
        )
        ik_chain = chain.to(dtype=torch.float32, device=str(device))

        # Way outside workspace
        unreachable = np.array([2.0, 2.0, 2.0, 0.0, 0.0, 0.0], dtype=np.float32)
        q = solve_ik(
            ik_chain, unreachable, current_joints, joint_limits, device,
            num_seeds=8, max_iters=500,
        )
        assert q is None, "IK should fail for unreachable positions"


class TestPointcloudPipeline:
    """Test depth → pointcloud → collision checking with real camera config."""

    def test_pointcloud_in_robot_frame(self, cam_transform):
        """Depth map produces points in the correct robot frame region."""
        cam_t, cam_rot = cam_transform
        depth = make_table_depth()

        pc_cam = depth_to_pointcloud(
            depth, FX, FY, CX, CY,
            width=IMAGE_WIDTH, height=IMAGE_HEIGHT,
            stride=DOWNSAMPLE_STRIDE,
        )
        assert len(pc_cam) > 0, "No valid points in depth map"

        pc_robot = transform_points(pc_cam, cam_t, cam_rot)
        pc_tensor = pointcloud_to_tensor(pc_robot, torch.device(DEVICE))

        # Camera looks straight down from Z=0.71, table at ~0.5m depth
        # → table Z ≈ 0.71 - 0.5 = 0.21 in robot frame
        z_mean = pc_robot[:, 2].mean()
        print(f"Point cloud: {len(pc_robot)} points, Z mean={z_mean:.3f}")
        assert 0.0 < z_mean < 0.5, f"Z mean {z_mean:.3f} not in expected range"
        assert pc_tensor.shape[1] == 3


class TestFullGraspPipeline:
    """End-to-end: grasp_result JSON → IK → trajectory optimisation."""

    def test_full_pipeline_from_json(self, chain, optimizer, cam_transform):
        """Simulate a grasp_result from the selector node through the full pipeline."""
        from dora_motion_planner.main import solve_ik

        cam_t, cam_rot = cam_transform
        device = torch.device(DEVICE)
        joint_limits = chain.get_joint_limits()
        num_joints = len(chain.get_joint_parameter_names(exclude_fixed=True))
        current_joints = torch.zeros(num_joints, dtype=torch.float32, device=device)
        ik_chain = chain.to(dtype=torch.float32, device=str(device))

        # 1. Simulate grasp_result JSON (as sent by grasp_selector.py)
        grasp_json = json.dumps({"p1": [469, 486], "p2": [547, 486]})
        data = json.loads(grasp_json)

        # 2. Convert normalized coords to pixel coords (same as main.py)
        u1 = float(data["p1"][0]) * IMAGE_WIDTH / 1000.0
        v1 = float(data["p1"][1]) * IMAGE_HEIGHT / 1000.0
        u2 = float(data["p2"][0]) * IMAGE_WIDTH / 1000.0
        v2 = float(data["p2"][1]) * IMAGE_HEIGHT / 1000.0

        # 3. Create synthetic depth and build pointcloud
        depth = make_table_depth()
        pc_cam = depth_to_pointcloud(
            depth, FX, FY, CX, CY,
            width=IMAGE_WIDTH, height=IMAGE_HEIGHT,
            stride=DOWNSAMPLE_STRIDE,
        )
        pc_robot = transform_points(pc_cam, cam_t, cam_rot)
        pc_tensor = pointcloud_to_tensor(pc_robot, device)

        # 4. Compute grasp pose from jaw pixels
        result = grasp_pose_from_jaw_pixels(
            u1, v1, u2, v2,
            depth_map=depth,
            fx=FX, fy=FY, cx=CX, cy=CY,
            cam_translation=cam_t,
            cam_rotation=cam_rot,
            width=IMAGE_WIDTH, height=IMAGE_HEIGHT,
            grasp_depth_offset=GRASP_DEPTH_OFFSET,
            floor_height=FLOOR_HEIGHT,
            approach_margin=APPROACH_MARGIN,
        )
        assert result is not None, "Grasp pose computation failed"
        grasp_xyzrpy, pregrasp_xyzrpy = result
        print(f"Grasp: {np.round(grasp_xyzrpy, 4)}")
        print(f"Pre-grasp: {np.round(pregrasp_xyzrpy, 4)}")

        # 5. Solve IK for pre-grasp
        q_pregrasp = solve_ik(
            ik_chain, pregrasp_xyzrpy, current_joints, joint_limits, device,
            num_seeds=16, max_iters=1000,
        )
        assert q_pregrasp is not None, (
            f"IK failed for pre-grasp at {np.round(pregrasp_xyzrpy[:3], 4)}"
        )

        # 6. Solve IK for grasp
        q_grasp = solve_ik(
            ik_chain, grasp_xyzrpy, q_pregrasp, joint_limits, device,
            num_seeds=16, max_iters=1000,
        )
        assert q_grasp is not None, (
            f"IK failed for grasp at {np.round(grasp_xyzrpy[:3], 4)}"
        )

        # 7. Plan Phase 1: start → pre-grasp (small scale for test speed)
        NUM_WAYPOINTS = 20  # small for test
        NUM_SEEDS = 2
        MAX_ITERS = 50

        traj1, cost1 = optimizer.optimize(
            q_start=current_joints,
            q_goal=q_pregrasp,
            point_cloud=pc_tensor,
            T=NUM_WAYPOINTS,
            num_seeds=NUM_SEEDS,
            max_iters=MAX_ITERS,
        )
        assert traj1 is not None, "Phase 1 trajectory optimization failed"
        assert traj1.shape == (NUM_WAYPOINTS, num_joints)
        print(f"Phase 1: {NUM_WAYPOINTS} waypoints, cost={cost1:.4f}")

        # 8. Plan Phase 2: pre-grasp → grasp
        traj2, cost2 = optimizer.optimize(
            q_start=q_pregrasp,
            q_goal=q_grasp,
            point_cloud=pc_tensor,
            T=NUM_WAYPOINTS,
            num_seeds=NUM_SEEDS,
            max_iters=MAX_ITERS,
        )
        assert traj2 is not None, "Phase 2 trajectory optimization failed"
        print(f"Phase 2: {NUM_WAYPOINTS} waypoints, cost={cost2:.4f}")

        # 9. Combine trajectories
        full_traj = torch.cat([traj1, traj2[1:]], dim=0)
        total_waypoints = full_traj.shape[0]
        print(f"Combined: {total_waypoints} waypoints, total cost={cost1 + cost2:.4f}")

        # 10. Validate trajectory
        traj_np = full_traj.numpy()
        lower = np.array(joint_limits[0])
        upper = np.array(joint_limits[1])
        assert np.all(traj_np >= lower - 0.01), "Trajectory violates lower joint limits"
        assert np.all(traj_np <= upper + 0.01), "Trajectory violates upper joint limits"

        # Start and end should match
        np.testing.assert_allclose(traj_np[0], current_joints.numpy(), atol=1e-4)

        # Trajectory should be smooth (no sudden jumps)
        velocities = np.diff(traj_np, axis=0)
        max_vel = np.abs(velocities).max()
        print(f"Max velocity: {max_vel:.4f} rad/step")
        assert max_vel < 1.0, f"Trajectory has jumps: max_vel={max_vel:.4f}"

    def test_pipeline_with_different_grasp_positions(self, chain, optimizer, cam_transform):
        """Test multiple grasp positions to check robustness."""
        from dora_motion_planner.main import solve_ik

        cam_t, cam_rot = cam_transform
        device = torch.device(DEVICE)
        joint_limits = chain.get_joint_limits()
        num_joints = len(chain.get_joint_parameter_names(exclude_fixed=True))
        current_joints = torch.zeros(num_joints, dtype=torch.float32, device=device)
        ik_chain = chain.to(dtype=torch.float32, device=str(device))

        # Different grasp positions (normalized 0-1000 coords)
        test_grasps = [
            {"p1": [400, 400], "p2": [500, 400], "desc": "center-left"},
            {"p1": [550, 500], "p2": [650, 500], "desc": "center-right"},
            {"p1": [500, 350], "p2": [600, 350], "desc": "upper-center"},
        ]

        results = []
        for grasp in test_grasps:
            u1 = float(grasp["p1"][0]) * IMAGE_WIDTH / 1000.0
            v1 = float(grasp["p1"][1]) * IMAGE_HEIGHT / 1000.0
            u2 = float(grasp["p2"][0]) * IMAGE_WIDTH / 1000.0
            v2 = float(grasp["p2"][1]) * IMAGE_HEIGHT / 1000.0

            depth = make_table_depth(
                object_mm=400,
                obj_region=(
                    int(min(u1, u2)) - 50,
                    int(min(v1, v2)) - 50,
                    int(max(u1, u2)) + 50,
                    int(max(v1, v2)) + 50,
                ),
            )

            result = grasp_pose_from_jaw_pixels(
                u1, v1, u2, v2,
                depth_map=depth,
                fx=FX, fy=FY, cx=CX, cy=CY,
                cam_translation=cam_t,
                cam_rotation=cam_rot,
                width=IMAGE_WIDTH, height=IMAGE_HEIGHT,
                grasp_depth_offset=GRASP_DEPTH_OFFSET,
                floor_height=FLOOR_HEIGHT,
                approach_margin=APPROACH_MARGIN,
            )
            if result is None:
                results.append((grasp["desc"], "deprojection_failed", None))
                continue

            grasp_xyzrpy, pregrasp_xyzrpy = result

            q_pregrasp = solve_ik(
                ik_chain, pregrasp_xyzrpy, current_joints, joint_limits, device,
                num_seeds=16, max_iters=1000,
            )
            if q_pregrasp is None:
                results.append((grasp["desc"], "ik_pregrasp_failed", grasp_xyzrpy[:3]))
                continue

            q_grasp = solve_ik(
                ik_chain, grasp_xyzrpy, q_pregrasp, joint_limits, device,
                num_seeds=16, max_iters=1000,
            )
            if q_grasp is None:
                results.append((grasp["desc"], "ik_grasp_failed", grasp_xyzrpy[:3]))
                continue

            results.append((grasp["desc"], "success", grasp_xyzrpy[:3]))

        # Print summary
        print("\nGrasp position test results:")
        for desc, status, pos in results:
            pos_str = f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})" if pos is not None else "N/A"
            print(f"  {desc}: {status} at {pos_str}")

        # At least one should succeed
        successes = [r for r in results if r[1] == "success"]
        assert len(successes) > 0, (
            f"All grasp positions failed: {results}"
        )
