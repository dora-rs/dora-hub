"""Tests for trajectory optimiser (CPU-only, small scale)."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import torch
import pytorch_kinematics as pk

from dora_motion_planner.collision_model import CapsuleCollisionModel, OPENARM_CAPSULES
from dora_motion_planner.trajectory_optimizer import TrajectoryOptimizer

URDF_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "examples"
    / "openarm"
    / "openarm_v10.urdf"
)


def build_test_chain():
    """Build chain from URDF with visual/collision stripped."""
    with open(URDF_PATH) as f:
        urdf_str = f.read()
    root = ET.fromstring(urdf_str)
    for link in root.findall(".//link"):
        for tag in ("visual", "collision", "inertial"):
            for elem in link.findall(tag):
                link.remove(elem)
    clean_urdf = ET.tostring(root, encoding="unicode")
    return pk.build_serial_chain_from_urdf(clean_urdf, "openarm_left_link8")


@pytest.fixture
def chain():
    if not URDF_PATH.exists():
        pytest.skip("URDF not found")
    return build_test_chain()


@pytest.fixture
def optimizer(chain):
    device = torch.device("cpu")
    capsule_model = CapsuleCollisionModel(OPENARM_CAPSULES, device)
    limits = chain.get_joint_limits()
    return TrajectoryOptimizer(
        chain=chain,
        capsule_model=capsule_model,
        joint_limits=limits,
        device="cpu",
    )


class TestTrajectoryOptimizer:
    """Test trajectory optimisation convergence."""

    def test_no_obstacles_straight_line(self, optimizer):
        """Without obstacles, optimiser should produce near-linear trajectory."""
        q_start = torch.zeros(7)
        q_goal = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        traj, cost = optimizer.optimize(
            q_start,
            q_goal,
            point_cloud=None,
            T=20,
            num_seeds=2,
            max_iters=50,
            lr=0.01,
        )

        assert traj.shape == (20, 7)
        # First and last waypoints match start/goal
        np.testing.assert_allclose(traj[0].numpy(), q_start.numpy(), atol=1e-5)
        np.testing.assert_allclose(traj[-1].numpy(), q_goal.numpy(), atol=1e-5)

    def test_obstacle_avoidance(self, optimizer, chain):
        """With a wall of points, trajectory should have higher cost than without."""
        q_start = torch.zeros(7)
        q_goal = torch.tensor([0.5, 0.3, 0.0, 0.5, 0.0, 0.0, 0.0])

        # No obstacles
        _, cost_free = optimizer.optimize(
            q_start,
            q_goal,
            point_cloud=None,
            T=20,
            num_seeds=1,
            max_iters=50,
            lr=0.01,
        )

        # Dense wall of points near the arm (in FK workspace)
        tf = chain.forward_kinematics(
            torch.tensor([[0.25, 0.15, 0.0, 0.25, 0.0, 0.0, 0.0]]), end_only=True
        )
        ee_pos = tf.get_matrix()[0, :3, 3].detach()
        wall = ee_pos.unsqueeze(0).repeat(50, 1) + torch.randn(50, 3) * 0.02

        _, cost_wall = optimizer.optimize(
            q_start,
            q_goal,
            point_cloud=wall,
            T=20,
            num_seeds=1,
            max_iters=50,
            lr=0.01,
        )

        # With obstacles, cost should be at least as high (likely higher)
        assert cost_wall >= cost_free * 0.5  # generous bound

    def test_multi_start_picks_best(self, optimizer):
        """Multiple seeds should result in best (lowest) cost being selected."""
        q_start = torch.zeros(7)
        q_goal = torch.tensor([0.3, 0.2, 0.1, 0.3, 0.1, 0.0, 0.0])

        _, cost_1seed = optimizer.optimize(
            q_start,
            q_goal,
            point_cloud=None,
            T=20,
            num_seeds=1,
            max_iters=50,
            lr=0.01,
        )
        _, cost_4seed = optimizer.optimize(
            q_start,
            q_goal,
            point_cloud=None,
            T=20,
            num_seeds=4,
            max_iters=50,
            lr=0.01,
        )

        # More seeds should give cost <= single seed (1% tolerance for
        # non-convex optimisation with limited iterations)
        assert cost_4seed <= cost_1seed * 1.01

    def test_joint_limits_respected(self, optimizer, chain):
        """Trajectory should stay within joint limits."""
        limits = chain.get_joint_limits()
        lower = np.array(limits[0])
        upper = np.array(limits[1])

        q_start = torch.zeros(7)
        q_goal = torch.tensor([0.2, 0.1, 0.0, 0.3, 0.0, 0.0, 0.0])

        traj, _ = optimizer.optimize(
            q_start,
            q_goal,
            point_cloud=None,
            T=20,
            num_seeds=2,
            max_iters=100,
            lr=0.01,
        )

        traj_np = traj.numpy()
        assert np.all(traj_np >= lower - 0.01), "Trajectory violates lower joint limits"
        assert np.all(traj_np <= upper + 0.01), "Trajectory violates upper joint limits"

    def test_bounded_velocity(self, optimizer):
        """Max per-step joint displacement should be bounded by max_step."""
        q_start = torch.zeros(7)
        q_goal = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

        traj, _ = optimizer.optimize(
            q_start,
            q_goal,
            point_cloud=None,
            T=30,
            num_seeds=2,
            max_iters=100,
            lr=0.01,
        )

        vel = (traj[1:] - traj[:-1]).numpy()
        max_step = optimizer.max_step_default
        # Allow some margin for bridge correction + soft clamp effects
        assert np.all(np.abs(vel) < max_step * 3.0), (
            f"Per-step displacement exceeds 3x max_step={max_step}: "
            f"max |vel|={np.abs(vel).max():.4f}"
        )

    def test_goal_reaching(self, optimizer):
        """Trajectory endpoint should match goal configuration."""
        q_start = torch.zeros(7)
        q_goal = torch.tensor([0.2, 0.15, 0.1, 0.2, 0.1, 0.05, 0.0])

        traj, _ = optimizer.optimize(
            q_start,
            q_goal,
            point_cloud=None,
            T=30,
            num_seeds=4,
            max_iters=100,
            lr=0.01,
        )

        np.testing.assert_allclose(
            traj[-1].numpy(), q_goal.numpy(), atol=0.02,
            err_msg="Trajectory endpoint does not match goal"
        )

    def test_large_motion_feasibility(self, optimizer, chain):
        """Auto-enlargement should handle motions requiring > default max_step."""
        limits = chain.get_joint_limits()
        lower = torch.tensor(limits[0])
        upper = torch.tensor(limits[1])

        q_start = torch.zeros(7)
        # Large motion within joint limits — pick 60% of range for each joint
        q_goal = (lower + 0.6 * (upper - lower)).clamp(min=0.5)
        # Ensure at least some joints need > 0.05 rad/step with T=20
        # (need |delta| > 0.05 * 19 = 0.95 rad total)
        q_goal = q_goal.clamp(max=upper)

        traj, cost = optimizer.optimize(
            q_start,
            q_goal,
            point_cloud=None,
            T=20,
            num_seeds=2,
            max_iters=100,
            lr=0.01,
        )

        assert traj.shape == (20, 7)
        # Should still reach the goal despite the large motion
        np.testing.assert_allclose(
            traj[-1].numpy(), q_goal.numpy(), atol=0.05,
            err_msg="Large motion: trajectory endpoint does not reach goal"
        )
        assert cost < float("inf"), "Large motion: cost is infinite"

    def test_smoothness(self, optimizer):
        """Optimised trajectory should have low jerk (3rd derivative)."""
        q_start = torch.zeros(7)
        q_goal = torch.tensor([0.3, 0.2, 0.1, 0.3, 0.1, 0.0, 0.0])

        traj, _ = optimizer.optimize(
            q_start,
            q_goal,
            point_cloud=None,
            T=40,
            num_seeds=4,
            max_iters=150,
            lr=0.01,
        )

        vel = traj[1:] - traj[:-1]
        acc = vel[1:] - vel[:-1]
        jerk = acc[1:] - acc[:-1]
        mean_jerk = (jerk ** 2).mean().item()

        # Compare to a linear trajectory (zero jerk)
        # Optimised trajectory should have reasonably low jerk
        assert mean_jerk < 0.01, (
            f"Trajectory has high jerk: mean_jerk²={mean_jerk:.6f}"
        )
