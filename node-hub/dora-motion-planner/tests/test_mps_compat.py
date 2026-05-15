"""Test MPS (Apple Metal) compatibility for motion planner operations.

Run with: pytest tests/test_mps_compat.py -v
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import torch

mps_available = (
    hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
)
pytestmark = pytest.mark.skipif(not mps_available, reason="MPS not available")

URDF_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "examples"
    / "openarm"
    / "openarm_v10.urdf"
)


@pytest.fixture
def device():
    return torch.device("mps")


class TestMPSPrimitives:
    """Test individual PyTorch ops used by the motion planner on MPS."""

    def test_grid_sample_5d_not_supported(self, device):
        """grid_sample 5D is NOT supported on MPS — we use manual trilinear instead."""
        grid = torch.randn(1, 1, 16, 16, 16, device=device)
        pts = torch.rand(1, 1, 1, 100, 3, device=device) * 2 - 1
        with pytest.raises(NotImplementedError):
            torch.nn.functional.grid_sample(
                grid, pts, mode="bilinear", padding_mode="border", align_corners=True
            )

    def test_softplus(self, device):
        x = torch.randn(64, device=device)
        out = torch.nn.functional.softplus(x, beta=10.0)
        assert torch.isfinite(out).all()

    def test_tanh_atanh(self, device):
        x = torch.rand(64, device=device) * 1.8 - 0.9  # in (-0.9, 0.9)
        t = torch.tanh(x)
        at = torch.atanh(t)
        torch.testing.assert_close(at, x, atol=1e-5, rtol=1e-5)

    def test_einsum(self, device):
        R = torch.randn(8, 3, 3, device=device)
        p = torch.randn(3, device=device)
        out = torch.einsum("bij,j->bi", R, p)
        assert out.shape == (8, 3)

    def test_adam_optimizer(self, device):
        param = torch.randn(8, 7, device=device, requires_grad=True)
        opt = torch.optim.Adam([param], lr=0.01)
        loss = (param**2).sum()
        loss.backward()
        opt.step()
        assert torch.isfinite(param).all()

    def test_linspace_arange(self, device):
        ls = torch.linspace(0, 1, 50, device=device)
        ar = torch.arange(10, device=device)
        assert ls.device.type == "mps"
        assert ar.device.type == "mps"


class TestMPSPytorchKinematics:
    """Test pytorch_kinematics chain on MPS."""

    @pytest.fixture
    def chain(self):
        if not URDF_PATH.exists():
            pytest.skip("URDF not found")
        import pytorch_kinematics as pk

        with open(URDF_PATH) as f:
            urdf_str = f.read()
        root = ET.fromstring(urdf_str)
        for link in root.findall(".//link"):
            for tag in ("visual", "collision", "inertial"):
                for elem in link.findall(tag):
                    link.remove(elem)
        clean_urdf = ET.tostring(root, encoding="unicode")
        return pk.build_serial_chain_from_urdf(clean_urdf, "openarm_left_link8")

    def test_chain_to_mps(self, chain, device):
        """Can move chain to MPS device."""
        mps_chain = chain.to(dtype=torch.float32, device="mps")
        assert mps_chain is not None

    def test_forward_kinematics_single(self, chain, device):
        """FK with single configuration on MPS."""
        mps_chain = chain.to(dtype=torch.float32, device="mps")
        q = torch.zeros(1, 7, device=device)
        tf = mps_chain.forward_kinematics(q, end_only=True)
        mat = tf.get_matrix()
        assert mat.shape == (1, 4, 4)
        assert torch.isfinite(mat).all()

    def test_forward_kinematics_batched(self, chain, device):
        """Batched FK (S*T configs) on MPS — this is the hot path."""
        mps_chain = chain.to(dtype=torch.float32, device="mps")
        batch = 64  # 8 seeds * 8 waypoints
        q = torch.randn(batch, 7, device=device) * 0.3
        tf = mps_chain.forward_kinematics(q, end_only=False)
        # Should return transforms for all links
        assert isinstance(tf, dict)


class TestMPSCollisionModel:
    """Test collision model components on MPS."""

    def test_voxel_sdf_on_mps(self, device):
        """Build VoxelSDF and query on MPS."""
        from dora_motion_planner.collision_model import VoxelSDF

        cloud = torch.randn(200, 3, device=device) * 0.3
        sdf = VoxelSDF(cloud)
        query = torch.randn(50, 3, device=device) * 0.2
        dists = sdf.query(query)
        assert dists.shape == (50,)
        assert torch.isfinite(dists).all()

    def test_capsule_model_on_mps(self, device):
        """CapsuleCollisionModel on MPS."""
        from dora_motion_planner.collision_model import (
            CapsuleCollisionModel,
            OPENARM_CAPSULES,
        )

        model = CapsuleCollisionModel(OPENARM_CAPSULES, device)
        assert model.device == device


class TestMPSFullOptimizer:
    """End-to-end optimizer test on MPS."""

    @pytest.fixture
    def chain(self):
        if not URDF_PATH.exists():
            pytest.skip("URDF not found")
        import pytorch_kinematics as pk

        with open(URDF_PATH) as f:
            urdf_str = f.read()
        root = ET.fromstring(urdf_str)
        for link in root.findall(".//link"):
            for tag in ("visual", "collision", "inertial"):
                for elem in link.findall(tag):
                    link.remove(elem)
        clean_urdf = ET.tostring(root, encoding="unicode")
        return pk.build_serial_chain_from_urdf(clean_urdf, "openarm_left_link8")

    def test_optimize_no_obstacles(self, chain):
        """Full optimization loop on MPS without obstacles."""
        from dora_motion_planner.collision_model import (
            CapsuleCollisionModel,
            OPENARM_CAPSULES,
        )
        from dora_motion_planner.trajectory_optimizer import TrajectoryOptimizer

        device = torch.device("mps")
        capsule_model = CapsuleCollisionModel(OPENARM_CAPSULES, device)
        limits = chain.get_joint_limits()
        optimizer = TrajectoryOptimizer(
            chain=chain,
            capsule_model=capsule_model,
            joint_limits=limits,
            device="mps",
        )

        q_start = torch.zeros(7)
        q_goal = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        traj, cost = optimizer.optimize(
            q_start, q_goal, point_cloud=None, T=20, num_seeds=2, max_iters=50, lr=0.01
        )
        assert traj.shape == (20, 7)
        assert cost < float("inf")

    def test_optimize_with_point_cloud(self, chain):
        """Full optimization with SDF obstacle on MPS."""
        from dora_motion_planner.collision_model import (
            CapsuleCollisionModel,
            OPENARM_CAPSULES,
        )
        from dora_motion_planner.trajectory_optimizer import TrajectoryOptimizer

        device = torch.device("mps")
        capsule_model = CapsuleCollisionModel(OPENARM_CAPSULES, device)
        limits = chain.get_joint_limits()
        optimizer = TrajectoryOptimizer(
            chain=chain,
            capsule_model=capsule_model,
            joint_limits=limits,
            device="mps",
        )

        q_start = torch.zeros(7)
        q_goal = torch.tensor([0.3, 0.2, 0.0, 0.3, 0.0, 0.0, 0.0])
        # Some obstacle points
        cloud = torch.randn(100, 3) * 0.1 + torch.tensor([0.2, 0.1, 0.3])

        traj, cost = optimizer.optimize(
            q_start, q_goal, point_cloud=cloud, T=20, num_seeds=2, max_iters=50, lr=0.01
        )
        assert traj.shape == (20, 7)
        assert cost < float("inf")
