"""Tests for the capsule collision model."""

import torch
import pytest
from dora_motion_planner.collision_model import (
    capsule_points_distance,
    capsule_capsule_distance,
    CapsuleCollisionModel,
    OPENARM_CAPSULES,
)


@pytest.fixture
def device():
    return torch.device("cpu")


class TestCapsulePointsDistance:
    """Test capsule-to-point-cloud signed distance."""

    def test_point_far_away(self):
        """Point far from capsule has positive distance."""
        cap_p0 = torch.tensor([[0.0, 0.0, 0.0]])
        cap_p1 = torch.tensor([[0.0, 0.0, 1.0]])
        radius = 0.05
        points = torch.tensor([[5.0, 0.0, 0.5]])

        dist = capsule_points_distance(cap_p0, cap_p1, radius, points)
        assert dist.shape == (1, 1)
        assert dist[0, 0].item() > 0  # positive = no collision

    def test_point_inside_capsule(self):
        """Point inside capsule has negative distance."""
        cap_p0 = torch.tensor([[0.0, 0.0, 0.0]])
        cap_p1 = torch.tensor([[0.0, 0.0, 1.0]])
        radius = 0.1
        points = torch.tensor([[0.01, 0.0, 0.5]])

        dist = capsule_points_distance(cap_p0, cap_p1, radius, points)
        assert dist[0, 0].item() < 0  # negative = penetrating

    def test_batched_capsule(self):
        """Multiple capsule configurations (batch) against multiple points."""
        B = 3
        cap_p0 = torch.zeros(B, 3)
        cap_p1 = torch.tensor([[0, 0, 1.0]] * B)
        radius = 0.05
        N = 10
        points = torch.randn(N, 3) * 5  # random far-away points

        dist = capsule_points_distance(cap_p0, cap_p1, radius, points)
        assert dist.shape == (B, N)

    def test_gradient_flows(self):
        """Verify gradients flow through the distance computation."""
        cap_p0 = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
        cap_p1 = torch.tensor([[0.0, 0.0, 1.0]], requires_grad=True)
        radius = 0.05
        points = torch.tensor([[0.1, 0.0, 0.5]])

        dist = capsule_points_distance(cap_p0, cap_p1, radius, points)
        loss = dist.sum()
        loss.backward()
        assert cap_p0.grad is not None
        assert cap_p1.grad is not None


class TestCapsuleCapsuleDistance:
    """Test capsule-to-capsule signed distance."""

    def test_parallel_capsules_far(self):
        """Two parallel capsules far apart."""
        a_p0 = torch.tensor([[0.0, 0.0, 0.0]])
        a_p1 = torch.tensor([[0.0, 0.0, 1.0]])
        b_p0 = torch.tensor([[5.0, 0.0, 0.0]])
        b_p1 = torch.tensor([[5.0, 0.0, 1.0]])

        dist = capsule_capsule_distance(a_p0, a_p1, 0.05, b_p0, b_p1, 0.05)
        assert dist[0].item() > 4.0  # ~4.9

    def test_overlapping_capsules(self):
        """Two capsules that overlap."""
        a_p0 = torch.tensor([[0.0, 0.0, 0.0]])
        a_p1 = torch.tensor([[0.0, 0.0, 1.0]])
        b_p0 = torch.tensor([[0.01, 0.0, 0.0]])
        b_p1 = torch.tensor([[0.01, 0.0, 1.0]])

        dist = capsule_capsule_distance(a_p0, a_p1, 0.05, b_p0, b_p1, 0.05)
        assert dist[0].item() < 0  # penetrating

    def test_gradient_flows(self):
        """Verify gradients flow through capsule-capsule distance."""
        a_p0 = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
        a_p1 = torch.tensor([[0.0, 0.0, 1.0]], requires_grad=True)
        b_p0 = torch.tensor([[0.5, 0.0, 0.0]])
        b_p1 = torch.tensor([[0.5, 0.0, 1.0]])

        dist = capsule_capsule_distance(a_p0, a_p1, 0.05, b_p0, b_p1, 0.05)
        dist.sum().backward()
        assert a_p0.grad is not None


class TestCapsuleCollisionModel:
    """Test the collision model with URDF-derived capsules."""

    def test_model_init(self, device):
        """Model initialises with all OpenArm links."""
        model = CapsuleCollisionModel(OPENARM_CAPSULES, device)
        assert len(model.link_names) == 8

    def test_all_radii_positive(self, device):
        """All capsule radii should be positive."""
        model = CapsuleCollisionModel(OPENARM_CAPSULES, device)
        for name in model.link_names:
            assert model.radii[name] > 0
