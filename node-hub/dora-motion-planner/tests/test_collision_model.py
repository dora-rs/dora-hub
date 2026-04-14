"""Tests for the capsule and box collision models."""

import torch
import pytest
from dora_motion_planner.collision_model import (
    capsule_points_distance,
    capsule_capsule_distance,
    box_points_distance,
    Box,
    CapsuleCollisionModel,
    OPENARM_CAPSULES,
    OPENARM_GRIPPER_BOXES,
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
        assert len(model.link_names) == 9  # link0-link8 (includes wrist adapter)

    def test_all_radii_positive(self, device):
        """All capsule radii should be positive."""
        model = CapsuleCollisionModel(OPENARM_CAPSULES, device)
        for name in model.link_names:
            assert model.radii[name] > 0

    def test_model_with_boxes(self, device):
        """Model initialises with capsules and boxes (palm + 2 fingers)."""
        model = CapsuleCollisionModel(OPENARM_CAPSULES, device, boxes=OPENARM_GRIPPER_BOXES)
        assert len(model.link_names) == 9  # link0-link8 (includes wrist adapter)
        assert len(model.box_link_names) == 3  # palm + left finger + right finger
        assert "openarm_left_hand" in model.box_link_names
        # Finger boxes use hand link as parent for FK transform lookup
        assert model.box_parent_link["openarm_left_left_finger"] == "openarm_left_hand"
        assert model.box_parent_link["openarm_left_right_finger"] == "openarm_left_hand"


class TestBoxPointsDistance:
    """Test box SDF (signed distance field) against point cloud."""

    def _make_identity_transforms(self, batch_size=1):
        """Create a mock transforms dict with identity transform."""
        import pytorch_kinematics as pk

        tf = pk.Transform3d(
            pos=torch.zeros(batch_size, 3),
            rot=torch.tensor([[1.0, 0, 0, 0]] * batch_size),  # qw, qx, qy, qz
        )
        return {"test_link": tf}

    def test_point_far_away(self):
        """Point far from box has positive distance."""
        transforms = self._make_identity_transforms()
        center = torch.tensor([0.0, 0.0, 0.0])
        half_extents = torch.tensor([0.1, 0.1, 0.1])
        points = torch.tensor([[5.0, 0.0, 0.0]])

        dist = box_points_distance(transforms, "test_link", center, half_extents, points)
        assert dist.shape == (1, 1)
        assert dist[0, 0].item() > 4.0

    def test_point_inside_box(self):
        """Point inside box has negative distance."""
        transforms = self._make_identity_transforms()
        center = torch.tensor([0.0, 0.0, 0.0])
        half_extents = torch.tensor([0.1, 0.1, 0.1])
        points = torch.tensor([[0.0, 0.0, 0.0]])

        dist = box_points_distance(transforms, "test_link", center, half_extents, points)
        assert dist[0, 0].item() < 0  # negative = inside

    def test_point_on_surface(self):
        """Point exactly on box surface has ~zero distance."""
        transforms = self._make_identity_transforms()
        center = torch.tensor([0.0, 0.0, 0.0])
        half_extents = torch.tensor([0.1, 0.1, 0.1])
        points = torch.tensor([[0.1, 0.0, 0.0]])

        dist = box_points_distance(transforms, "test_link", center, half_extents, points)
        assert abs(dist[0, 0].item()) < 1e-5

    def test_corner_distance(self):
        """Point at a corner axis — distance is to the nearest face, not corner."""
        transforms = self._make_identity_transforms()
        center = torch.tensor([0.0, 0.0, 0.0])
        half_extents = torch.tensor([1.0, 1.0, 1.0])
        # Point outside the corner
        points = torch.tensor([[2.0, 2.0, 2.0]])

        dist = box_points_distance(transforms, "test_link", center, half_extents, points)
        # Corner offset = (1, 1, 1), distance = sqrt(3) ≈ 1.732
        expected = (3.0) ** 0.5
        assert abs(dist[0, 0].item() - expected) < 0.01

    def test_batched_transforms(self):
        """Multiple configurations (batch) against multiple points."""
        B = 4
        transforms = self._make_identity_transforms(batch_size=B)
        center = torch.tensor([0.0, 0.0, 0.0])
        half_extents = torch.tensor([0.1, 0.1, 0.1])
        N = 10
        points = torch.randn(N, 3)

        dist = box_points_distance(transforms, "test_link", center, half_extents, points)
        assert dist.shape == (B, N)

    def test_with_offset_center(self):
        """Box with non-zero center offset in link frame."""
        transforms = self._make_identity_transforms()
        center = torch.tensor([0.0, 0.0, 0.5])  # box center at z=0.5
        half_extents = torch.tensor([0.1, 0.1, 0.1])
        # Point at box center should be inside
        points = torch.tensor([[0.0, 0.0, 0.5]])

        dist = box_points_distance(transforms, "test_link", center, half_extents, points)
        assert dist[0, 0].item() < 0

        # Point at origin should be outside (box is at z=0.4 to z=0.6)
        points_origin = torch.tensor([[0.0, 0.0, 0.0]])
        dist_origin = box_points_distance(
            transforms, "test_link", center, half_extents, points_origin
        )
        assert dist_origin[0, 0].item() > 0

    def test_gradient_flows(self):
        """Verify gradients flow through box SDF."""
        import pytorch_kinematics as pk

        pos = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
        tf = pk.Transform3d(
            pos=pos,
            rot=torch.tensor([[1.0, 0, 0, 0]]),
        )
        transforms = {"test_link": tf}
        center = torch.tensor([0.0, 0.0, 0.0])
        half_extents = torch.tensor([0.1, 0.1, 0.1])
        points = torch.tensor([[0.2, 0.0, 0.0]])

        dist = box_points_distance(transforms, "test_link", center, half_extents, points)
        dist.sum().backward()
        assert pos.grad is not None
