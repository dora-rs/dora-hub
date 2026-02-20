"""Capsule-based collision model for OpenArm links.

Each arm link is approximated as a capsule (line segment + radius) in the
link-local frame.  Differentiable distance functions allow gradient-based
trajectory optimisation via PyTorch autograd.
"""

from dataclasses import dataclass

import torch


@dataclass
class Capsule:
    """A capsule primitive defined by two endpoints and a radius in link-local frame."""

    p0: list[float]
    p1: list[float]
    radius: float


# Capsule definitions for each link in the left-arm serial chain.
# Endpoints are in the link-local frame, derived from the URDF joint origins.
# Key: link name as it appears in the serial chain (pk strips prefix for
# build_serial_chain_from_urdf).
OPENARM_CAPSULES: dict[str, Capsule] = {
    "openarm_left_link0": Capsule([0, 0, 0], [0, 0, 0.0625], 0.04),
    "openarm_left_link1": Capsule([0, 0, 0], [-0.0301, 0, 0.06], 0.04),
    "openarm_left_link2": Capsule([0, 0, 0], [0.0301, 0, 0.06625], 0.03),
    "openarm_left_link3": Capsule([0, 0, 0], [0, 0.0315, 0.15375], 0.04),
    "openarm_left_link4": Capsule([0, 0, 0], [0, -0.0315, 0.0955], 0.04),
    "openarm_left_link5": Capsule([0, 0, 0], [0.0375, 0, 0.1205], 0.03),
    "openarm_left_link6": Capsule([0, 0, 0], [-0.0375, 0, 0], 0.025),
    "openarm_left_link7": Capsule([0, 0, 0], [0, 0.0205, 0], 0.025),
}

# Non-adjacent link pairs for self-collision checking.
# Skip (i, i+1) because adjacent links are connected by joints and always
# "collide" at the joint origin.
LINK_NAMES = list(OPENARM_CAPSULES.keys())
SELF_COLLISION_PAIRS: list[tuple[int, int]] = [
    (i, j) for i in range(len(LINK_NAMES)) for j in range(i + 2, len(LINK_NAMES))
]

# Right-arm capsules — identical geometry in link-local frames, different link names.
OPENARM_RIGHT_CAPSULES: dict[str, Capsule] = {
    "openarm_right_link0": Capsule([0, 0, 0], [0, 0, 0.0625], 0.04),
    "openarm_right_link1": Capsule([0, 0, 0], [-0.0301, 0, 0.06], 0.04),
    "openarm_right_link2": Capsule([0, 0, 0], [0.0301, 0, 0.06625], 0.03),
    "openarm_right_link3": Capsule([0, 0, 0], [0, 0.0315, 0.15375], 0.04),
    "openarm_right_link4": Capsule([0, 0, 0], [0, -0.0315, 0.0955], 0.04),
    "openarm_right_link5": Capsule([0, 0, 0], [0.0375, 0, 0.1205], 0.03),
    "openarm_right_link6": Capsule([0, 0, 0], [-0.0375, 0, 0], 0.025),
    "openarm_right_link7": Capsule([0, 0, 0], [0, 0.0205, 0], 0.025),
}

RIGHT_LINK_NAMES = list(OPENARM_RIGHT_CAPSULES.keys())
SELF_COLLISION_PAIRS_RIGHT: list[tuple[int, int]] = [
    (i, j)
    for i in range(len(RIGHT_LINK_NAMES))
    for j in range(i + 2, len(RIGHT_LINK_NAMES))
]


class CapsuleCollisionModel:
    """Differentiable collision model using capsule approximations."""

    def __init__(self, capsules: dict[str, Capsule], device: torch.device):
        self.device = device
        self.link_names = list(capsules.keys())
        # Pre-compute local capsule endpoints as tensors
        self.local_p0 = {}
        self.local_p1 = {}
        self.radii = {}
        for name, cap in capsules.items():
            self.local_p0[name] = torch.tensor(
                cap.p0, dtype=torch.float32, device=device
            )
            self.local_p1[name] = torch.tensor(
                cap.p1, dtype=torch.float32, device=device
            )
            self.radii[name] = cap.radius

    def capsule_endpoints_world(
        self, link_name: str, transforms: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform local capsule endpoints to world frame using FK transforms.

        Args:
            link_name: Name of the link.
            transforms: Dict of link_name -> Transform3d from FK (batched).

        Returns:
            (p0_world, p1_world) each of shape (B, 3).
        """
        tf = transforms[link_name]
        mat = tf.get_matrix()  # (B, 4, 4)
        R = mat[:, :3, :3]  # (B, 3, 3)
        t = mat[:, :3, 3]  # (B, 3)
        p0_w = torch.einsum("bij,j->bi", R, self.local_p0[link_name]) + t
        p1_w = torch.einsum("bij,j->bi", R, self.local_p1[link_name]) + t
        return p0_w, p1_w


def closest_point_on_segment(
    a: torch.Tensor, b: torch.Tensor, p: torch.Tensor
) -> torch.Tensor:
    """Find closest point on segment [a, b] to point(s) p.

    Args:
        a: (B, 3) or (3,) segment start
        b: (B, 3) or (3,) segment end
        p: (N, 3) query points

    Returns:
        Closest points, shape (B, N, 3) or (N, 3).
    """
    ab = b - a  # (B, 3) or (3,)
    if ab.ndim == 1:
        # Unbatched segment, batched points
        ap = p - a  # (N, 3)
        t = (ap @ ab) / (ab @ ab + 1e-8)  # (N,)
        t = t.clamp(0.0, 1.0)
        return a + t.unsqueeze(-1) * ab  # (N, 3)
    else:
        # Batched segments (B, 3), points (N, 3) -> (B, N, 3)
        # a: (B, 3), b: (B, 3), ab: (B, 3), p: (N, 3)
        ap = p.unsqueeze(0) - a.unsqueeze(1)  # (B, N, 3)
        ab_unsq = ab.unsqueeze(1)  # (B, 1, 3)
        t = (ap * ab_unsq).sum(-1) / ((ab * ab).sum(-1, keepdim=True) + 1e-8)  # (B, N)
        t = t.clamp(0.0, 1.0)
        return a.unsqueeze(1) + t.unsqueeze(-1) * ab_unsq  # (B, N, 3)


def capsule_points_distance(
    cap_p0: torch.Tensor,
    cap_p1: torch.Tensor,
    radius: float,
    points: torch.Tensor,
) -> torch.Tensor:
    """Signed distance from capsule surface to point cloud (negative = penetration).

    Args:
        cap_p0: (B, 3) capsule segment start in world frame.
        cap_p1: (B, 3) capsule segment end in world frame.
        radius: Capsule radius.
        points: (N, 3) point cloud in world frame.

    Returns:
        (B, N) signed distances (positive = no collision).
    """
    # closest point on segment to each query point
    ab = cap_p1 - cap_p0  # (B, 3)
    ap = points.unsqueeze(0) - cap_p0.unsqueeze(1)  # (B, N, 3)
    ab_unsq = ab.unsqueeze(1)  # (B, 1, 3)
    t = (ap * ab_unsq).sum(-1) / ((ab * ab).sum(-1, keepdim=True) + 1e-8)  # (B, N)
    t = t.clamp(0.0, 1.0)
    closest = cap_p0.unsqueeze(1) + t.unsqueeze(-1) * ab_unsq  # (B, N, 3)
    dist = (points.unsqueeze(0) - closest).norm(dim=-1)  # (B, N)
    return dist - radius


def capsule_capsule_distance(
    a_p0: torch.Tensor,
    a_p1: torch.Tensor,
    a_radius: float,
    b_p0: torch.Tensor,
    b_p1: torch.Tensor,
    b_radius: float,
) -> torch.Tensor:
    """Minimum signed distance between two capsules (negative = penetration).

    Uses sampled points along each segment for a differentiable approximation.

    Args:
        a_p0, a_p1: (B, 3) endpoints of capsule A.
        b_p0, b_p1: (B, 3) endpoints of capsule B.
        a_radius, b_radius: capsule radii.

    Returns:
        (B,) signed distances.
    """
    # Sample points along each capsule axis
    n_samples = 8
    ts = torch.linspace(0, 1, n_samples, device=a_p0.device)  # (S,)

    # Points along capsule A: (B, S, 3)
    pts_a = a_p0.unsqueeze(1) + ts.view(1, -1, 1) * (a_p1 - a_p0).unsqueeze(1)
    # Points along capsule B: (B, S, 3)
    pts_b = b_p0.unsqueeze(1) + ts.view(1, -1, 1) * (b_p1 - b_p0).unsqueeze(1)

    # Pairwise distances: (B, S_a, S_b)
    diff = pts_a.unsqueeze(2) - pts_b.unsqueeze(1)  # (B, S, S, 3)
    dists = diff.norm(dim=-1)  # (B, S, S)

    # Minimum distance between the two line segments (approximate)
    min_dist = dists.min(dim=-1).values.min(dim=-1).values  # (B,)
    return min_dist - a_radius - b_radius
