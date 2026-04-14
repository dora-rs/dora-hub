"""Collision primitives for OpenArm links.

Each arm link is approximated as a capsule (line segment + radius) or box
(oriented bounding box) in the link-local frame.  Differentiable distance
functions allow gradient-based trajectory optimisation via PyTorch autograd.

Includes a voxel SDF (Signed Distance Field) built from a point cloud via
scipy's Euclidean distance transform.  Query via differentiable trilinear
interpolation — O(1) per sample point, independent of point cloud size.
"""

import time
from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


@dataclass
class Capsule:
    """A capsule primitive defined by two endpoints and a radius in link-local frame."""

    p0: list[float]
    p1: list[float]
    radius: float


@dataclass
class Box:
    """An oriented bounding box defined by center and half-extents in a link-local frame.

    Args:
        center: Box center in the parent link's local frame.
        half_extents: Box half-extents along each axis.
        parent_link: FK link name whose transform positions this box.
            If None, the dict key is used as the link name (backward compat).
    """

    center: list[float]
    half_extents: list[float]
    parent_link: str | None = None


# Capsule definitions for each link in the left-arm serial chain.
# Endpoints are in the link-local frame, derived from the URDF joint origins.
# Key: link name as it appears in the serial chain (pk strips prefix for
# build_serial_chain_from_urdf).
OPENARM_CAPSULES: dict[str, Capsule] = {
    "openarm_left_link0": Capsule([0, 0, 0], [0, 0, 0.0625], 0.04),
    "openarm_left_link1": Capsule([0, 0, 0], [-0.0301, 0, 0.06], 0.04),
    "openarm_left_link2": Capsule([0, 0, 0], [0.0301, 0, 0.11625], 0.03),
    "openarm_left_link3": Capsule([0, 0, 0], [0, 0.0315, 0.15375], 0.04),
    "openarm_left_link4": Capsule([0, 0, 0], [0, -0.0315, 0.0955], 0.04),
    "openarm_left_link5": Capsule([0, 0, 0], [0.0375, 0, 0.1205], 0.03),
    "openarm_left_link6": Capsule([0, 0, 0], [-0.0375, 0, 0], 0.025),
    "openarm_left_link7": Capsule([0, 0, 0], [0, 0.0205, 0], 0.025),
    # Wrist adapter: link8 → hand joint (100mm segment, URDF: xyz="0 -0.025 0.1001")
    "openarm_left_link8": Capsule([0, 0, 0], [0, -0.025, 0.1001], 0.025),
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
    "openarm_right_link2": Capsule([0, 0, 0], [0.0301, 0, 0.11625], 0.03),
    "openarm_right_link3": Capsule([0, 0, 0], [0, 0.0315, 0.15375], 0.04),
    "openarm_right_link4": Capsule([0, 0, 0], [0, -0.0315, 0.0955], 0.04),
    "openarm_right_link5": Capsule([0, 0, 0], [0.0375, 0, 0.1205], 0.03),
    "openarm_right_link6": Capsule([0, 0, 0], [-0.0375, 0, 0], 0.025),
    "openarm_right_link7": Capsule([0, 0, 0], [0, 0.0205, 0], 0.025),
    # Wrist adapter: link8 → hand joint (100mm segment)
    "openarm_right_link8": Capsule([0, 0, 0], [0, -0.025, 0.1001], 0.025),
}

RIGHT_LINK_NAMES = list(OPENARM_RIGHT_CAPSULES.keys())
SELF_COLLISION_PAIRS_RIGHT: list[tuple[int, int]] = [
    (i, j)
    for i in range(len(RIGHT_LINK_NAMES))
    for j in range(i + 2, len(RIGHT_LINK_NAMES))
]

# Gripper box primitives — oriented bounding boxes in link-local frame.
# Modeled as parallel jaw gripper: palm body + two separate finger boxes,
# leaving a gap between the fingers for grasped objects.
#
# From URDF (left hand):
#   - Hand body: origin at link8 + (0, -0.025, 0.1001), TCP at hand + (0, 0, 0.08)
#   - Left finger:  prismatic joint at hand + (0, 0.012, 0.015), axis +Y, travel 0..0.044m
#   - Right finger: prismatic joint at hand + (0, 0.000, 0.015), axis -Y, travel 0..0.044m
#   - Fingers extend from z=0.015 to z=0.08 in hand frame (65mm, matching TCP height)
#
# We model fingers at fully-open position (max travel = 0.044m) so the
# collision envelope is conservative.  The gap between fingers (~24mm at
# closed, ~112mm at open) is intentionally left empty.
OPENARM_GRIPPER_BOXES: dict[str, Box] = {
    # Palm body (between link8 joint and finger base)
    "openarm_left_hand": Box(center=[0, -0.006, 0.007], half_extents=[0.025, 0.025, 0.015]),
    # Left finger (fully open: Y = 0.012 + 0.044 = 0.056 from hand origin)
    # Z range: 0.015 (finger joint) to 0.08 (TCP height) → center 0.0475, half 0.0325
    # Coords are in hand-link local frame; parent_link tells FK where to look.
    "openarm_left_left_finger": Box(
        center=[0, 0.049, 0.0475], half_extents=[0.008, 0.008, 0.0325],
        parent_link="openarm_left_hand",
    ),
    # Right finger (fully open: Y = 0.0 - 0.044 = -0.044 from hand origin)
    "openarm_left_right_finger": Box(
        center=[0, -0.037, 0.0475], half_extents=[0.008, 0.008, 0.0325],
        parent_link="openarm_left_hand",
    ),
}

# Static body capsule — the vertical support pole in the world frame.
# Extends from the base (z=0) to the arm attachment point (z≈0.7m).
# Radius ~60mm covers the main structural beam.
OPENARM_BODY_CAPSULE = Capsule([0, 0, 0], [0, 0, 0.70], 0.10)

OPENARM_RIGHT_GRIPPER_BOXES: dict[str, Box] = {
    # Palm body
    "openarm_right_hand": Box(center=[0, -0.006, 0.007], half_extents=[0.025, 0.025, 0.015]),
    # Left finger (fully open)
    "openarm_right_left_finger": Box(
        center=[0, 0.049, 0.0475], half_extents=[0.008, 0.008, 0.0325],
        parent_link="openarm_right_hand",
    ),
    # Right finger (fully open)
    "openarm_right_right_finger": Box(
        center=[0, -0.037, 0.0475], half_extents=[0.008, 0.008, 0.0325],
        parent_link="openarm_right_hand",
    ),
}


class CapsuleCollisionModel:
    """Differentiable collision model using capsule and box approximations."""

    def __init__(
        self,
        capsules: dict[str, Capsule],
        device: torch.device,
        boxes: dict[str, Box] | None = None,
        body_capsule: Capsule | None = None,
        body_skip_links: int = 3,
    ):
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

        # Static body capsule (world frame) — e.g. the robot support pole.
        # Checked against arm links during self-collision, skipping the first
        # body_skip_links links (they're mounted on the body and always overlap).
        self.body_capsule = body_capsule
        self.body_skip_links = body_skip_links
        if body_capsule is not None:
            self.body_p0 = torch.tensor(
                body_capsule.p0, dtype=torch.float32, device=device
            )
            self.body_p1 = torch.tensor(
                body_capsule.p1, dtype=torch.float32, device=device
            )
            self.body_radius = body_capsule.radius

        # Box primitives (e.g. for gripper fingers).
        # box_parent_link maps box name -> FK link name for transform lookup.
        # When Box.parent_link is set, the box coordinates are in that link's
        # frame even though the box has a different name (e.g. finger boxes
        # defined in the hand frame because the FK chain doesn't include
        # prismatic finger joints).
        self.box_link_names: list[str] = []
        self.box_parent_link: dict[str, str] = {}
        self.box_centers: dict[str, torch.Tensor] = {}
        self.box_half_extents: dict[str, torch.Tensor] = {}
        if boxes:
            for name, box in boxes.items():
                self.box_link_names.append(name)
                self.box_parent_link[name] = box.parent_link or name
                self.box_centers[name] = torch.tensor(
                    box.center, dtype=torch.float32, device=device
                )
                self.box_half_extents[name] = torch.tensor(
                    box.half_extents, dtype=torch.float32, device=device
                )

    def set_gripper_opening(self, opening_m: float) -> None:
        """Shift finger box Y-centers to match a given gripper opening.

        Args:
            opening_m: Total jaw opening in metres (0 = closed, 0.088 = fully open).
                Each finger moves half this distance from its base position.
        """
        # Finger travel per side (URDF prismatic joint range: 0..0.044m)
        travel = min(opening_m / 2.0, 0.044)
        # Base positions (closed): left finger Y=+0.012, right finger Y=0.0
        # URDF axis: left finger moves +Y, right finger moves -Y
        # Box center Z stays the same, only Y shifts
        finger_configs = {
            "left_finger": ("openarm_left_left_finger", +0.012 + travel),
            "right_finger": ("openarm_left_right_finger", 0.0 - travel),
            "left_finger_r": ("openarm_right_left_finger", +0.012 + travel),
            "right_finger_r": ("openarm_right_right_finger", 0.0 - travel),
        }
        for _, (box_name, new_y) in finger_configs.items():
            if box_name in self.box_centers:
                c = self.box_centers[box_name]
                # Box center: [X, Y_finger_center, Z]
                # Y_finger_center = finger_base_y + finger_length/2
                # finger length ~ 0.065m (from Z extents), center offset from joint
                self.box_centers[box_name] = torch.tensor(
                    [c[0].item(), new_y, c[2].item()],
                    dtype=torch.float32, device=self.device,
                )

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


def box_points_distance(
    transforms: dict,
    link_name: str,
    box_center: torch.Tensor,
    box_half_extents: torch.Tensor,
    points: torch.Tensor,
) -> torch.Tensor:
    """Signed distance from an oriented bounding box to a point cloud.

    Uses the standard box SDF: transform points to box-local frame,
    then compute distance from the box surface.  Negative = penetration.

    Args:
        transforms: FK transforms dict (link_name -> Transform3d, batched).
        link_name: Link the box is attached to.
        box_center: (3,) box center in link-local frame.
        box_half_extents: (3,) box half-extents in link-local frame.
        points: (N, 3) point cloud in world frame.

    Returns:
        (B, N) signed distances (positive = no collision).
    """
    tf = transforms[link_name]
    mat = tf.get_matrix()  # (B, 4, 4)
    R = mat[:, :3, :3]  # (B, 3, 3)
    t = mat[:, :3, 3]  # (B, 3)

    # Transform points to link-local frame: p_local = R^T @ (p - t)
    # points: (N, 3), t: (B, 3) -> diff: (B, N, 3)
    diff = points.unsqueeze(0) - t.unsqueeze(1)  # (B, N, 3)
    # R^T @ diff for each batch: (B, 3, 3)^T @ (B, N, 3) -> einsum
    p_local = torch.einsum("bji,bnj->bni", R, diff)  # (B, N, 3)

    # Shift to box-centered frame
    p_box = p_local - box_center  # (B, N, 3), broadcasts (3,)

    # Box SDF: q = |p| - half_extents
    q = p_box.abs() - box_half_extents  # (B, N, 3)
    # Outside distance: length of max(q, 0)
    outside = torch.clamp(q, min=0.0).norm(dim=-1)  # (B, N)
    # Inside distance: min(max(q_x, q_y, q_z), 0)  — negative when inside
    inside = torch.clamp(q.max(dim=-1).values, max=0.0)  # (B, N)

    return outside + inside  # (B, N)


def _point_in_polygon(pts_xy: torch.Tensor, polygon: torch.Tensor) -> torch.Tensor:
    """Ray-casting point-in-polygon test (differentiable-friendly boolean mask).

    Args:
        pts_xy: (..., 2) query points.
        polygon: (V, 2) polygon vertices (ordered, closed automatically).

    Returns:
        (...) boolean mask, True if point is inside the polygon.
    """
    n = polygon.shape[0]
    flat = pts_xy.reshape(-1, 2)  # (M, 2)
    px, py = flat[:, 0], flat[:, 1]  # (M,)

    inside = torch.zeros(flat.shape[0], dtype=torch.bool, device=flat.device)
    for i in range(n):
        j = (i + 1) % n
        xi, yi = polygon[i, 0], polygon[i, 1]
        xj, yj = polygon[j, 0], polygon[j, 1]
        # Edge crosses the horizontal ray from point going +X?
        cond = ((yi > py) != (yj > py)) & (
            px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi
        )
        inside = inside ^ cond

    return inside.reshape(pts_xy.shape[:-1])


def capsule_halfplane_distance(
    cap_p0: torch.Tensor,
    cap_p1: torch.Tensor,
    radius: float,
    plane_z: float,
    plane_bounds: tuple[float, float, float, float] | None = None,
    plane_polygon: torch.Tensor | None = None,
) -> torch.Tensor:
    """Signed distance from capsule to a horizontal half-plane at z=plane_z.

    The half-plane represents a table top: anything below plane_z within the
    XY bounds is considered collision.  Distance is measured from the lowest
    point on the capsule surface (segment + radius) to plane_z.

    Args:
        cap_p0: (B, 3) capsule segment start in world frame.
        cap_p1: (B, 3) capsule segment end in world frame.
        radius: Capsule radius.
        plane_z: Z height of the plane (table top).
        plane_bounds: Optional (x_min, x_max, y_min, y_max) AABB — only penalise
            capsule points within these XY bounds.
        plane_polygon: Optional (V, 2) polygon vertices — if provided, used
            instead of plane_bounds for a rotated table region.

    Returns:
        (B,) signed distances (positive = above plane, negative = below).
    """
    # Sample points along the capsule axis for better coverage
    n_samples = 8
    ts = torch.linspace(0, 1, n_samples, device=cap_p0.device)
    # (B, S, 3)
    pts = cap_p0.unsqueeze(1) + ts.view(1, -1, 1) * (cap_p1 - cap_p0).unsqueeze(1)

    # Z distance from each sample to plane, minus radius
    z_dist = pts[:, :, 2] - plane_z - radius  # (B, S)

    if plane_polygon is not None:
        in_xy = _point_in_polygon(pts[:, :, :2], plane_polygon)  # (B, S)
        z_dist = torch.where(in_xy, z_dist, torch.tensor(1.0, device=z_dist.device))
    elif plane_bounds is not None:
        x_min, x_max, y_min, y_max = plane_bounds
        in_xy = (
            (pts[:, :, 0] >= x_min)
            & (pts[:, :, 0] <= x_max)
            & (pts[:, :, 1] >= y_min)
            & (pts[:, :, 1] <= y_max)
        )
        z_dist = torch.where(in_xy, z_dist, torch.tensor(1.0, device=z_dist.device))

    return z_dist.min(dim=1).values  # (B,)


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


class VoxelSDF:
    """3D distance field built from a point cloud.

    Discretises the point cloud into a voxel grid, then computes the
    Euclidean distance transform so that each voxel stores the distance
    to the nearest occupied voxel.  Query points are looked up via
    differentiable trilinear interpolation on GPU.

    Build time is O(V) via scipy EDT on CPU; query is O(1) per point
    on GPU, independent of the original point cloud size N.
    """

    def __init__(
        self,
        point_cloud: torch.Tensor,
        resolution: float = 0.01,
        padding: float = 0.15,
    ):
        """Build the SDF from a point cloud.

        Args:
            point_cloud: (N, 3) points in robot frame.
            resolution: Voxel edge length in metres (default 1 cm).
            padding: Extra space around the point cloud bounding box.
        """
        t0 = time.perf_counter()
        device = point_cloud.device
        pc_np = point_cloud.detach().cpu().numpy()

        self.origin = pc_np.min(axis=0) - padding
        self.extent = pc_np.max(axis=0) + padding
        grid_shape = np.ceil(
            (self.extent - self.origin) / resolution
        ).astype(int) + 1

        # Binary occupancy grid
        occupancy = np.zeros(grid_shape, dtype=bool)
        idx = ((pc_np - self.origin) / resolution).astype(int)
        idx = np.clip(idx, 0, grid_shape - 1)
        occupancy[idx[:, 0], idx[:, 1], idx[:, 2]] = True

        # EDT: distance from every empty voxel to nearest occupied one
        # (occupied voxels get distance 0).  Result is in voxel units.
        dist = distance_transform_edt(~occupancy).astype(np.float32)
        dist *= resolution  # convert to metres

        # Store as raw 3D tensor (Gx, Gy, Gz) for manual trilinear lookup.
        dist_t = torch.from_numpy(dist).to(device)
        self.grid = dist_t  # (Gx, Gy, Gz)
        self.grid_shape = torch.tensor(
            grid_shape, dtype=torch.float32, device=device
        )  # (3,)

        self.origin_t = torch.tensor(
            self.origin, dtype=torch.float32, device=device
        )
        self.extent_t = torch.tensor(
            self.extent, dtype=torch.float32, device=device
        )

        t_build = time.perf_counter() - t0
        print(
            f"[sdf] built {grid_shape[0]}×{grid_shape[1]}×{grid_shape[2]} "
            f"voxels @ {resolution*100:.0f}cm from {len(pc_np)} pts "
            f"({t_build:.3f}s)"
        )

    def query(self, points: torch.Tensor) -> torch.Tensor:
        """Look up distance-to-nearest-obstacle at arbitrary world points.

        Uses differentiable trilinear interpolation (manual implementation
        that works on CUDA, MPS, and CPU).  Points outside the grid are
        clamped to the border value (conservative).

        Args:
            points: (..., 3) query points in robot frame.

        Returns:
            (...) distance to nearest obstacle surface (metres).
        """
        shape = points.shape[:-1]
        pts = points.reshape(-1, 3)

        # Map world coords to continuous voxel indices.
        voxel = (pts - self.origin_t) / (self.extent_t - self.origin_t) * (
            self.grid_shape - 1
        )  # (M, 3)

        # Clamp to valid grid range (border behaviour).
        max_idx = self.grid_shape - 1
        voxel = voxel.clamp(min=torch.zeros(3, device=pts.device),
                            max=max_idx)

        # Floor / ceil indices for trilinear interpolation.
        v0 = voxel.floor().long()  # (M, 3)
        v1 = (v0 + 1).clamp(max=max_idx.long())
        frac = voxel - v0.float()  # fractional part  (M, 3)
        fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]
        x0, y0, z0 = v0[:, 0], v0[:, 1], v0[:, 2]
        x1, y1, z1 = v1[:, 0], v1[:, 1], v1[:, 2]

        # 8-corner lookup + trilinear blend.
        c000 = self.grid[x0, y0, z0]
        c001 = self.grid[x0, y0, z1]
        c010 = self.grid[x0, y1, z0]
        c011 = self.grid[x0, y1, z1]
        c100 = self.grid[x1, y0, z0]
        c101 = self.grid[x1, y0, z1]
        c110 = self.grid[x1, y1, z0]
        c111 = self.grid[x1, y1, z1]

        out = (
            c000 * (1 - fx) * (1 - fy) * (1 - fz)
            + c001 * (1 - fx) * (1 - fy) * fz
            + c010 * (1 - fx) * fy * (1 - fz)
            + c011 * (1 - fx) * fy * fz
            + c100 * fx * (1 - fy) * (1 - fz)
            + c101 * fx * (1 - fy) * fz
            + c110 * fx * fy * (1 - fz)
            + c111 * fx * fy * fz
        )
        return out.reshape(shape)
