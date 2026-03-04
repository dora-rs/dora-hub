"""Depth map to point cloud conversion for collision checking.

Converts mono16 (uint16 millimetre) depth images from dora-pyrealsense
into downsampled point clouds in the robot base frame.
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def depth_to_pointcloud(
    depth_map: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int = 640,
    height: int = 480,
    stride: int = 8,
    min_depth_mm: int = 100,
    max_depth_mm: int = 2000,
    color_image: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Convert a mono16 depth image to a 3D point cloud in camera frame.

    Args:
        depth_map: Flat uint16 array (H*W) of depth in millimetres.
        fx, fy: Focal lengths in pixels.
        cx, cy: Principal point in pixels.
        width, height: Image dimensions.
        stride: Downsample factor (take every stride-th pixel in each axis).
        min_depth_mm, max_depth_mm: Valid depth range.
        color_image: Optional (H, W, 3) uint8 RGB image.  When provided,
            returns ``(points, colors)`` where *colors* is (N, 3) float32
            in [0, 1] matching each 3D point.

    Returns:
        (N, 3) float32 points, or ``(points, colors)`` when *color_image*
        is provided.
    """
    depth_2d = depth_map.reshape(height, width)

    # Subsample
    depth_sub = depth_2d[::stride, ::stride]
    h_sub, w_sub = depth_sub.shape

    # Pixel grid
    v_indices, u_indices = np.mgrid[:h_sub, :w_sub]
    u_orig = u_indices * stride
    v_orig = v_indices * stride

    # Flatten
    d = depth_sub.ravel().astype(np.float32)
    u = u_orig.ravel().astype(np.float32)
    v = v_orig.ravel().astype(np.float32)

    # Filter valid depths
    valid = (d >= min_depth_mm) & (d <= max_depth_mm)
    d = d[valid]
    u = u[valid]
    v = v[valid]

    if len(d) == 0:
        empty = np.zeros((0, 3), dtype=np.float32)
        return (empty, empty) if color_image is not None else empty

    # Deproject: camera frame (z forward, x right, y down)
    z = d * 0.001  # mm -> m
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack([x, y, z], axis=-1).astype(np.float32)

    if color_image is not None:
        ui = u.astype(np.intp)
        vi = v.astype(np.intp)
        colors = color_image[vi, ui].astype(np.float32) / 255.0
        return points, colors

    return points


def parse_camera_transform(transform_str: str) -> tuple[np.ndarray, Rotation]:
    """Parse camera-to-robot transform from a string.

    Accepted formats:
        6 values: "x y z roll_deg pitch_deg yaw_deg"
            Values as shown in the openarm.html 3D viewer.  The viewer uses
            THREE.js Y-up scene coordinates; this function converts to the
            URDF Z-up robot frame automatically.

            Conversion:
              - Position: (x_s, y_s, z_s) → (x_s, -z_s, y_s)  [Rx(90°)]
              - Rotation: R_urdf = Rx(90°) @ R_scene @ diag(-1,-1,1)
                where R_scene = Ry(yaw) @ Rx(roll) @ Ry(pitch).
                The viewer's IMU path decomposes gravity as ZYX euler
                (roll=x, pitch=y) then applies yaw separately around Y.
                diag(-1,-1,1) accounts for the viewer's negated XY
                deprojection (see openarm.html updatePointCloudGeneric).

        7 values: "tx ty tz qw qx qy qz"
            Quaternion (scalar-first) already in URDF robot frame.

    Returns:
        (translation (3,), Rotation) — maps standard CV camera-frame points
        (x-right, y-down, z-forward) to URDF robot-frame (Z-up).
    """
    parts = [float(x) for x in transform_str.split()]
    if len(parts) == 6:
        # Position: THREE.js scene (Y-up) → URDF (Z-up) via Rx(90°)
        x_s, y_s, z_s = parts[0], parts[1], parts[2]
        t = np.array([x_s, -z_s, y_s], dtype=np.float32)

        # Rotation: scene R_scene → URDF R_correct
        # IMU convention: gravity is decomposed as ZYX euler (roll=x, pitch=y),
        # then yaw is applied separately around Y.
        # R_scene = Ry(yaw) @ Rx(roll) @ Ry(pitch)
        roll_deg, pitch_deg, yaw_deg = parts[3], parts[4], parts[5]
        R_scene = (
            Rotation.from_euler("y", yaw_deg, degrees=True)
            * Rotation.from_euler("x", roll_deg, degrees=True)
            * Rotation.from_euler("y", pitch_deg, degrees=True)
        ).as_matrix()
        # Rx(90°): converts Y-up scene to Z-up URDF
        Rx90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
        # F: viewer negates X and Y in deprojection vs standard CV
        F = np.diag([-1.0, -1.0, 1.0])
        R_correct = Rx90 @ R_scene @ F
        rot = Rotation.from_matrix(R_correct)
    elif len(parts) == 7:
        t = np.array(parts[:3], dtype=np.float32)
        qw, qx, qy, qz = parts[3], parts[4], parts[5], parts[6]
        rot = Rotation.from_quat([qx, qy, qz, qw])  # scipy uses xyzw
    else:
        raise ValueError(
            f"CAMERA_TRANSFORM expects 6 (xyzrpy°) or 7 (xyzquat) values, got {len(parts)}"
        )
    return t, rot


def transform_points(
    points: np.ndarray,
    translation: np.ndarray,
    rotation: Rotation,
) -> np.ndarray:
    """Transform (N, 3) points from camera frame to robot base frame."""
    return (rotation.as_matrix() @ points.T).T + translation


def pointcloud_to_tensor(points: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert numpy point cloud to GPU tensor."""
    return torch.tensor(points, dtype=torch.float32, device=device)


def filter_below_table(
    pc_robot: np.ndarray,
    table_z: float,
    margin: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove points at or below the table surface.

    Keeps only points more than *margin* above *table_z*.  This removes
    table surface points, floor points, and arm-below-table reflections.

    Returns:
        ``(filtered_points, mask)`` — the filtered (M, 3) array and the
        boolean mask applied to *pc_robot*.
    """
    mask = pc_robot[:, 2] > table_z + margin
    return pc_robot[mask], mask


def mask_near_point(
    pc: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Boolean mask excluding points within *radius* of *center*.

    Useful for removing a grasped object's points from the SDF so the
    optimizer doesn't treat the held object as an obstacle during
    transport/retreat phases.

    Returns:
        (N,) boolean mask — True for points to KEEP.
    """
    dist = np.linalg.norm(pc - center, axis=1)
    return dist > radius


def compute_table_plane(
    cam_t: np.ndarray,
    cam_rot: Rotation,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    w: int,
    h: int,
    pc_robot: np.ndarray,
    device: torch.device | str = "cpu",
    z_percentile: float = 33.0,
) -> tuple[tuple[float, tuple[float, float, float, float]], torch.Tensor | None]:
    """Compute a table collision plane from camera frustum and point cloud.

    Projects the four image corners as rays from the camera origin and
    intersects them with a horizontal plane at the detected table height.
    Uses histogram peak detection (1 cm bins) to find the dominant
    horizontal surface, which is robust against floor and arm points
    that would skew a simple percentile estimate.

    Args:
        cam_t: (3,) camera translation in robot frame.
        cam_rot: Camera rotation (scipy Rotation, camera→robot).
        fx, fy, cx, cy: Camera intrinsics.
        w, h: Image width/height in pixels.
        pc_robot: (N, 3) point cloud already in robot frame.
        device: Torch device for the returned polygon tensor.
        z_percentile: Fallback percentile (used only when histogram
            detection cannot run).

    Returns:
        ``(table_plane, table_polygon)`` where *table_plane* is
        ``(plane_z, (x_min, x_max, y_min, y_max))`` and *table_polygon*
        is a ``(4, 2)`` tensor of the frustum footprint (or ``None`` if
        fewer than 4 corners could be projected).
    """
    # Histogram peak detection: bin Z values into 1 cm bins, find
    # significant horizontal surfaces, and pick the *highest* one as the
    # table.  This distinguishes the table from the ground/floor: both
    # are dense horizontal surfaces, but the table is higher.
    z_vals = pc_robot[:, 2]
    bin_edges = np.arange(float(z_vals.min()), float(z_vals.max()) + 0.01, 0.01)
    if len(bin_edges) >= 2:
        counts, edges = np.histogram(z_vals, bins=bin_edges)
        # A "significant" surface has at least 5% of total points.
        threshold = max(len(z_vals) * 0.05, 1)
        significant = np.where(counts >= threshold)[0]
        if len(significant) > 0:
            # Group significant bins into contiguous clusters.  A gap of
            # >3 cm (3 bins) between significant bins starts a new cluster.
            # The table is the *highest* cluster (floor is lower).
            clusters: list[list[int]] = [[significant[0]]]
            for i in range(1, len(significant)):
                if significant[i] - significant[i - 1] > 3:
                    clusters.append([])
                clusters[-1].append(significant[i])
            # Pick the highest cluster — its densest bin is the table.
            best_cluster = np.array(clusters[-1])
            peak_in_cluster = best_cluster[int(counts[best_cluster].argmax())]
            table_top_z = float((edges[peak_in_cluster] + edges[peak_in_cluster + 1]) / 2)
        else:
            # No significant bin — fall back to global peak.
            peak_idx = int(counts.argmax())
            table_top_z = float((edges[peak_idx] + edges[peak_idx + 1]) / 2)
    else:
        table_top_z = float(np.percentile(z_vals, z_percentile))

    rot_matrix = cam_rot.as_matrix() if hasattr(cam_rot, "as_matrix") else np.array(cam_rot)
    corners_uv = [(0, 0), (w, 0), (w, h), (0, h)]
    table_corners = []
    for u, v in corners_uv:
        ray_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
        ray_robot = rot_matrix @ ray_cam
        if abs(ray_robot[2]) < 1e-6:
            continue
        t_param = (table_top_z - cam_t[2]) / ray_robot[2]
        if t_param < 0:
            continue
        table_corners.append(cam_t + t_param * ray_robot)

    if len(table_corners) >= 4:
        table_corners = np.array(table_corners)
        # Pad the polygon outward from its centroid so the table region
        # covers the full arm workspace, not just the camera FOV.
        centroid = table_corners[:, :2].mean(axis=0)
        pad = 0.20  # metres — enough to cover arm workspace near table edge
        dirs = table_corners[:, :2] - centroid
        norms = np.linalg.norm(dirs, axis=1, keepdims=True).clip(1e-6, None)
        table_corners[:, :2] += dirs / norms * pad

        table_polygon = torch.tensor(
            table_corners[:, :2], dtype=torch.float32, device=device
        )
        bounds = (
            float(table_corners[:, 0].min()),
            float(table_corners[:, 0].max()),
            float(table_corners[:, 1].min()),
            float(table_corners[:, 1].max()),
        )
    else:
        table_polygon = None
        bounds = (
            float(pc_robot[:, 0].min()) - 0.05,
            float(pc_robot[:, 0].max()) + 0.05,
            float(pc_robot[:, 1].min()) - 0.05,
            float(pc_robot[:, 1].max()) + 0.05,
        )

    table_plane = (table_top_z, bounds)
    return table_plane, table_polygon
