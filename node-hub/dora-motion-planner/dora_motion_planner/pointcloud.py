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
    min_depth_mm: int = 10,
    max_depth_mm: int = 2000,
) -> np.ndarray:
    """Convert a mono16 depth image to a 3D point cloud in camera frame.

    Args:
        depth_map: Flat uint16 array (H*W) of depth in millimetres.
        fx, fy: Focal lengths in pixels.
        cx, cy: Principal point in pixels.
        width, height: Image dimensions.
        stride: Downsample factor (take every stride-th pixel in each axis).
        min_depth_mm, max_depth_mm: Valid depth range.

    Returns:
        (N, 3) float32 array of 3D points in camera frame (metres).
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
        return np.zeros((0, 3), dtype=np.float32)

    # Deproject: camera frame (z forward, x right, y down)
    z = d * 0.001  # mm -> m
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return np.stack([x, y, z], axis=-1).astype(np.float32)


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
                where R_scene = Rz(yaw) * Ry(pitch) * Rx(roll) in degrees,
                and diag(-1,-1,1) accounts for the viewer's negated XY
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
        roll_deg, pitch_deg, yaw_deg = parts[3], parts[4], parts[5]
        R_scene = Rotation.from_euler(
            "xyz",
            [roll_deg, pitch_deg, yaw_deg],
            degrees=True,
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
