"""Compute grasp poses from two jaw pixel locations on an RGBD image.

Given two pixels representing where each gripper jaw should contact the
object, deprojects them to 3D, computes a grasp center + orientation,
and adjusts height so the gripper approaches from above without hitting
the table.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def pixel_to_3d(
    u: float,
    v: float,
    depth_map: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    width: int = 640,
    height: int = 480,
    patch_size: int = 5,
) -> np.ndarray | None:
    """Deproject a pixel to 3D in camera frame, averaging a patch for robustness.

    Returns (3,) array [x, y, z] in metres, or None if depth is invalid.
    """
    depth_2d = depth_map.reshape(height, width)
    u_int, v_int = int(round(u)), int(round(v))
    half = patch_size // 2

    # Extract patch, clamp to image bounds
    v_lo = max(0, v_int - half)
    v_hi = min(height, v_int + half + 1)
    u_lo = max(0, u_int - half)
    u_hi = min(width, u_int + half + 1)

    patch = depth_2d[v_lo:v_hi, u_lo:u_hi].astype(np.float32)
    valid = (patch > 10) & (patch < 5000)  # 10mm–5m
    if not np.any(valid):
        return None

    z_mm = np.median(patch[valid])
    z = z_mm * 0.001  # mm -> m
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.array([x, y, z], dtype=np.float32)


def grasp_pose_from_jaw_pixels(
    u1: float,
    v1: float,
    u2: float,
    v2: float,
    depth_map: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    cam_translation: np.ndarray,
    cam_rotation: Rotation,
    width: int = 640,
    height: int = 480,
    grasp_depth_offset: float = -0.01,
    floor_height: float = 0.005,
    approach_margin: float = 0.03,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute a grasp pose from two jaw pixel positions.

    The two pixels mark where each gripper jaw should contact the object.
    Returns (position, rpy) in robot base frame suitable for IK, or None
    if depth is invalid.

    Args:
        u1, v1: Pixel coordinates of jaw 1.
        u2, v2: Pixel coordinates of jaw 2.
        depth_map: Flat uint16 array (H*W) of depth in mm.
        fx, fy, cx, cy: Camera intrinsics.
        cam_translation: (3,) camera-to-robot translation.
        cam_rotation: Camera-to-robot rotation (scipy Rotation).
        width, height: Image dimensions.
        grasp_depth_offset: Z offset from object surface (negative = go deeper).
        floor_height: Minimum Z in robot frame to avoid table collision.
        approach_margin: Extra height above grasp for the pre-grasp waypoint.

    Returns:
        (grasp_xyzrpy, pregrasp_xyzrpy, object_top_z, object_width) or None.
        object_top_z is the highest Z along the jaw line (robot frame).
        object_width is the jaw-to-jaw distance in metres (robot frame).
    """
    # Deproject both jaw pixels to 3D camera frame
    p1_cam = pixel_to_3d(u1, v1, depth_map, fx, fy, cx, cy, width, height)
    p2_cam = pixel_to_3d(u2, v2, depth_map, fx, fy, cx, cy, width, height)

    if p1_cam is None or p2_cam is None:
        return None

    print(
        f"[grasp] cam-frame: p1={np.round(p1_cam, 4)} p2={np.round(p2_cam, 4)} "
        f"intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}"
    )

    # Transform to robot base frame
    R_cam = cam_rotation.as_matrix()
    p1_rob = R_cam @ p1_cam + cam_translation
    p2_rob = R_cam @ p2_cam + cam_translation
    print(f"[grasp] robot-frame: p1={np.round(p1_rob, 4)} p2={np.round(p2_rob, 4)}")

    # Scan all pixels along the jaw line to find the object's highest point.
    # This is more robust than just using the two jaw endpoints, which may
    # land on the table surface or a noisy depth pixel.
    n_samples = max(20, int(np.hypot(u2 - u1, v2 - v1)))
    us = np.linspace(u1, u2, n_samples)
    vs = np.linspace(v1, v2, n_samples)
    depth_2d = depth_map.reshape(height, width)
    best_z_robot = -np.inf
    for u_s, v_s in zip(us, vs):
        ui, vi = int(round(u_s)), int(round(v_s))
        if 0 <= vi < height and 0 <= ui < width:
            d_mm = float(depth_2d[vi, ui])
            if 10 < d_mm < 5000:
                z_m = d_mm * 0.001
                pt_cam = np.array([
                    (u_s - cx) * z_m / fx,
                    (v_s - cy) * z_m / fy,
                    z_m,
                ], dtype=np.float32)
                pt_rob = R_cam @ pt_cam + cam_translation
                if pt_rob[2] > best_z_robot:
                    best_z_robot = pt_rob[2]

    # object_top_z = highest point along jaw line in robot frame
    object_top_z = best_z_robot if best_z_robot > -np.inf else max(p1_rob[2], p2_rob[2])
    print(f"[grasp] object top z={object_top_z:.4f}m (scanned {n_samples} pixels between jaws)")

    # Grasp center = midpoint of the two jaw points (XY),
    # Z will be adjusted later using table plane if available.
    center = (p1_rob + p2_rob) / 2.0
    grasp_z = object_top_z + grasp_depth_offset
    grasp_z = max(grasp_z, floor_height)
    center[2] = grasp_z

    # Orientation: align gripper Y-axis with jaw axis, Z-axis pointing down
    jaw_axis = p2_rob - p1_rob
    jaw_axis[2] = 0  # project to horizontal plane for stable grasp
    jaw_len = np.linalg.norm(jaw_axis)
    object_width = float(jaw_len)  # distance between jaw contact points (metres)
    print(f"[grasp] object width={object_width*1000:.1f}mm (jaw distance in robot frame)")
    if jaw_len < 1e-6:
        jaw_axis = np.array([0.0, 1.0, 0.0])
    else:
        jaw_axis = jaw_axis / jaw_len

    # Approach direction: straight down (-Z in robot frame)
    approach = np.array([0.0, 0.0, -1.0])

    # Build rotation matrix: Z_ee = approach, Y_ee = jaw_axis, X_ee = Y x Z
    z_ee = approach
    y_ee = jaw_axis
    x_ee = np.cross(y_ee, z_ee)
    x_norm = np.linalg.norm(x_ee)
    if x_norm < 1e-6:
        # jaw_axis is parallel to approach — fall back to default orientation
        x_ee = np.array([1.0, 0.0, 0.0])
        y_ee = np.cross(z_ee, x_ee)
    else:
        x_ee = x_ee / x_norm
        # Re-orthogonalise Y
        y_ee = np.cross(z_ee, x_ee)

    rot_matrix = np.stack([x_ee, y_ee, z_ee], axis=1)  # (3, 3) columns are axes
    rpy = Rotation.from_matrix(rot_matrix).as_euler("XYZ")

    grasp_xyzrpy = np.concatenate([center, rpy]).astype(np.float32)

    # Pre-grasp: same orientation, higher Z
    pregrasp_center = center.copy()
    pregrasp_center[2] = grasp_z + approach_margin
    pregrasp_xyzrpy = np.concatenate([pregrasp_center, rpy]).astype(np.float32)

    return grasp_xyzrpy, pregrasp_xyzrpy, float(object_top_z), object_width
