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
    patch_size: int = 21,
    cam_translation: np.ndarray | None = None,
    cam_rotation: "Rotation | None" = None,
    table_z: float = 0.0,
) -> np.ndarray | None:
    """Deproject a pixel to 3D in camera frame, averaging a patch for robustness.

    If depth is invalid and camera transform is provided, falls back to
    intersecting the pixel ray with the table plane at *table_z* in robot
    frame, then converting back to camera frame depth.

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
    valid = (patch > 50) & (patch < 5000)  # 50mm–5m (skip sensor noise)
    if np.any(valid):
        z_mm = np.median(patch[valid])
        z = z_mm * 0.001  # mm -> m
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.array([x, y, z], dtype=np.float32)

    # Fallback: intersect pixel ray with table plane in robot frame
    if cam_translation is not None and cam_rotation is not None:
        ray_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)
        R = cam_rotation.as_matrix()
        ray_robot = R @ ray_cam
        if abs(ray_robot[2]) > 1e-6:
            t_param = (table_z - cam_translation[2]) / ray_robot[2]
            if t_param > 0:
                # Distance along the ray in camera frame = depth
                z = t_param * np.linalg.norm(ray_cam) / np.linalg.norm(ray_cam)  # = t_param (unit ray_cam has z=1)
                z = t_param  # ray_cam z-component is 1.0, so depth = t_param * 1.0
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                print(f"[pixel_to_3d] Fallback table-plane depth at ({u:.0f},{v:.0f}): "
                      f"z={z:.3f}m ({z*1000:.0f}mm)")
                return np.array([x, y, z], dtype=np.float32)

    print(f"[pixel_to_3d] No valid depth at ({u:.0f},{v:.0f}), "
          f"patch {patch_size}x{patch_size}: "
          f"min={patch.min():.0f} max={patch.max():.0f} "
          f"nonzero={np.count_nonzero(patch)}/{patch.size}")
    return None


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
    grasp_depth_offset: float = 0.02,
    floor_height: float = 0.04,
    approach_margin: float = 0.03,
    jaw_contact_depth: float = 0.02,
    approach_angle_deg: float = 0.0,
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
        jaw_contact_depth: How far behind the TCP the jaw contact surface is.
            The TCP sits at the finger tips; the actual gripping zone is this
            far behind.  The grasp center is shifted along the approach vector
            by this amount so the contact zone (not the tips) aligns with the
            object center.
        approach_angle_deg: Tilt angle of the gripper from vertical (0 = straight
            down, 45 = fingertips down at 45° with wrist angled up).  The tilt
            direction is from the arm base toward the object in the XY plane.

    Returns:
        (grasp_xyzrpy, pregrasp_xyzrpy, object_top_z, object_width) or None.
        object_top_z is the highest Z along the jaw line (robot frame).
        object_width is the jaw-to-jaw distance in metres (robot frame).
    """
    # Deproject both jaw pixels to 3D camera frame
    p1_cam = pixel_to_3d(
        u1, v1, depth_map, fx, fy, cx, cy, width, height,
        cam_translation=cam_translation, cam_rotation=cam_rotation,
        table_z=floor_height,
    )
    p2_cam = pixel_to_3d(
        u2, v2, depth_map, fx, fy, cx, cy, width, height,
        cam_translation=cam_translation, cam_rotation=cam_rotation,
        table_z=floor_height,
    )

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
            if 100 < d_mm < 5000:
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

    # Grasp center = deproject the center pixel (midpoint of jaw pixels in
    # image space).  This is more robust than averaging the 3D jaw endpoints,
    # because when the jaws are wider than the object the endpoint depths
    # may land on the background surface and their 3D midpoint drifts away
    # from the actual object.
    u_center = (u1 + u2) / 2.0
    v_center = (v1 + v2) / 2.0
    p_center_cam = pixel_to_3d(
        u_center, v_center, depth_map, fx, fy, cx, cy, width, height,
        cam_translation=cam_translation, cam_rotation=cam_rotation,
        table_z=floor_height,
    )
    if p_center_cam is not None:
        center = R_cam @ p_center_cam + cam_translation
        print(f"[grasp] center from pixel ({u_center:.0f},{v_center:.0f}): "
              f"{np.round(center, 4)}")
    else:
        center = (p1_rob + p2_rob) / 2.0
        print(f"[grasp] center from jaw midpoint (fallback): "
              f"{np.round(center, 4)}")
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

    # Gripper orientation: tilted from vertical by approach_angle_deg.
    # Z_ee (fingertip direction) tilts from straight-down toward the
    # object.  horiz points from the arm base toward the object in XY.
    # 0° = straight down, 90° = fully horizontal facing the object.
    if approach_angle_deg > 0.1:
        horiz = np.array([center[0], center[1], 0.0])
        horiz_norm = np.linalg.norm(horiz)
        if horiz_norm > 1e-6:
            horiz = horiz / horiz_norm
        else:
            horiz = np.array([1.0, 0.0, 0.0])
        angle_rad = np.radians(approach_angle_deg)
        approach = np.sin(angle_rad) * horiz + np.cos(angle_rad) * np.array([0.0, 0.0, -1.0])
        approach = approach / np.linalg.norm(approach)
    else:
        approach = np.array([0.0, 0.0, -1.0])
    print(f"[grasp] approach direction: {np.round(approach, 3)} ({approach_angle_deg:.0f}° from vertical)")

    # Build rotation matrix: Z_ee = approach direction (fingertips).
    # For near-vertical approach (< 30°), Y_ee = jaw axis (from pixel pair).
    # For tilted/horizontal approach (>= 30°), the jaw axis may be nearly
    # parallel to Z_ee, so instead force jaws horizontal: X_ee = up,
    # Y_ee = cross(Z_ee, up) — always perpendicular and horizontal.
    z_ee = approach
    if approach_angle_deg >= 30.0:
        # Horizontal jaws: X_ee ≈ up, Y_ee = horizontal perpendicular to Z_ee
        up = np.array([0.0, 0.0, 1.0])
        y_ee = np.cross(z_ee, up)
        y_norm = np.linalg.norm(y_ee)
        if y_norm < 1e-6:
            # Z_ee is vertical — fall back to jaw axis
            y_ee = jaw_axis
        else:
            y_ee = y_ee / y_norm
        x_ee = np.cross(y_ee, z_ee)
        x_ee = x_ee / np.linalg.norm(x_ee)
        # Re-orthogonalise Y
        y_ee = np.cross(z_ee, x_ee)
    else:
        # Near-vertical: use jaw pixel axis for Y_ee
        y_ee = jaw_axis
        x_ee = np.cross(y_ee, z_ee)
        x_norm = np.linalg.norm(x_ee)
        if x_norm < 1e-6:
            x_ee = np.array([1.0, 0.0, 0.0])
            y_ee = np.cross(z_ee, x_ee)
        else:
            x_ee = x_ee / x_norm
            y_ee = np.cross(z_ee, x_ee)
    print(f"[grasp] Y_ee (jaw axis): {np.round(y_ee, 3)}, "
          f"jaw tilt from horizontal: {np.degrees(np.arcsin(abs(y_ee[2]))):.1f}°")

    rot_matrix = np.stack([x_ee, y_ee, z_ee], axis=1)  # (3, 3) columns are axes
    rpy = Rotation.from_matrix(rot_matrix).as_euler("XYZ")

    # Shift TCP along approach so jaw contact zone (not fingertips) aligns
    # with the object center.  Only shift along Z so the gripper stays
    # centered over the pixel in XY — the angled approach means a full 3D
    # shift would pull the gripper sideways.
    if jaw_contact_depth > 0:
        z_shift = approach[2] * jaw_contact_depth  # Z component only
        center[2] += z_shift
        print(f"[grasp] jaw_contact_depth={jaw_contact_depth*1000:.0f}mm, "
              f"Z shift={z_shift*1000:.1f}mm, TCP at {np.round(center, 4)}")

    grasp_xyzrpy = np.concatenate([center, rpy]).astype(np.float32)

    # Pre-grasp: always directly above the grasp.  The arm descends
    # vertically to the grasp regardless of gripper orientation.
    pregrasp_center = center.copy()
    pregrasp_center[2] += approach_margin
    pregrasp_xyzrpy = np.concatenate([pregrasp_center, rpy]).astype(np.float32)

    return grasp_xyzrpy, pregrasp_xyzrpy, float(object_top_z), object_width


def place_pose_from_pixel(
    u: float,
    v: float,
    depth_map: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    cam_translation: np.ndarray,
    cam_rotation: Rotation,
    grasp_rpy: np.ndarray,
    width: int = 640,
    height: int = 480,
    place_depth_offset: float = 0.06,
    floor_height: float = 0.04,
    approach_margin: float = 0.05,
    jaw_contact_depth: float = 0.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Compute a place pose from a single pixel (container centroid).

    Deprojects the pixel to 3D in robot frame, builds a place pose using
    the grasp orientation (object stays level) and a pre-place waypoint
    above it.

    Args:
        u, v: Pixel coordinates of the place target center.
        depth_map: Flat uint16 array (H*W) of depth in mm.
        fx, fy, cx, cy: Camera intrinsics.
        cam_translation: (3,) camera-to-robot translation.
        cam_rotation: Camera-to-robot rotation (scipy Rotation).
        grasp_rpy: (3,) RPY from the grasp pose — reused for orientation.
        width, height: Image dimensions.
        place_depth_offset: Z offset above the surface (positive = higher).
        floor_height: Minimum Z in robot frame.
        approach_margin: Extra height above place for pre-place waypoint.
        jaw_contact_depth: Distance from TCP (fingertips) to jaw contact
            center.  The TCP is offset along the approach direction so the
            jaw center (where the object sits) aligns with the target pixel.

    Returns:
        (place_xyzrpy, preplace_xyzrpy) or None if depth is invalid.
    """
    pt_cam = pixel_to_3d(
        u, v, depth_map, fx, fy, cx, cy, width, height,
        cam_translation=cam_translation, cam_rotation=cam_rotation,
        table_z=floor_height,
    )
    if pt_cam is None:
        return None

    R_cam = cam_rotation.as_matrix()
    pt_rob = R_cam @ pt_cam + cam_translation
    print(f"[place] pixel=({u:.0f},{v:.0f}) -> robot={np.round(pt_rob, 4)}")

    # Offset TCP so the jaw center (not fingertips) aligns with the target.
    # The jaw center is jaw_contact_depth behind the TCP along the approach
    # direction (Z_ee).  Move the TCP forward along Z_ee to compensate.
    z_ee = Rotation.from_euler("XYZ", grasp_rpy).as_matrix()[:, 2]
    tcp_offset = jaw_contact_depth * z_ee  # XY + Z offset
    place_xy = pt_rob[:2] + tcp_offset[:2]  # apply XY correction only
    print(f"[place] jaw offset: {np.round(tcp_offset[:2]*1000, 1)}mm "
          f"(jaw_contact_depth={jaw_contact_depth*1000:.0f}mm)")

    place_z = max(pt_rob[2] + place_depth_offset, floor_height)
    place_pos = np.array([place_xy[0], place_xy[1], place_z], dtype=np.float32)
    place_xyzrpy = np.concatenate([place_pos, grasp_rpy]).astype(np.float32)

    preplace_pos = place_pos.copy()
    preplace_pos[2] = place_z + approach_margin
    preplace_xyzrpy = np.concatenate([preplace_pos, grasp_rpy]).astype(np.float32)

    print(
        f"[place] place={np.round(place_xyzrpy[:3], 4)}, "
        f"preplace_z={preplace_xyzrpy[2]:.4f}"
    )
    return place_xyzrpy, preplace_xyzrpy
