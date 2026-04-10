"""Tests for depth-to-pointcloud conversion."""

import numpy as np
import pytest
from dora_motion_planner.pointcloud import (
    depth_to_pointcloud,
    parse_camera_transform,
    transform_points,
)


class TestDepthToPointcloud:
    """Test depth map to 3D point cloud conversion."""

    def test_basic_conversion(self):
        """Center pixel at known depth maps to correct 3D point."""
        W, H = 640, 480
        fx, fy = 600.0, 600.0
        cx, cy = 320.0, 240.0

        depth = np.zeros(H * W, dtype=np.uint16)
        # Place a single valid depth at center pixel
        depth[240 * W + 320] = 1000  # 1 metre

        pc = depth_to_pointcloud(depth, fx, fy, cx, cy, W, H, stride=1)
        assert len(pc) == 1
        # Center pixel: x=(320-320)*1/600=0, y=(240-240)*1/600=0, z=1.0
        np.testing.assert_allclose(pc[0], [0.0, 0.0, 1.0], atol=0.01)

    def test_invalid_depth_filtered(self):
        """Depths outside valid range are excluded."""
        W, H = 640, 480
        depth = np.zeros(H * W, dtype=np.uint16)
        depth[0] = 5  # too close (< 10mm)
        depth[1] = 3000  # too far (> 2000mm)

        pc = depth_to_pointcloud(depth, 600, 600, 320, 240, W, H, stride=1)
        assert len(pc) == 0

    def test_stride_downsampling(self):
        """Stride reduces point count."""
        W, H = 640, 480
        depth = np.full(H * W, 500, dtype=np.uint16)  # all valid

        pc_s1 = depth_to_pointcloud(depth, 600, 600, 320, 240, W, H, stride=1)
        pc_s8 = depth_to_pointcloud(depth, 600, 600, 320, 240, W, H, stride=8)
        # stride=8 should give ~1/64 the points
        assert len(pc_s8) < len(pc_s1)
        assert len(pc_s8) == (H // 8) * (W // 8)


class TestCameraTransform:
    """Test camera transform parsing and point transformation."""

    # --- 7-value quaternion format (URDF frame, no conversion) ---

    def test_quat_identity(self):
        t, rot = parse_camera_transform("0 0 0 1 0 0 0")
        np.testing.assert_allclose(t, [0, 0, 0])
        np.testing.assert_allclose(rot.as_matrix(), np.eye(3), atol=1e-6)

    def test_quat_translation_only(self):
        t, rot = parse_camera_transform("1 2 3 1 0 0 0")
        np.testing.assert_allclose(t, [1, 2, 3])
        points = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        transformed = transform_points(points, t, rot)
        np.testing.assert_allclose(transformed[0], [1, 2, 3], atol=1e-6)

    def test_quat_rotation_90deg_z(self):
        """90-degree rotation about z-axis: qw=cos(45)=0.707, qz=sin(45)=0.707."""
        import math

        qw = math.cos(math.pi / 4)
        qz = math.sin(math.pi / 4)
        t, rot = parse_camera_transform(f"0 0 0 {qw} 0 0 {qz}")

        points = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        transformed = transform_points(points, np.zeros(3), rot)
        # 90deg around z: [1,0,0] -> [0,1,0]
        np.testing.assert_allclose(transformed[0], [0, 1, 0], atol=1e-5)

    # --- 6-value xyzrpy format (THREE.js viewer -> URDF conversion) ---

    def test_xyzrpy_position_conversion(self):
        """6-value position converts from THREE.js Y-up to URDF Z-up."""
        t, _ = parse_camera_transform("-0.23 0.71 0.3 0 0 0")
        # Scene (x, y, z) -> URDF (x, -z, y)
        np.testing.assert_allclose(t, [-0.23, -0.3, 0.71])

    def test_xyzrpy_depth_maps_downward(self):
        """With viewer RPY=(90,-45,0), camera depth maps to -Z (down) in URDF."""
        _, rot = parse_camera_transform("-0.23 0.71 0.3 90 0 -45")
        R = rot.as_matrix()
        # CV camera Z-axis (depth/forward) should map to URDF -Z (downward)
        cam_z_in_urdf = R @ [0, 0, 1]
        np.testing.assert_allclose(cam_z_in_urdf, [0, 0, -1], atol=1e-6)

    def test_xyzrpy_center_pixel_projection(self):
        """Center pixel at 0.5m depth projects directly below camera in URDF."""
        t, rot = parse_camera_transform("-0.23 0.71 0.3 90 0 -45")
        p_cv = np.array([[0, 0, 0.5]], dtype=np.float32)
        p_robot = transform_points(p_cv, t, rot)
        # Camera at URDF (-0.23, -0.3, 0.71) looking straight down
        # 0.5m depth: Z = 0.71 - 0.5 = 0.21
        np.testing.assert_allclose(p_robot[0], [-0.23, -0.3, 0.21], atol=0.01)

    def test_xyzrpy_off_center_pixel(self):
        """Off-center pixel at depth 0.4m projects to an offset position."""
        t, rot = parse_camera_transform("-0.23 0.71 0.3 90 0 -45")
        # Pixel (640, 500) at depth 0.4m, intrinsics: fx=906, fy=905, cx=644, cy=379
        fx, fy, cx, cy = 906, 905, 644, 379
        u, v, d = 640, 500, 0.4
        p_cv = np.array([[(u - cx) / fx * d, (v - cy) / fy * d, d]], dtype=np.float32)
        p_robot = transform_points(p_cv, t, rot)
        # Should be within arm workspace (X: [-0.45, 0.72], Y: [-0.72, 0.44], Z: [0.09, 1.31])
        assert -0.5 < p_robot[0, 0] < 0.8  # X
        assert -0.8 < p_robot[0, 1] < 0.5  # Y
        assert 0.0 < p_robot[0, 2] < 1.0  # Z (should be below camera Z=0.71)

    def test_bad_value_count_raises(self):
        """Wrong number of values raises ValueError."""
        with pytest.raises(ValueError):
            parse_camera_transform("1 2 3 4 5")
