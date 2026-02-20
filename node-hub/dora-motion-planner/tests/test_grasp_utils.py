"""Tests for grasp pose computation from jaw pixels."""

import numpy as np
from scipy.spatial.transform import Rotation

from dora_motion_planner.grasp_utils import (
    pixel_to_3d,
    grasp_pose_from_jaw_pixels,
)


class TestPixelTo3D:
    """Test single-pixel deprojection."""

    def test_center_pixel(self):
        """Center pixel at 1m depth should give (0, 0, 1)."""
        W, H = 640, 480
        depth = np.zeros(H * W, dtype=np.uint16)
        depth[240 * W + 320] = 1000  # 1m at center
        p = pixel_to_3d(320, 240, depth, 600, 600, 320, 240, W, H)
        assert p is not None
        np.testing.assert_allclose(p, [0, 0, 1.0], atol=0.01)

    def test_invalid_depth_returns_none(self):
        """Zero depth should return None."""
        depth = np.zeros(640 * 480, dtype=np.uint16)
        p = pixel_to_3d(320, 240, depth, 600, 600, 320, 240)
        assert p is None

    def test_patch_averaging(self):
        """Uses median of valid patch pixels."""
        W, H = 640, 480
        depth = np.zeros(H * W, dtype=np.uint16)
        # Fill a 5x5 patch around (320, 240) with varying depths
        for dv in range(-2, 3):
            for du in range(-2, 3):
                depth[(240 + dv) * W + (320 + du)] = 500
        # One outlier
        depth[240 * W + 320] = 1000
        p = pixel_to_3d(320, 240, depth, 600, 600, 320, 240, W, H, patch_size=5)
        assert p is not None
        # Median should be 500mm = 0.5m (24 pixels at 500, 1 at 1000)
        assert abs(p[2] - 0.5) < 0.01


class TestGraspPoseFromJawPixels:
    """Test full grasp pose computation."""

    def make_depth_with_object(
        self, W=640, H=480, obj_depth_mm=300, table_depth_mm=500
    ):
        """Create a depth map with an object above a table."""
        depth = np.full(H * W, table_depth_mm, dtype=np.uint16)
        # Object region in center
        for v in range(200, 280):
            for u in range(280, 360):
                depth[v * W + u] = obj_depth_mm
        return depth

    def test_basic_grasp(self):
        """Two jaw pixels on an object produce a valid grasp pose."""
        W, H = 640, 480
        depth = self.make_depth_with_object(W, H)
        # Identity camera transform
        cam_t = np.zeros(3, dtype=np.float32)
        cam_rot = Rotation.identity()

        result = grasp_pose_from_jaw_pixels(
            u1=300,
            v1=240,  # left jaw
            u2=340,
            v2=240,  # right jaw
            depth_map=depth,
            fx=600,
            fy=600,
            cx=320,
            cy=240,
            cam_translation=cam_t,
            cam_rotation=cam_rot,
            width=W,
            height=H,
        )

        assert result is not None
        grasp_xyzrpy, pregrasp_xyzrpy = result

        assert grasp_xyzrpy.shape == (6,)
        assert pregrasp_xyzrpy.shape == (6,)

        # Pre-grasp should be higher than grasp
        assert pregrasp_xyzrpy[2] > grasp_xyzrpy[2]

    def test_grasp_center_between_jaws(self):
        """Grasp X/Y should be approximately the midpoint of the two jaw points."""
        W, H = 640, 480
        depth = self.make_depth_with_object(W, H)
        cam_t = np.zeros(3, dtype=np.float32)
        cam_rot = Rotation.identity()

        result = grasp_pose_from_jaw_pixels(
            u1=300,
            v1=240,
            u2=340,
            v2=240,
            depth_map=depth,
            fx=600,
            fy=600,
            cx=320,
            cy=240,
            cam_translation=cam_t,
            cam_rotation=cam_rot,
            width=W,
            height=H,
        )
        grasp_xyzrpy, _ = result

        # Midpoint pixel is (320, 240) = center → X should be near 0
        assert abs(grasp_xyzrpy[0]) < 0.05

    def test_floor_height_clamp(self):
        """Grasp Z should not go below floor_height."""
        W, H = 640, 480
        # Very close depth → very high Z after transform, but with a huge
        # negative offset it should clamp to floor
        depth = np.full(H * W, 100, dtype=np.uint16)  # 10cm
        cam_t = np.zeros(3, dtype=np.float32)
        cam_rot = Rotation.identity()

        result = grasp_pose_from_jaw_pixels(
            u1=310,
            v1=240,
            u2=330,
            v2=240,
            depth_map=depth,
            fx=600,
            fy=600,
            cx=320,
            cy=240,
            cam_translation=cam_t,
            cam_rotation=cam_rot,
            width=W,
            height=H,
            grasp_depth_offset=-10.0,  # huge negative offset
            floor_height=0.01,
        )
        assert result is not None
        grasp_xyzrpy, _ = result
        assert grasp_xyzrpy[2] >= 0.01 - 1e-6  # clamped to floor (float32 tolerance)

    def test_invalid_depth_returns_none(self):
        """If jaw pixels have no valid depth, returns None."""
        depth = np.zeros(640 * 480, dtype=np.uint16)
        cam_t = np.zeros(3, dtype=np.float32)
        cam_rot = Rotation.identity()

        result = grasp_pose_from_jaw_pixels(
            u1=320,
            v1=240,
            u2=340,
            v2=240,
            depth_map=depth,
            fx=600,
            fy=600,
            cx=320,
            cy=240,
            cam_translation=cam_t,
            cam_rotation=cam_rot,
        )
        assert result is None
