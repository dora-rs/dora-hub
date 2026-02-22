"""Tests for dora-teleop-xr node."""

import numpy as np

from dora_teleop_xr.main import pose_to_array, reorder_joint_state_for_rerun


def test_pose_to_array():
    """Test pose_to_array function."""

    class MockPose:
        """Mock pose object."""

        def __init__(self, pos, ori):
            self.position = pos
            self.orientation = ori

    pose = MockPose(
        pos={"x": 1.0, "y": 2.0, "z": 3.0},
        ori={"x": 0.1, "y": 0.2, "z": 0.3, "w": 0.9},
    )
    result = pose_to_array(pose)
    assert result is not None
    expected = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9], dtype=np.float32)
    assert np.allclose(result, expected)


def test_pose_to_array_none():
    """Test pose_to_array with None."""
    assert pose_to_array(None) is None


def test_so101_robot_init():
    """Test SO101 robot initialization from teleop_xr package."""
    from teleop_xr.ik.robots.so101 import SO101Robot

    robot = SO101Robot()
    assert robot is not None
    assert len(robot.actuated_joint_names) == 5


def test_reorder_joint_state_for_rerun_so101_order():
    """Test joint state reordering for SO101 URDF traversal order."""
    SO101_ORDER = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    )

    class MockRobot:
        actuated_joint_names = [
            "wrist_roll",
            "wrist_flex",
            "elbow_flex",
            "shoulder_lift",
            "shoulder_pan",
        ]

    q_current = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    reordered = reorder_joint_state_for_rerun(MockRobot(), q_current, SO101_ORDER)

    expected = np.array([5, 4, 3, 2, 1], dtype=np.float32)
    assert np.allclose(reordered, expected)


def test_reorder_joint_state_for_rerun_unknown_names_keeps_input():
    """Test unknown joint names are passed through unchanged."""

    class MockRobot:
        actuated_joint_names = ["joint_a", "joint_b"]

    q_current = np.array([0.1, -0.1], dtype=np.float32)
    reordered = reorder_joint_state_for_rerun(MockRobot(), q_current, None)

    assert np.allclose(reordered, q_current)
