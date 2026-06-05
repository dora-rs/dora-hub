"""Tests for opencv_video_capture.main."""

import sys
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from opencv_video_capture.main import main


def _make_frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _make_tick_event():
    return {"type": "INPUT", "id": "tick", "metadata": {}}


def _run_main(events, mock_cap):
    mock_node = MagicMock()
    mock_node.__iter__ = MagicMock(return_value=iter(events))

    with patch("opencv_video_capture.main.cv2.VideoCapture", return_value=mock_cap), \
         patch("opencv_video_capture.main.Node", return_value=mock_node), \
         patch("opencv_video_capture.main.RUNNER_CI", False):
        main()
    return mock_node


@pytest.fixture(autouse=True)
def clear_argv():
    """Prevent argparse from picking up pytest's own command-line arguments."""
    with patch.object(sys, "argv", ["opencv-video-capture"]):
        yield


def test_import_main():
    """Node entrypoint is importable."""
    # Check that everything is working, and catch dora Runtime Exception as we're not running in a dora dataflow.
    with pytest.raises(RuntimeError):
        main()



def test_read_failure_during_normal_operation_raise():
    """video_capture.read() returning False without prior shutdown signal must raise RuntimeError."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (False, None)

    with pytest.raises(RuntimeError, match="cannot read frame"):
        events = [_make_tick_event()]
        _run_main(events, mock_cap)


def test_stop_exits_cleanly():
    """STOP event must exit cleanly without raising."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True

    events = [{"type": "STOP"}]
    mock_node = _run_main(events, mock_cap)
    mock_node.send_output.assert_not_called()



def test_input_closed_tick_exits_cleanly():
    """INPUT_CLOSED for tick must exit cleanly without raising."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True

    events = [{"type": "INPUT_CLOSED", "id": "tick"}]
    mock_node = _run_main(events, mock_cap)
    mock_node.send_output.assert_not_called()


def test_input_closed_other_does_not_exit():
    """INPUT_CLOSED for an input other than tick must not stop the node."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, _make_frame())

    events = [
        {"type": "INPUT_CLOSED", "id": "other"},
        _make_tick_event(),
    ]
    mock_node = _run_main(events, mock_cap)
    mock_node.send_output.assert_called_once()


def test_capture_fps_sets_cap_prop_fps(monkeypatch):
    """CAPTURE_FPS must be requested from the camera via CAP_PROP_FPS."""
    monkeypatch.setenv("CAPTURE_FPS", "60")
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 60.0

    _run_main([{"type": "STOP"}], mock_cap)
    mock_cap.set.assert_any_call(cv2.CAP_PROP_FPS, 60)


def test_capture_fps_not_set_by_default(monkeypatch):
    """Without CAPTURE_FPS, the camera fps must be left untouched."""
    monkeypatch.delenv("CAPTURE_FPS", raising=False)
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True

    _run_main([{"type": "STOP"}], mock_cap)
    fps_calls = [
        c for c in mock_cap.set.call_args_list
        if c.args and c.args[0] == cv2.CAP_PROP_FPS
    ]
    assert not fps_calls


def test_capture_fps_warns_when_camera_negotiates_other_fps(monkeypatch, capsys):
    """A warning must be printed when the camera does not deliver the requested fps."""
    monkeypatch.setenv("CAPTURE_FPS", "60")
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 30.0

    _run_main([{"type": "STOP"}], mock_cap)
    out = capsys.readouterr().out
    assert "Warning" in out
    assert "60" in out
    assert "30" in out


def test_capture_fps_no_warning_when_fps_matches(monkeypatch, capsys):
    """No warning must be printed when the camera delivers the requested fps."""
    monkeypatch.setenv("CAPTURE_FPS", "60")
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 60.0

    _run_main([{"type": "STOP"}], mock_cap)
    assert "Warning" not in capsys.readouterr().out


def test_capture_fourcc_sets_cap_prop_fourcc(monkeypatch):
    """CAPTURE_FOURCC must be applied to the camera via CAP_PROP_FOURCC."""
    monkeypatch.setenv("CAPTURE_FOURCC", "MJPG")
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True

    _run_main([{"type": "STOP"}], mock_cap)
    mock_cap.set.assert_any_call(
        cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")
    )


def test_capture_property_order_fourcc_resolution_fps(monkeypatch):
    """Properties must be set as fourcc -> width -> height -> fps (required by V4L2)."""
    monkeypatch.setenv("CAPTURE_FOURCC", "MJPG")
    monkeypatch.setenv("IMAGE_WIDTH", "1280")
    monkeypatch.setenv("IMAGE_HEIGHT", "720")
    monkeypatch.setenv("CAPTURE_FPS", "60")
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 60.0

    _run_main([{"type": "STOP"}], mock_cap)
    props = [c.args[0] for c in mock_cap.set.call_args_list]
    assert props == [
        cv2.CAP_PROP_FOURCC,
        cv2.CAP_PROP_FRAME_WIDTH,
        cv2.CAP_PROP_FRAME_HEIGHT,
        cv2.CAP_PROP_FPS,
    ]
