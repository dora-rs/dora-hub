"""Tests for opencv_video_capture.main."""

import sys
from unittest.mock import MagicMock, patch

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
