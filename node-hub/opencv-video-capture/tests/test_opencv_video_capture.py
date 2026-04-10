"""Tests for opencv_video_capture.main."""

import sys
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def clear_argv():
    """Prevent argparse from picking up pytest's own command-line arguments."""
    with patch.object(sys, "argv", ["opencv-video-capture"]):
        yield


def test_import_main():
    """Node entrypoint is importable."""
    from opencv_video_capture.main import main

    # Check that everything is working, and catch dora Runtime Exception as we're not running in a dora dataflow.
    with pytest.raises(RuntimeError):
        main()
