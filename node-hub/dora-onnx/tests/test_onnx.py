"""Tests for the dora-onnx inference node."""

import pytest


def test_import_main():
    """Test that the main module can be imported."""
    from dora_onnx.main import main

    # Check that everything is working, and catch dora Runtime Exception as we're not running in a dora dataflow.
    with pytest.raises(RuntimeError):
        main()
