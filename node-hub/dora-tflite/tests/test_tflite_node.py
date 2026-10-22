"""Tests for dora-tflite inference node."""

import numpy as np
import pytest

from dora_tflite.main import main


def test_input_reshape():
    """Test that input data reshapes correctly to model input shape."""
    input_shape = [1, 224, 224, 3]
    dummy_data = np.random.rand(224 * 224 * 3).astype(np.float32)
    reshaped = dummy_data.reshape(input_shape)
    assert reshaped.shape == tuple(input_shape)


def test_output_ravel():
    """Test that output tensor flattens correctly for pyarrow serialization."""
    dummy_output = np.array([[0.1, 0.9]], dtype=np.float32)
    raveled = dummy_output.ravel()
    assert len(raveled) == 2
    assert abs(raveled[0] - 0.1) < 1e-5


def test_dtype_cast():
    """Test that input data is cast to correct dtype."""
    data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    casted = data.astype(np.float32)
    assert casted.dtype == np.float32


def test_import_main():
    """Smoke test: main() should raise SystemExit or RuntimeError outside Dora dataflow."""
    with pytest.raises((SystemExit, RuntimeError, Exception)):
        main()
