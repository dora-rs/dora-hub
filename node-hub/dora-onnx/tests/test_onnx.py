"""Tests for the dora-onnx inference node."""
import pytest
import numpy as np
import pyarrow as pa
from dora_onnx.main import preprocess_image

def test_preprocess_rgb8():
    """Test that a raw rgb8 PyArrow array is correctly converted to NCHW tensor."""
    width, height = 640, 480
    channels = 3
    
    # Create a dummy image (HWC layout) with random uint8 pixels
    dummy_image = np.random.randint(0, 255, size=(height, width, channels), dtype=np.uint8)
    
    # Flatten it into a PyArrow array (simulating Dora dataflow memory)
    pa_storage = pa.array(dummy_image.ravel())
    
    # Run our preprocessing function
    tensor = preprocess_image(pa_storage, width, height, "rgb8")
    
    # Assertions
    assert tensor.shape == (1, 3, height, width), "Tensor must be in NCHW format"
    assert tensor.dtype == np.float32, "Tensor must be cast to float32"
    assert np.max(tensor) <= 1.0, "Tensor must be normalized between 0 and 1"

def test_preprocess_unsupported_encoding():
    """Test that unsupported encodings raise a clear error."""
    pa_storage = pa.array([0, 1, 2])
    with pytest.raises(RuntimeError, match="Unsupported image encoding"):
        preprocess_image(pa_storage, 10, 10, "nv12")