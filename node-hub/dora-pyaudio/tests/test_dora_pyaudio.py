"""TODO: Add docstring."""

import numpy as np
import pytest


def test_import_main():
    """TODO: Add docstring."""
    from dora_pyaudio.main import main

    # Check that everything is working, and catch dora Runtime Exception as we're not running in a dora dataflow.
    with pytest.raises(RuntimeError):
        main()


class _RecordingStream:
    """Stand-in for a pyaudio output stream that keeps the written bytes."""

    def __init__(self):
        self.written = b""

    def write(self, data):
        self.written += data


def test_play_audio_saturates_loud_float_samples():
    """Loud float samples must clip to the int16 range instead of wrapping around."""
    from dora_pyaudio.main import play_audio

    stream = _RecordingStream()
    audio = np.array([0.0, 0.1, 0.5, 0.9, 1.0, -0.5, -1.0], dtype=np.float32)

    play_audio(audio, 16000, stream)

    played = np.frombuffer(stream.written, dtype=np.int16)
    np.testing.assert_array_equal(
        played, [0, 7000, 32767, 32767, 32767, -32768, -32768],
    )
