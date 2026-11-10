"""Test module for dora_dataset_record package."""

import pytest


def test_import_main():
    """Test importing and running the main function."""
    from dora_dataset_record.main import main

    # Check that everything is working, and catch Dora RuntimeError
    # as we're not running in a Dora dataflow.
    with pytest.raises(RuntimeError):
        main()


def test_record_and_finalize_episode(tmp_path, monkeypatch):
    """Write an episode through LeRobot and verify the finalized Parquet data."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from dora_dataset_record.main import DoraLeRobotRecorder

    root = tmp_path / "dataset"
    for name, value in {
        "ROBOT_JOINTS": "arm",
        "REPO_ID": "local/dora-recorder-test",
        "ROOT_PATH": str(root),
        "CAMERA_NAMES": "front",
        "CAMERA_FRONT_RESOLUTION": "16,16,3",
        "IMAGE_WRITER_THREADS": "0",
        "FPS": "4",
        "SINGLE_TASK": "hold",
        "USE_VIDEOS": "false",
        "PUSH_TO_HUB": "false",
    }.items():
        monkeypatch.setenv(name, value)

    # Drive the recorder synchronously without starting a Dora dataflow.
    monkeypatch.setattr(DoraLeRobotRecorder, "_start_frame_timer", lambda self: None)
    recorder = DoraLeRobotRecorder()
    recorder._start_episode()
    recorder.recording_started = True
    pixels = pa.array([0] * (16 * 16 * 3), type=pa.uint8())
    for value in (0.25, 0.5):
        recorder.handle_input("robot_action", pa.array([value]), {})
        recorder.handle_input("robot_state", pa.array([value]), {})
        recorder.handle_input("front", pixels, {"height": 16, "width": 16, "encoding": "rgb8"})
        recorder._add_frame()
    recorder.finalize_dataset()

    table = pq.read_table(next((root / "data").rglob("*.parquet")))
    assert table.num_rows == 2
    assert table["timestamp"].to_pylist() == [0.0, 0.25]
    assert table["observation.images.front"].null_count == 0
