import json
from datetime import datetime, timedelta, timezone

import pyarrow as pa
import pyarrow.parquet as pq

from dora_parquet_recorder import main


def test_records_datetime_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "LOG_DIR", str(tmp_path))
    recorder = main.DoraParquetRecorder()
    offset = timezone(timedelta(hours=5, minutes=30))
    timestamp = datetime(2026, 1, 2, 3, 4, 5, 123456, tzinfo=offset)
    value = pa.array([123], type=pa.int32())
    try:
        recorder.handle_input(
            "data",
            value,
            {
                "timestamp": timestamp,
                "sequence": 7,
                "active": True,
                "nested": [timestamp, None],
            },
        )
        recorder.handle_input("data", value, {"sequence": 8})
    finally:
        recorder._shutdown()

    table = pq.read_table(tmp_path / "data.parquet")
    expected_timestamp = "2026-01-02T03:04:05.123456+05:30"
    assert [json.loads(row) for row in table["metadata"].to_pylist()] == [
        {
            "timestamp": expected_timestamp,
            "sequence": 7,
            "active": True,
            "nested": [expected_timestamp, None],
        },
        {"sequence": 8},
    ]
    assert table["data"].to_pylist() == [value.buffers()[1].to_pybytes()] * 2
