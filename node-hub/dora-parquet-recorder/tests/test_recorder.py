from decimal import Decimal

import pyarrow as pa
import pyarrow.parquet as pq

from dora_parquet_recorder import main
from dora_parquet_recorder.main import raw_value_bytes


def test_import():
    from dora_parquet_recorder.main import DoraParquetRecorder
    assert DoraParquetRecorder is not None


def test_string_records_text_not_offsets():
    assert raw_value_bytes(pa.array(["hello"])) == b"hello"


def test_large_string_and_binary():
    assert raw_value_bytes(pa.array(["hello"], type=pa.large_string())) == b"hello"
    assert raw_value_bytes(pa.array([b"\x01\x02"], type=pa.binary())) == b"\x01\x02"


def test_sliced_string_only_takes_the_slice():
    value = pa.array(["ab", "cde", "f", "ghij"])[1:3]
    assert raw_value_bytes(value) == b"cdef"


def test_sliced_primitive_only_takes_the_slice():
    value = pa.array([1, 2, 3, 4], type=pa.int32())[1:3]
    expected = pa.array([2, 3], type=pa.int32()).buffers()[1].to_pybytes()
    assert raw_value_bytes(value) == expected


def test_empty_strings():
    assert raw_value_bytes(pa.array([], type=pa.string())) == b""
    assert raw_value_bytes(pa.array(["", ""])) == b""


def test_fixed_size_binary_and_decimal_stay_raw_bytes():
    value = pa.array([b"ab", b"cd", b"ef"], type=pa.binary(2))[1:3]
    assert raw_value_bytes(value) == b"cdef"
    value = pa.array([Decimal("1.23")], type=pa.decimal128(5, 2))
    assert raw_value_bytes(value) == value.buffers()[1].to_pybytes()


def test_types_without_a_single_value_buffer():
    assert raw_value_bytes(pa.array([True, False])) is None
    assert raw_value_bytes(pa.array([[1, 2]])) is None


def test_recorded_file_contains_the_string(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "LOG_DIR", str(tmp_path))
    recorder = main.DoraParquetRecorder()
    recorder.handle_input("text", pa.array(["hello"]), {})
    recorder._shutdown()

    table = pq.read_table(tmp_path / "text.parquet")
    assert table.column("data").to_pylist() == [b"hello"]
