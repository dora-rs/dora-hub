"""High-Performance Batched Parquet Recorder"""

import json
import os
import queue
import threading
from datetime import datetime
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from dora import Node

# CONFIGURATION
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "30"))
LOG_DIR = os.getenv("LOG_DIR", "data_logs")

class DoraParquetRecorder:
    def __init__(self):
        self.write_queue = queue.Queue()
        self.writers = {}
        self.shutdown_flag = False

        os.makedirs(LOG_DIR, exist_ok=True)

        # Start the background writer
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()
        print(f"[Recorder] Online. Batch Size: {BATCH_SIZE}", flush=True)

    def _writer_loop(self):
        """Collects small tables and writes them in big chunks.
        """
        # Buffer to hold tables for each input_id: { "cam_feed": [table1, table2...] }
        buffers = {}

        while not self.shutdown_flag or not self.write_queue.empty():
            try:
                # 1. Get data (Wait up to 0.1s so we can check shutdown flag often)
                data = self.write_queue.get(timeout=0.1)
                input_id, table = data

                # 2. Add to local buffer
                if input_id not in buffers:
                    buffers[input_id] = []
                buffers[input_id].append(table)

                # 3. Check if bucket is full
                if len(buffers[input_id]) >= BATCH_SIZE:
                    self._flush_buffer(input_id, buffers[input_id])
                    buffers[input_id] = [] # Empty the bucket

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Recorder] Write error: {e}", flush=True)

        # FINAL CLEANUP: Flush whatever is left in the buckets
        print("[Recorder] Flushing remaining data...", flush=True)
        for input_id, buf in buffers.items():
            if buf:
                self._flush_buffer(input_id, buf)

        # Close files
        for w in self.writers.values():
            w.close()

    def _flush_buffer(self, input_id, table_list):
        """Merges small tables into one big table and writes it."""
        try:
            if not table_list:
                return

            # Combine 30 small tables into 1 big table (Very fast)
            batch_table = pa.concat_tables(table_list)

            # Create writer if it doesn't exist
            if input_id not in self.writers:
                file_path = os.path.join(LOG_DIR, f"{input_id}.parquet")
                # 'compression=None' is faster for CPU, 'snappy' saves disk space
                self.writers[input_id] = pq.ParquetWriter(
                    file_path,
                    batch_table.schema,
                    compression='NONE')
                print(f"[Recorder] Created log: {file_path}", flush=True)

            # One single write for 30 frames!
            self.writers[input_id].write_table(batch_table)

        except Exception as e:
            print(f"[Recorder] Flush failed: {e}", flush=True)

    def handle_input(self, input_id: str, value: Any, metadata: Any):
        if self.shutdown_flag:
            return

        try:
            # 1. Fast Metadata Serialize
            meta_json = json.dumps(metadata)

            # 2. Fast Binary Copy (Zero-Copyish)
            # Try to get raw C-buffer bytes if possible
            if hasattr(value, "buffers"):
                try:
                    data_blob = value.buffers()[1].to_pybytes()
                except Exception:
                    data_blob = value.to_string().encode('utf-8')
            else:
                # Fallback for strings/other types
                if not isinstance(value, (pa.Array, pa.ChunkedArray)):
                    value = pa.array([value])
                data_blob = value.to_pylist()[0] # Fallback (slower but safe)

            # 3. Queue it up
            timestamp = datetime.now().isoformat()

            table = pa.Table.from_pydict({
                "timestamp": [timestamp],
                "data": [data_blob],
                "metadata": [meta_json]
            })

            self.write_queue.put((input_id, table))

        except Exception as e:
            print(f"[Recorder] Serialize error: {e}", flush=True)

    def _shutdown(self):
        self.shutdown_flag = True
        if self.writer_thread.is_alive():
            self.writer_thread.join(timeout=5.0)

def main():
    node = Node()
    recorder = DoraParquetRecorder()

    # --- HANDSHAKE ---
    print("[Recorder] Ready. Sending Signal...", flush=True)
    node.send_output("status", pa.array(["READY"]))
    # -----------------

    for event in node:
        if event["type"] == "INPUT":
            recorder.handle_input(
                event["id"], event["value"], event.get("metadata", {})
            )
        elif event["type"] == "STOP":
            break

    recorder._shutdown()

if __name__ == "__main__":
    main()
