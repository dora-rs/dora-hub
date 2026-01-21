# Camera Logging Example

Record high-bandwidth video streams using the dora-parquet recorder with proper synchronization.

## Prerequisites

Install the recorder package:
```bash
cd ../..
pip install -e .
```

Install OpenCV:
```bash
pip install opencv-python
```

## Running the Example

Start the dataflow:
```bash
dora start dataflow.yml --attach
```

## Important: The Handshake Pattern

To prevent data loss, **data source nodes must wait for the recorder's "READY" signal** before sending data.

### Why This Matters

The recorder node needs time to:
- Initialize the Parquet writer
- Set up batching buffers
- Prepare the output file

If your camera starts sending frames immediately, early frames will be lost before the recorder is ready.

### How It Works

1. Recorder node starts and sends a "READY" signal
2. Camera node waits for this signal (handshake)
3. Only after receiving "READY", the camera begins streaming frames
4. All 50 frames are captured without loss to `logs/cam_feed.parquet`

This handshake pattern is **critical for high-bandwidth data sources** like cameras, LiDAR, or sensors where missing initial data could corrupt your dataset.

## Verifying the Recording

Run the decoder to extract images and verify all frames were captured:
```bash
python decode_images.py
```

Check `extracted_photos/` for all 50 frames.