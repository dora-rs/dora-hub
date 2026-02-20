#!/usr/bin/env python3
"""Capture a RealSense depth frame and save as test fixture.

Tries xoq_realsense (remote via relay) first, falls back to pyrealsense2 (local).

Usage:
    python -m tests.capture_realsense_fixture [--path anon/realsense] [--output tests/fixtures/realsense_frame.npz]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np


IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720


def capture_xoq(relay_path: str):
    """Capture via xoq_realsense (remote relay)."""
    import xoq_realsense as rs

    pipeline = rs.pipeline()
    config = rs.config()
    # xoq_realsense uses the relay path as the device identifier
    config.enable_device(relay_path)
    config.enable_stream(rs.stream.depth, IMAGE_WIDTH, IMAGE_HEIGHT, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, IMAGE_WIDTH, IMAGE_HEIGHT, rs.format.rgb8, 30)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    try:
        # Skip frames for auto-exposure
        print("[capture] Warming up (10 frames)...")
        for _ in range(10):
            pipeline.wait_for_frames()

        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        # xoq_realsense uses get_intrinsics() method, not .intrinsics property
        vsp = depth_frame.profile.as_video_stream_profile()
        intr = vsp.get_intrinsics() if hasattr(vsp, "get_intrinsics") else vsp.intrinsics
        fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy
        actual_w, actual_h = intr.width, intr.height

        depth = np.asanyarray(depth_frame.get_data()).ravel().astype(np.uint16)
        color = np.asanyarray(color_frame.get_data()).astype(np.uint8)

        return depth, color, (fx, fy, cx, cy), (actual_w, actual_h)
    finally:
        pipeline.stop()


def capture_local():
    """Capture via pyrealsense2 (local USB)."""
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, IMAGE_WIDTH, IMAGE_HEIGHT, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, IMAGE_WIDTH, IMAGE_HEIGHT, rs.format.rgb8, 30)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    try:
        print(f"[capture] Warming up (30 frames)...")
        for _ in range(30):
            pipeline.wait_for_frames()

        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        intr = depth_frame.profile.as_video_stream_profile().intrinsics
        fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy
        actual_w, actual_h = intr.width, intr.height

        depth = np.asanyarray(depth_frame.get_data()).ravel().astype(np.uint16)
        color = np.asanyarray(color_frame.get_data()).astype(np.uint8)

        return depth, color, (fx, fy, cx, cy), (actual_w, actual_h)
    finally:
        pipeline.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="anon/realsense", help="xoq relay path")
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "fixtures" / "realsense_frame.npz"),
    )
    parser.add_argument("--local", action="store_true", help="Use local pyrealsense2")
    args = parser.parse_args()

    if args.local:
        print("[capture] Using local pyrealsense2...")
        depth, color, intrinsics, image_size = capture_local()
    else:
        print(f"[capture] Trying xoq_realsense relay: {args.path}")
        try:
            depth, color, intrinsics, image_size = capture_xoq(args.path)
        except Exception as e:
            print(f"[capture] xoq_realsense failed: {e}")
            print("[capture] Falling back to local pyrealsense2...")
            depth, color, intrinsics, image_size = capture_local()

    fx, fy, cx, cy = intrinsics
    w, h = image_size
    print(f"[capture] Frame captured: {w}x{h}")
    print(f"[capture] Intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

    depth_2d = depth.reshape(h, w)
    valid = depth_2d[depth_2d > 0]
    print(
        f"[capture] Depth: range={valid.min()}-{valid.max()}mm, "
        f"median={np.median(valid):.0f}mm, "
        f"valid={len(valid)}/{w*h} ({100*len(valid)/(w*h):.1f}%)"
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        depth=depth,
        color=color,
        intrinsics=np.array([fx, fy, cx, cy], dtype=np.float32),
        image_size=np.array([w, h], dtype=np.int32),
        camera_transform_str="-0.23 0.71 0.3 90 -45 0",
    )
    print(f"[capture] Saved to {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
