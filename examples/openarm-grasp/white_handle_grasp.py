#!/usr/bin/env python3
"""Detect a white handle and output grasp targets for grasp_trajectory.py.

Captures a color + depth frame from a RealSense camera, detects a white
handle via HSV filtering, and writes a targets.json file with the
detected jaw pixel coordinates in 0-1000 normalized format.

Usage:
    # Detect and save targets
    python white_handle_grasp.py --config openarm-config.json --camera champagne-realsense

    # Then plan a trajectory with the detected targets
    python grasp_trajectory.py --config openarm-config.json --targets output/white_handle_targets.json \\
        --camera champagne-realsense

    # Or do both in one shot (detect + plan + GIF)
    python white_handle_grasp.py --config openarm-config.json --camera champagne-realsense --plan
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# --- Constants ---
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
OUTPUT_DIR = Path(__file__).parent / "output"

# HSV thresholds for white handle detection
WHITE_S_MAX = 60   # low saturation
WHITE_V_MIN = 180  # high value (brightness)


# ---- Camera config (shared with grasp_trajectory.py) ----

def load_camera_config(config_path, camera_label=None):
    """Load camera transform and relay path from openarm config JSON."""
    with open(config_path) as f:
        cfg = json.load(f)

    cameras = cfg.get("realsense", [])
    if not cameras:
        print(f"[config] No realsense entries in {config_path}")
        return None

    entry = None
    if camera_label:
        for cam in cameras:
            if cam.get("label") == camera_label:
                entry = cam
                break
        if entry is None:
            labels = [c.get("label", "?") for c in cameras]
            print(f"[config] Camera '{camera_label}' not found. Available: {labels}")
            return None
    else:
        for cam in cameras:
            if cam.get("enabled", True):
                entry = cam
                break
        if entry is None:
            entry = cameras[0]

    pos = entry["position"]
    rot = entry["rotation"]
    transform_str = (f"{pos['x']} {pos['y']} {pos['z']} "
                     f"{rot['roll']} {rot['pitch']} {rot['yaw']}")
    relay = entry["path"]

    print(f"[config] Camera: {entry.get('label', '?')}")
    print(f"[config] Relay: {relay}")
    print(f"[config] Transform: {transform_str}")
    return transform_str, relay


# ---- RealSense capture ----

def capture_frames(relay_path, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    """Capture aligned color + depth via xoq_realsense relay.

    Returns (color_rgb, depth_flat, (fx, fy, cx, cy)).
    """
    import xoq_realsense as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(relay_path)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, 30)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    try:
        print("[realsense] Warming up (10 frames)...")
        for _ in range(10):
            pipeline.wait_for_frames()

        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)

        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()

        vsp = depth_frame.profile.as_video_stream_profile()
        intr = vsp.get_intrinsics()
        fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy

        depth_data = np.asanyarray(depth_frame.get_data()).ravel().astype(np.uint16)
        color_data = np.asanyarray(color_frame.get_data()).reshape(height, width, 3)

        print(f"[realsense] Frames: {width}x{height}")
        print(f"[realsense] Intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
        return color_data, depth_data, (fx, fy, cx, cy)
    finally:
        pipeline.stop()


# ---- Detection ----

def detect_white_handle(color_rgb, depth_flat, width=IMAGE_WIDTH, height=IMAGE_HEIGHT,
                        save_debug=True):
    """Detect a white handle via HSV + depth-edge segmentation.

    Returns (u1, v1, u2, v2) pixel coordinates for the two jaw contact
    points placed perpendicular to the handle's major axis, or None.
    """
    hsv = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2HSV)
    depth_2d = depth_flat.reshape(height, width).astype(np.float32)

    # White = low saturation, high value
    white_mask = cv2.inRange(hsv, (0, 0, WHITE_V_MIN), (180, WHITE_S_MAX, 255))

    # Depth validity mask
    depth_valid = (depth_2d > 150) & (depth_2d < 600)
    white_mask = cv2.bitwise_and(white_mask, white_mask,
                                  mask=depth_valid.astype(np.uint8) * 255)

    # Depth edges: large depth gradients indicate object boundaries
    depth_smooth = cv2.GaussianBlur(depth_2d, (5, 5), 0)
    grad_x = cv2.Sobel(depth_smooth, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_smooth, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    depth_edges = (grad_mag > 15).astype(np.uint8) * 255
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    depth_edges = cv2.dilate(depth_edges, edge_kernel, iterations=1)

    # Cut white mask along depth edges
    mask = cv2.bitwise_and(white_mask, cv2.bitwise_not(depth_edges))

    # ROI: center region
    roi_mask = np.zeros_like(mask)
    margin_x, margin_y = width // 8, height // 8
    roi_mask[margin_y:height - margin_y, margin_x:width - margin_x] = 255
    mask = cv2.bitwise_and(mask, roi_mask)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Save debug image
    if save_debug:
        debug = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR).copy()
        green_overlay = np.zeros_like(debug)
        green_overlay[:, :, 1] = mask
        red_overlay = np.zeros_like(debug)
        red_overlay[:, :, 2] = depth_edges
        debug = cv2.addWeighted(debug, 0.6, green_overlay, 0.25, 0)
        debug = cv2.addWeighted(debug, 1.0, red_overlay, 0.15, 0)
        for c in contours:
            area = cv2.contourArea(c)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect).astype(int)
            color = (0, 255, 255) if area >= 300 else (128, 128, 128)
            cv2.drawContours(debug, [box], 0, color, 2)
            (cx_r, cy_r), _, _ = rect
            cv2.putText(debug, f"{area:.0f}", (int(cx_r), int(cy_r)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(OUTPUT_DIR / "white_handle_debug.png"), debug)
        print(f"[detect] Debug image saved ({len(contours)} contours)")

    if not contours:
        print("[detect] No white regions found")
        return None

    # Filter and score contours
    MIN_AREA = 300
    MAX_AREA = 20000
    MIN_ASPECT = 1.8
    img_cx, img_cy = width / 2, height / 2

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA or area > MAX_AREA:
            continue
        rect = cv2.minAreaRect(c)
        (cx_r, cy_r), (w_r, h_r), angle = rect
        if min(w_r, h_r) < 1:
            continue
        aspect = max(w_r, h_r) / min(w_r, h_r)
        if aspect < MIN_ASPECT:
            continue
        dist_to_center = np.hypot(cx_r - img_cx, cy_r - img_cy)
        score = (area * aspect) / (1 + dist_to_center / 50)
        candidates.append((c, area, rect, aspect, score))
        print(f"[detect]   candidate: center=({cx_r:.0f},{cy_r:.0f}), "
              f"area={area:.0f}, aspect={aspect:.1f}, score={score:.0f}")

    if not candidates:
        # Relaxed fallback
        for c in contours:
            area = cv2.contourArea(c)
            if 200 <= area <= MAX_AREA * 2:
                rect = cv2.minAreaRect(c)
                (cx_r, cy_r), (w_r, h_r), _ = rect
                if min(w_r, h_r) < 1:
                    continue
                aspect = max(w_r, h_r) / min(w_r, h_r)
                dist = np.hypot(cx_r - img_cx, cy_r - img_cy)
                score = area / (1 + dist / 50)
                candidates.append((c, area, rect, aspect, score))
        if not candidates:
            print(f"[detect] No handle-shaped contours (found {len(contours)} total)")
            return None

    best = max(candidates, key=lambda x: x[4])
    _, area, rect, aspect, score = best
    (cx_r, cy_r), (w_r, h_r), angle = rect
    print(f"[detect] Selected: center=({cx_r:.0f},{cy_r:.0f}), "
          f"size=({w_r:.0f}x{h_r:.0f}), angle={angle:.1f}, "
          f"area={area:.0f}px, aspect={aspect:.1f}, score={score:.0f}")

    # Major axis along the longer dimension
    if w_r < h_r:
        major_angle_rad = np.deg2rad(angle + 90)
    else:
        major_angle_rad = np.deg2rad(angle)

    # Jaw pixels perpendicular to major axis
    perp_angle = major_angle_rad + np.pi / 2
    jaw_spread = min(w_r, h_r) * 0.5
    jaw_spread = max(jaw_spread, 15)

    u1 = cx_r + jaw_spread * np.cos(perp_angle)
    v1 = cy_r + jaw_spread * np.sin(perp_angle)
    u2 = cx_r - jaw_spread * np.cos(perp_angle)
    v2 = cy_r - jaw_spread * np.sin(perp_angle)

    u1 = np.clip(u1, 0, width - 1)
    v1 = np.clip(v1, 0, height - 1)
    u2 = np.clip(u2, 0, width - 1)
    v2 = np.clip(v2, 0, height - 1)

    print(f"[detect] Jaw pixels: ({u1:.0f},{v1:.0f}) ({u2:.0f},{v2:.0f})")
    return float(u1), float(v1), float(u2), float(v2)


def find_valid_grasp_region(depth, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    """Fallback: find closest object in center region and suggest jaw pixels."""
    depth_2d = depth.reshape(height, width)
    center = depth_2d[height // 4:3 * height // 4, width // 4:3 * width // 4]
    valid_mask = (center > 100) & (center < 2000)

    if not np.any(valid_mask):
        print("[fallback] No valid depth in center region (100-2000mm)")
        return None

    min_depth = center[valid_mask].min()
    obj_mask = valid_mask & (center < min_depth + 50)

    if np.sum(obj_mask) < 10:
        print(f"[fallback] Too few object pixels ({np.sum(obj_mask)})")
        return None

    ys, xs = np.where(obj_mask)
    xs = xs + width // 4
    ys = ys + height // 4

    cx_obj = int(np.median(xs))
    cy_obj = int(np.median(ys))
    spread = max(20, int(np.std(xs)))

    u1 = cx_obj - spread // 2
    u2 = cx_obj + spread // 2
    print(f"[fallback] Object center=({cx_obj},{cy_obj}), depth={min_depth}mm, spread={spread}px")
    return float(u1), float(cy_obj), float(u2), float(cy_obj)


def save_annotated_image(color_rgb, jaw_pixels, output_path):
    """Save color image with jaw points and grasp line overlaid."""
    img = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
    u1, v1, u2, v2 = jaw_pixels
    p1 = (int(round(u1)), int(round(v1)))
    p2 = (int(round(u2)), int(round(v2)))

    cv2.line(img, p1, p2, (0, 255, 0), 2)
    cv2.circle(img, p1, 8, (0, 0, 255), -1)
    cv2.circle(img, p2, 8, (255, 0, 0), -1)

    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    cv2.putText(img, "grasp", (mid[0] + 10, mid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    print(f"[save] Annotated image: {output_path}")


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(
        description="Detect white handle and output grasp targets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Output: output/white_handle_targets.json (feed to grasp_trajectory.py)

Example pipeline:
    python white_handle_grasp.py --config openarm-config.json --camera champagne-realsense
    python grasp_trajectory.py --config openarm-config.json \\
        --targets output/white_handle_targets.json --camera champagne-realsense
""",
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to openarm config JSON")
    parser.add_argument("--camera", type=str, default=None,
                        help="Camera label in config")
    parser.add_argument("--plan", action="store_true",
                        help="Also run grasp_trajectory.py after detection")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device for trajectory planning (with --plan)")
    parser.add_argument("--no-gif", action="store_true",
                        help="Skip GIF rendering (with --plan)")
    args = parser.parse_args()

    # Resolve camera
    relay_path = None
    if args.config:
        result = load_camera_config(args.config, args.camera)
        if result:
            _, relay_path = result
        else:
            print("ERROR: Config load failed")
            return
    else:
        # Default relay
        relay_path = "anon/a13af1d39199/realsense-233522074606"
        print(f"[config] No --config provided, using default relay: {relay_path}")

    print("\n=== White Handle Detection ===")

    # 1. Capture
    print("\n[1/3] Capturing from RealSense...")
    color_rgb, depth, intrinsics = capture_frames(relay_path)

    # 2. Detect
    print("\n[2/3] Detecting white handle...")
    jaw = detect_white_handle(color_rgb, depth)
    if jaw is None:
        print("  White detection failed, trying depth-based fallback...")
        jaw = find_valid_grasp_region(depth)
        if jaw is None:
            print("  ERROR: No graspable object found")
            return
    u1, v1, u2, v2 = jaw

    # Save annotated image
    save_annotated_image(color_rgb, jaw, OUTPUT_DIR / "white_handle_grasp.png")

    # 3. Output targets.json
    print("\n[3/3] Writing targets...")
    p1_norm = [int(round(u1 * 1000 / IMAGE_WIDTH)), int(round(v1 * 1000 / IMAGE_HEIGHT))]
    p2_norm = [int(round(u2 * 1000 / IMAGE_WIDTH)), int(round(v2 * 1000 / IMAGE_HEIGHT))]

    targets = {
        "targets": [{
            "label": "white_handle",
            "p1": p1_norm,
            "p2": p2_norm,
        }]
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    targets_path = OUTPUT_DIR / "white_handle_targets.json"
    with open(targets_path, "w") as f:
        json.dump(targets, f, indent=2)
    print(f"  Targets saved: {targets_path}")
    print(f"  p1={p1_norm}, p2={p2_norm}")

    # Print next step
    config_arg = f" --config {args.config}" if args.config else ""
    camera_arg = f" --camera {args.camera}" if args.camera else ""
    print(f"\nNext step:")
    print(f"  python grasp_trajectory.py{config_arg} --targets {targets_path}{camera_arg}")

    # Optionally run planning
    if args.plan:
        import subprocess
        import sys

        cmd = [
            sys.executable, str(Path(__file__).parent / "grasp_trajectory.py"),
            "--targets", str(targets_path),
            "--device", args.device,
        ]
        if args.config:
            cmd += ["--config", args.config]
        if args.camera:
            cmd += ["--camera", args.camera]
        if args.no_gif:
            cmd += ["--no-gif"]
        print(f"\n{'='*50}")
        print(f"Running: {' '.join(cmd)}")
        print(f"{'='*50}\n")
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
