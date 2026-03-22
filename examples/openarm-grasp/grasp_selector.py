"""VLM-as-critic grasp selector with SAM2 segmentation and contour-based candidates.

Pipeline:
  Pass 1 (locate): Ask VLM for the best grip point on the target object.
  Segment: Send point to SAM2 for precise mask (fallback: depth/color).
  Contour analysis: Sample grasp lines through the mask, score by geometric
      contact quality, pick ~5 diverse candidates.
  Pass 2 (rate xN): For each candidate, rotate the image so the gripper appears
      horizontal, render it, ask VLM to rate 1-10. Pick the highest-rated.

Key insight: Qwen's vision encoder uses a grid of patches, so it perceives
horizontal/vertical patterns much better than diagonal ones. By rotating the
image, every candidate gripper appears horizontal to the VLM.

Outputs grasp result as {"p1":[x,y],"p2":[x,y]} in normalized 0-1000 coords,
compatible with test_visualize_grasp.py.
"""

import base64
import json
import math
import os
import re
from dataclasses import dataclass, field

import cv2
import numpy as np
import pyarrow as pa
from dora import Node

# --- Config from env ---
WIDTH = int(os.getenv("IMAGE_WIDTH", "640"))
HEIGHT = int(os.getenv("IMAGE_HEIGHT", "480"))
TARGET_OBJECT = os.getenv("TARGET_OBJECT", "the red cube")
USE_THINK = os.getenv("USE_THINK", "true").lower() in ("1", "true", "yes")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", ".")
DEFAULT_DEPTH_MM = float(os.getenv("DEFAULT_DEPTH_MM", "400"))
NUM_CANDIDATES = int(os.getenv("NUM_CANDIDATES", "5"))
EARLY_STOP_SCORE = int(os.getenv("EARLY_STOP_SCORE", "8"))  # stop rating if a candidate scores >= this
DEPTH_THRESHOLD_MM = float(os.getenv("DEPTH_THRESHOLD_MM", "30"))
USE_SAM2 = os.getenv("USE_SAM2", "true").lower() in ("1", "true", "yes")
USE_SAM3 = os.getenv("USE_SAM3", "false").lower() in ("1", "true", "yes")
PLACE_CONTAINER = os.getenv("PLACE_CONTAINER", "")  # e.g. "the pan" — enables place detection
SKIP_VLM_RATING = os.getenv("SKIP_VLM_RATING", "false").lower() in ("1", "true", "yes")

# Gripper physical dimensions (SO-100 / similar small gripper)
GRIPPER_MAX_OPENING_MM = 100.0  # full opening
GRIPPER_MIN_OPENING_MM = 5.0    # minimum useful opening
FINGER_THICKNESS_MM = 10.0
FINGER_DEPTH_MM = 30.0

# Color for the gripper overlay (green, in RGB)
GRIPPER_COLOR_RGB = (0, 220, 0)

# --- Prompts ---
LOCATE_PROMPT_TEMPLATE = (
    "Detect {object_name} in this image with a bounding box. "
    'Output JSON: {{"bbox_2d": [x_min, y_min, x_max, y_max], "label": "{object_name}"}}'
)

RATE_PROMPT_TEMPLATE = (
    "You are a robot grasp quality evaluator. This image shows {object_name} "
    "with a parallel-jaw gripper overlaid in green. The image has been rotated "
    "so the gripper appears horizontal. The two green rectangles are the finger "
    "pads that will close on the object along the dashed green line.\n"
    "Rate this grasp from 1 to 10 considering:\n"
    "- Are the fingers placed on graspable surfaces of the object?\n"
    "- Is the grip across a stable axis?\n"
    "- Will the fingers have good contact area?\n"
    "- Will the grasp be stable when lifting?\n"
    'Output ONLY JSON: {{"score": <1-10>, "reason": "<one sentence>"}}'
)

PLACE_LOCATE_PROMPT_TEMPLATE = (
    "Detect {container_name} in this image with a bounding box. "
    'Output JSON: {{"bbox_2d": [x_min, y_min, x_max, y_max], "label": "{container_name}"}}'
)

LOCATE_BOTH_PROMPT_TEMPLATE = (
    "Detect two objects in this image with bounding boxes:\n"
    "1. {pick_name}\n"
    "2. {place_name}\n"
    'Output JSON: {{"pick": {{"bbox_2d": [x_min, y_min, x_max, y_max]}}, '
    '"place": {{"bbox_2d": [x_min, y_min, x_max, y_max]}}}}'
)


@dataclass
class GraspCandidate:
    angle_deg: float
    # Grasp center in pixels
    center_x: float
    center_y: float
    # Jaw center positions in pixels
    jaw1_cx: float
    jaw1_cy: float
    jaw2_cx: float
    jaw2_cy: float
    # Pixel dimensions for rendering
    opening_px: float  # full opening (distance between jaw centers)
    finger_thickness_px: float
    finger_depth_px: float
    # Geometric score (for pre-filtering before VLM)
    geo_score: float = 0.0


# --- Helper functions ---


def strip_think_tags(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def send_vlm_request(node, image_rgb, prompt_text):
    """Encode image + prompt in dora-qwen-omni format and send."""
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _, png_bytes = cv2.imencode(".png", img_bgr)
    b64 = base64.b64encode(png_bytes.tobytes()).decode("utf-8")
    image_url = f"data:image/png;base64,{b64}"

    full_prompt = f"/think\n{prompt_text}" if USE_THINK else prompt_text
    texts = [
        f"<|user|>\n<|vision_start|>\n{image_url}",
        f"<|user|>\n<|im_start|>\n{full_prompt}",
    ]
    node.send_output("vlm_request", pa.array(texts))


def parse_locate_response(text):
    """Extract center + optional bbox from VLM response -> (cx_px, cy_px) or None.

    Handles both new bbox format and legacy center-point format:
      {"x_min": N, "y_min": N, "x_max": N, "y_max": N}   — pixel bbox (preferred)
      {"cx": N, "cy": N}                                    — legacy 0-1000 normalized
    Returns (cx_px, cy_px) in pixel coordinates, or None.
    Also sets last_vlm_bbox global when a bbox is parsed.
    """
    global last_vlm_bbox
    last_vlm_bbox = None
    text = strip_think_tags(text)
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        data = json.loads(text[start:end])
        result = _parse_bbox_or_center(data)
        if result is None:
            return None
        cx, cy, bbox = result
        if cx < 0 or cy < 0:
            return cx, cy
        last_vlm_bbox = bbox
        return cx, cy
    except (json.JSONDecodeError, KeyError, TypeError, ValueError, IndexError):
        return None


# Global to store the last VLM-returned bounding box (set by parse_locate_response)
last_vlm_bbox = None


def _parse_bbox_or_center(data):
    """Parse a single object dict — bbox or legacy center point.
    Returns (cx_px, cy_px, bbox_or_None) where bbox = (x_min, y_min, x_max, y_max)."""
    if not isinstance(data, dict):
        return None
    try:
        # bbox_2d format: [x_min, y_min, x_max, y_max] (Qwen3-VL native)
        if "bbox_2d" in data and isinstance(data["bbox_2d"], list) and len(data["bbox_2d"]) >= 4:
            b = data["bbox_2d"]
            x_min_n, y_min_n, x_max_n, y_max_n = float(b[0]), float(b[1]), float(b[2]), float(b[3])
            if x_min_n < 0 or y_min_n < 0:
                return x_min_n, y_min_n, None
            x_min = max(0, min(x_min_n * WIDTH / 1000.0, WIDTH - 1))
            y_min = max(0, min(y_min_n * HEIGHT / 1000.0, HEIGHT - 1))
            x_max = max(0, min(x_max_n * WIDTH / 1000.0, WIDTH - 1))
            y_max = max(0, min(y_max_n * HEIGHT / 1000.0, HEIGHT - 1))
            cx = (x_min + x_max) / 2.0
            cy = (y_min + y_max) / 2.0
            return cx, cy, (x_min, y_min, x_max, y_max)

        # Bounding box format (0-1000 normalized, converted to pixel)
        if "x_min" in data and "y_min" in data and "x_max" in data and "y_max" in data:
            x_min_n = float(data["x_min"])
            y_min_n = float(data["y_min"])
            x_max_n = float(data["x_max"])
            y_max_n = float(data["y_max"])
            if x_min_n < 0 or y_min_n < 0:
                return x_min_n, y_min_n, None
            x_min = max(0, min(x_min_n * WIDTH / 1000.0, WIDTH - 1))
            y_min = max(0, min(y_min_n * HEIGHT / 1000.0, HEIGHT - 1))
            x_max = max(0, min(x_max_n * WIDTH / 1000.0, WIDTH - 1))
            y_max = max(0, min(y_max_n * HEIGHT / 1000.0, HEIGHT - 1))
            cx = (x_min + x_max) / 2.0
            cy = (y_min + y_max) / 2.0
            return cx, cy, (x_min, y_min, x_max, y_max)

        # Legacy center-point format (0-1000 normalized)
        cx_norm, cy_norm = None, None
        if "cx" in data and "cy" in data:
            cx_raw, cy_raw = data["cx"], data["cy"]
            cx_norm = float(cx_raw[0]) if isinstance(cx_raw, list) else float(cx_raw)
            cy_norm = float(cy_raw[0]) if isinstance(cy_raw, list) else float(cy_raw)
        elif "x" in data and "y" in data:
            cx_norm, cy_norm = float(data["x"]), float(data["y"])
        elif "center" in data and isinstance(data["center"], list):
            cx_norm, cy_norm = float(data["center"][0]), float(data["center"][1])

        if cx_norm is None or cy_norm is None:
            return None
        if cx_norm < 0 or cy_norm < 0:
            return cx_norm, cy_norm, None
        cx_px = max(0.0, min(cx_norm * WIDTH / 1000.0, WIDTH - 1))
        cy_px = max(0.0, min(cy_norm * HEIGHT / 1000.0, HEIGHT - 1))
        return cx_px, cy_px, None
    except (TypeError, ValueError, IndexError):
        return None


def parse_locate_both_response(text):
    """Parse VLM response with both pick and place bboxes/centers.

    Handles both new bbox format and legacy center-point format.
    Returns (pick_px, place_px) where each is (cx_px, cy_px), or None on failure.
    Also sets last_vlm_bbox and last_vlm_place_bbox globals.
    """
    global last_vlm_bbox, last_vlm_place_bbox
    last_vlm_bbox = None
    last_vlm_place_bbox = None
    text = strip_think_tags(text)
    # Remove markdown code fences if present
    text = text.replace("```json", "").replace("```", "").strip()

    # Try parsing as JSON — could be object or array
    start = text.find("[") if text.find("[") >= 0 and (text.find("{") < 0 or text.find("[") < text.find("{")) else text.find("{")
    if start < 0:
        return None
    end = max(text.rfind("]"), text.rfind("}")) + 1
    if end <= start:
        return None
    try:
        data = json.loads(text[start:end])
    except (json.JSONDecodeError, ValueError):
        # VLM sometimes returns {{"bbox_2d":...}, {"bbox_2d":...}} (invalid JSON)
        # Fix by replacing outer {} with [] to make it a valid array
        snippet = text[start:end].strip()
        if snippet.startswith("{") and snippet.endswith("}") and snippet.count('{"bbox_2d"') >= 2:
            fixed = "[" + snippet[1:-1] + "]"
            try:
                data = json.loads(fixed)
            except (json.JSONDecodeError, ValueError):
                return None
        else:
            return None

    # Handle array format: [{"bbox_2d": [...], "label": "sausage"}, {"bbox_2d": [...], "label": "pan"}]
    if isinstance(data, list):
        pick_result = None
        place_result = None
        for item in data:
            if not isinstance(item, dict):
                continue
            result = _parse_bbox_or_center(item)
            if result is None:
                continue
            # First item = pick, second = place (by order)
            if pick_result is None:
                pick_result = result
            elif place_result is None:
                place_result = result
        if pick_result is None:
            return None
        pick_cx, pick_cy, pick_bbox = pick_result
        last_vlm_bbox = pick_bbox
        place_px = None
        if place_result is not None:
            place_cx, place_cy, place_bbox = place_result
            place_px = (place_cx, place_cy)
            last_vlm_place_bbox = place_bbox
        return (pick_cx, pick_cy), place_px

    # Handle object format: {"pick": {...}, "place": {...}} or {"pick": [...], "place": [...]}
    if not isinstance(data, dict):
        return None
    pick_data = data.get("pick")
    if pick_data is None:
        return None
    # Handle bare array: {"pick": [x_min, y_min, x_max, y_max]}
    if isinstance(pick_data, list) and len(pick_data) == 4:
        pick_data = {"bbox_2d": pick_data}
    if not isinstance(pick_data, dict):
        return None
    pick_result = _parse_bbox_or_center(pick_data)
    if pick_result is None:
        return None
    pick_cx, pick_cy, pick_bbox = pick_result
    last_vlm_bbox = pick_bbox

    place_data = data.get("place")
    place_px = None
    if isinstance(place_data, list) and len(place_data) == 4:
        place_data = {"bbox_2d": place_data}
    if isinstance(place_data, dict):
        place_result = _parse_bbox_or_center(place_data)
        if place_result is not None:
            place_cx, place_cy, place_bbox = place_result
            place_px = (place_cx, place_cy)
            last_vlm_place_bbox = place_bbox

    return (pick_cx, pick_cy), place_px


def parse_rate_response(text):
    """Extract {"score": N, "reason": "..."} -> (score, reason) or None."""
    text = strip_think_tags(text)
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        data = json.loads(text[start:end])
        score = int(data["score"])
        reason = str(data.get("reason", ""))
        if 1 <= score <= 10:
            return score, reason
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        pass
    return None


def compute_gripper_pixel_scale(cx_px, cy_px, depth_map, intrinsics):
    """Compute mm-to-pixel scale at the object center. Falls back to defaults."""
    z_mm = DEFAULT_DEPTH_MM
    fx = None

    if depth_map is not None and intrinsics is not None:
        ix = max(0, min(int(round(cx_px)), depth_map.shape[1] - 1))
        iy = max(0, min(int(round(cy_px)), depth_map.shape[0] - 1))
        # Sample a patch around the center for robust depth estimate
        patch_r = 10
        y0 = max(0, iy - patch_r)
        y1 = min(depth_map.shape[0], iy + patch_r + 1)
        x0 = max(0, ix - patch_r)
        x1 = min(depth_map.shape[1], ix + patch_r + 1)
        patch = depth_map[y0:y1, x0:x1].astype(np.float32)
        valid = patch[(patch > 50) & (patch < 5000)]  # ignore noise (<50mm) and far (>5m)
        if len(valid) > 0:
            z_mm = float(np.median(valid))
        fx = intrinsics[0]

    if fx is None:
        fx = 600.0

    print(f"  Depth at ({cx_px:.0f},{cy_px:.0f}): z={z_mm:.0f}mm, fx={fx:.0f}, mm_to_px={fx/z_mm:.3f}")
    return fx / z_mm


# --- Segmentation ---


def segment_object_depth(depth_map, cx_px, cy_px, threshold_mm=DEPTH_THRESHOLD_MM):
    """Segment the object using depth thresholding around the located point.

    Returns a binary mask (uint8, 0 or 255).
    """
    ix = max(0, min(int(round(cx_px)), depth_map.shape[1] - 1))
    iy = max(0, min(int(round(cy_px)), depth_map.shape[0] - 1))

    # Sample a small patch around the center to get robust depth estimate
    patch_r = 5
    y0 = max(0, iy - patch_r)
    y1 = min(depth_map.shape[0], iy + patch_r + 1)
    x0 = max(0, ix - patch_r)
    x1 = min(depth_map.shape[1], ix + patch_r + 1)
    patch = depth_map[y0:y1, x0:x1].astype(np.float32)
    valid = patch[patch > 0]
    if len(valid) == 0:
        return None
    center_depth = float(np.median(valid))

    # Threshold: pixels within threshold_mm of center depth
    depth_f = depth_map.astype(np.float32)
    mask = ((depth_f > 0) &
            (np.abs(depth_f - center_depth) < threshold_mm)).astype(np.uint8) * 255

    # Keep only the connected component containing the center point
    num_labels, labels = cv2.connectedComponents(mask)
    center_label = labels[iy, ix]
    if center_label == 0:
        # Center fell on background; find nearest foreground label
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            return None
        dists = (ys - iy) ** 2 + (xs - ix) ** 2
        nearest = np.argmin(dists)
        center_label = labels[ys[nearest], xs[nearest]]

    component_mask = (labels == center_label).astype(np.uint8) * 255

    # Clean up with morphological ops
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    component_mask = cv2.morphologyEx(component_mask, cv2.MORPH_CLOSE, kernel)
    component_mask = cv2.morphologyEx(component_mask, cv2.MORPH_OPEN, kernel)

    return component_mask


def segment_object_color(image_rgb, cx_px, cy_px):
    """Fallback segmentation using color similarity when no depth available.

    Uses flood-fill from the center point with adaptive tolerance.
    Returns a binary mask (uint8, 0 or 255).
    """
    img_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2Lab)
    ix = max(0, min(int(round(cx_px)), WIDTH - 1))
    iy = max(0, min(int(round(cy_px)), HEIGHT - 1))

    # Flood fill with tolerance in Lab space
    h, w = img_lab.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    tolerance = (25, 25, 25)
    cv2.floodFill(img_lab, flood_mask, (ix, iy), 0,
                  loDiff=tolerance, upDiff=tolerance,
                  flags=cv2.FLOODFILL_MASK_ONLY | (255 << 8))

    mask = flood_mask[1:-1, 1:-1]  # remove the 1-pixel border

    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


# --- Contour-based grasp candidate generation ---


def compute_contour_normal(contour, idx, window=3):
    """Compute the inward-pointing normal at contour point idx."""
    n = len(contour)
    # Tangent from finite differences
    i_prev = (idx - window) % n
    i_next = (idx + window) % n
    p_prev = contour[i_prev][0].astype(float)
    p_next = contour[i_next][0].astype(float)
    tangent = p_next - p_prev
    length = np.linalg.norm(tangent)
    if length < 1e-6:
        return None
    tangent /= length
    # Normal: rotate tangent 90° clockwise (points inward for CCW contours)
    normal = np.array([tangent[1], -tangent[0]])
    return normal


def sample_grasp_candidates_from_contour(mask, mm_to_px, n_samples=40):
    """Sample grasp candidates by casting rays through the object mask.

    For each sample point on the contour:
    - Compute the inward normal
    - Cast a ray along the normal through the mask
    - Find entry and exit points (the two jaw contact points)
    - Score by: antipodal quality (surface parallelism) + balance
      (proximity to mask centroid for stable lifting)

    Returns a list of (center_x, center_y, angle_deg, width_px, geo_score).
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []

    # Use the largest contour
    contour = max(contours, key=cv2.contourArea)
    n_pts = len(contour)
    if n_pts < 20:
        return []

    # Compute mask centroid for balance scoring
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return []
    centroid_x = float(np.mean(xs))
    centroid_y = float(np.mean(ys))

    # Characteristic size: half-diagonal of bounding box
    bbox_diag = math.sqrt((xs.max() - xs.min()) ** 2 + (ys.max() - ys.min()) ** 2)
    if bbox_diag < 1:
        bbox_diag = 1.0

    max_opening_px = GRIPPER_MAX_OPENING_MM * mm_to_px
    min_opening_px = GRIPPER_MIN_OPENING_MM * mm_to_px

    raw_candidates = []
    step = max(1, n_pts // n_samples)

    for i in range(0, n_pts, step):
        pt = contour[i][0].astype(float)
        normal = compute_contour_normal(contour, i)
        if normal is None:
            continue

        # Cast ray inward along normal to find the opposite contour crossing
        # Walk pixel by pixel along the normal direction
        max_dist = int(max_opening_px * 1.5)
        entered = False
        exit_pt = None
        entry_pt = pt.copy()

        for d in range(1, max_dist):
            px = int(round(pt[0] + normal[0] * d))
            py = int(round(pt[1] + normal[1] * d))
            if px < 0 or px >= WIDTH or py < 0 or py >= HEIGHT:
                break
            if mask[py, px] > 0:
                entered = True
            elif entered:
                # We've exited the mask — this is the opposite contact point
                exit_pt = np.array([px, py], dtype=float)
                break

        if exit_pt is None:
            continue

        # Compute width (distance between contact points)
        width_px = np.linalg.norm(exit_pt - entry_pt)
        if width_px < min_opening_px or width_px > max_opening_px:
            continue

        # Grasp center and angle
        center = (entry_pt + exit_pt) / 2.0
        direction = exit_pt - entry_pt
        angle_rad = math.atan2(direction[1], direction[0])
        angle_deg = math.degrees(angle_rad)

        # --- Antipodal score ---
        # How parallel are the surfaces at the two contact points?
        dists = np.linalg.norm(contour[:, 0, :].astype(float) - exit_pt, axis=1)
        exit_idx = np.argmin(dists)
        exit_normal = compute_contour_normal(contour, exit_idx)

        if exit_normal is not None:
            antipodal = float(np.dot(normal, -exit_normal))
            antipodal = max(0.0, antipodal)
        else:
            antipodal = 0.5

        # --- Balance score ---
        # How close is the grasp center to the mask centroid?
        # Normalized by object size so it's scale-invariant.
        # 1.0 = at centroid, decays with distance
        dist_to_centroid = math.sqrt(
            (center[0] - centroid_x) ** 2 + (center[1] - centroid_y) ** 2
        )
        balance = max(0.0, 1.0 - dist_to_centroid / (bbox_diag * 0.5))

        # Combined geo_score: 60% antipodal + 40% balance
        geo_score = 0.6 * antipodal + 0.4 * balance

        raw_candidates.append((
            float(center[0]), float(center[1]),
            angle_deg, width_px, geo_score
        ))

    print(f"  Centroid: ({centroid_x:.0f}, {centroid_y:.0f}), bbox_diag: {bbox_diag:.0f}px")
    return raw_candidates


def select_diverse_candidates(raw_candidates, mm_to_px, n_select=NUM_CANDIDATES):
    """Select the top N most diverse candidates from the raw list.

    Strategy: sort by geo_score, then greedily pick candidates that are
    sufficiently different in position and angle from already selected ones.
    """
    if not raw_candidates:
        return []

    # Sort by geo_score descending
    sorted_cands = sorted(raw_candidates, key=lambda c: -c[4])

    selected = []

    for cx, cy, angle_deg, width_px, geo_score in sorted_cands:
        # Normalize angle to [0, 180) — parallel-jaw gripper is symmetric
        norm_angle = angle_deg % 180

        # Check diversity: use a combined distance so candidates can be
        # close in position if they differ in angle, and vice versa
        too_close = False
        for sel in selected:
            pos_dist = math.sqrt((cx - sel.center_x) ** 2 + (cy - sel.center_y) ** 2)
            sel_norm = sel.angle_deg % 180
            a_diff = abs(norm_angle - sel_norm)
            if a_diff > 90:
                a_diff = 180 - a_diff

            # Weighted combination: must be far enough in the combined space
            # pos_dist normalized by 20px, angle by 20deg
            combined = (pos_dist / 20.0) + (a_diff / 20.0)
            if combined < 1.5:
                too_close = True
                break

        if too_close:
            continue

        half_opening = width_px / 2.0
        angle_rad = math.radians(angle_deg)
        dx = math.cos(angle_rad) * half_opening
        dy = math.sin(angle_rad) * half_opening

        # Scale finger dimensions proportionally to the grasp opening
        # so the rendering stays sensible regardless of mm_to_px accuracy
        finger_thickness_px = max(width_px * 0.25, 3.0)
        finger_depth_px = max(width_px * 0.6, 8.0)

        selected.append(GraspCandidate(
            angle_deg=angle_deg,
            center_x=cx, center_y=cy,
            jaw1_cx=cx + dx, jaw1_cy=cy + dy,
            jaw2_cx=cx - dx, jaw2_cy=cy - dy,
            opening_px=width_px,
            finger_thickness_px=finger_thickness_px,
            finger_depth_px=finger_depth_px,
            geo_score=geo_score,
        ))

        if len(selected) >= n_select:
            break

    return selected


def debug_draw_candidates_on_mask(mask, image_rgb, raw_candidates, candidates, output_path):
    """Draw contour, raw grasp rays, and selected candidates on the image for debugging."""
    debug = image_rgb.copy()

    # Draw mask contour in cyan
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        cv2.drawContours(debug, contours, -1, (0, 255, 255), 2)

    # Draw raw candidate rays as thin gray lines
    for cx, cy, angle_deg, width_px, geo in raw_candidates[:20]:
        angle_rad = math.radians(angle_deg)
        hw = width_px / 2.0
        dx, dy = math.cos(angle_rad) * hw, math.sin(angle_rad) * hw
        p1 = (int(cx - dx), int(cy - dy))
        p2 = (int(cx + dx), int(cy + dy))
        cv2.line(debug, p1, p2, (128, 128, 128), 1)

    # Draw selected candidates with colored thick lines + jaw positions
    colors = [(255, 0, 0), (0, 200, 0), (0, 100, 255), (220, 220, 0), (200, 0, 200)]
    for i, c in enumerate(candidates):
        color = colors[i % len(colors)]
        j1 = (int(c.jaw1_cx), int(c.jaw1_cy))
        j2 = (int(c.jaw2_cx), int(c.jaw2_cy))
        cv2.line(debug, j1, j2, color, 3)
        cv2.circle(debug, j1, 5, color, -1)
        cv2.circle(debug, j2, 5, color, -1)
        cv2.circle(debug, (int(c.center_x), int(c.center_y)), 3, (255, 255, 255), -1)
        cv2.putText(debug, f"#{i+1} {c.angle_deg:.0f}d g={c.geo_score:.2f}",
                    (j1[0] + 8, j1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.imwrite(output_path, cv2.cvtColor(debug, cv2.COLOR_RGB2BGR))
    print(f"  Saved debug candidates: {output_path}")


def generate_candidates_from_mask(mask, mm_to_px, image_rgb=None, output_dir=None, fc=0):
    """Full pipeline: contour analysis → diverse candidate selection."""
    raw = sample_grasp_candidates_from_contour(mask, mm_to_px)
    print(f"  Raw contour candidates: {len(raw)}")
    candidates = select_diverse_candidates(raw, mm_to_px)
    print(f"  Selected diverse candidates: {len(candidates)}")
    for i, c in enumerate(candidates):
        print(f"    #{i+1}: center=({c.center_x:.0f},{c.center_y:.0f}) "
              f"angle={c.angle_deg:.0f}° opening={c.opening_px:.0f}px "
              f"geo={c.geo_score:.2f}")

    # Debug visualization
    if image_rgb is not None and output_dir is not None:
        dbg_path = os.path.join(output_dir, f"critic_debug_{fc:03d}.png")
        debug_draw_candidates_on_mask(mask, image_rgb, raw, candidates, dbg_path)

    return candidates


def generate_candidates_fallback(cx_px, cy_px, mm_to_px):
    """Fallback: fixed angles at the VLM-located center (original approach)."""
    print("  Using fallback fixed-angle candidates")
    half_opening_px = (GRIPPER_MAX_OPENING_MM / 2.0) * mm_to_px
    opening_px = half_opening_px * 2
    finger_thickness_px = max(opening_px * 0.25, 3.0)
    finger_depth_px = max(opening_px * 0.6, 8.0)

    candidates = []
    for angle_deg in [0, 45, 90, 135]:
        angle_rad = math.radians(angle_deg)
        dx = math.cos(angle_rad) * half_opening_px
        dy = math.sin(angle_rad) * half_opening_px
        candidates.append(GraspCandidate(
            angle_deg=angle_deg,
            center_x=cx_px, center_y=cy_px,
            jaw1_cx=cx_px + dx, jaw1_cy=cy_px + dy,
            jaw2_cx=cx_px - dx, jaw2_cy=cy_px - dy,
            opening_px=opening_px,
            finger_thickness_px=finger_thickness_px,
            finger_depth_px=finger_depth_px,
            geo_score=0.5,
        ))
    return candidates


# --- Rendering ---


def _rotated_rect_pts(cx, cy, w, h, angle_rad):
    """Return 4 corners of a rotated rectangle centered at (cx, cy)."""
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    hw, hh = w / 2.0, h / 2.0
    pts = []
    for dx, dy in [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]:
        pts.append([int(round(cx + dx * cos_a - dy * sin_a)),
                     int(round(cy + dx * sin_a + dy * cos_a))])
    return np.array(pts, dtype=np.int32)


def _draw_dashed_line(img, pt1, pt2, color, thickness=2, dash_len=8):
    """Draw a dashed line from pt1 to pt2."""
    dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 1:
        return
    dx, dy = dx / dist, dy / dist
    d, drawing = 0, True
    while d < dist:
        end_d = min(d + dash_len, dist)
        if drawing:
            p1 = (int(round(pt1[0] + dx * d)), int(round(pt1[1] + dy * d)))
            p2 = (int(round(pt1[0] + dx * end_d)), int(round(pt1[1] + dy * end_d)))
            cv2.line(img, p1, p2, color, thickness)
        d = end_d
        drawing = not drawing


def render_single_candidate(image_rgb, candidate, color=GRIPPER_COLOR_RGB):
    """Draw a single candidate on a copy of the image. Returns annotated RGB."""
    overlay = image_rgb.copy()
    perp_rad = math.radians(candidate.angle_deg) + math.pi / 2.0

    for jcx, jcy in [(candidate.jaw1_cx, candidate.jaw1_cy),
                     (candidate.jaw2_cx, candidate.jaw2_cy)]:
        pts = _rotated_rect_pts(
            jcx, jcy,
            candidate.finger_depth_px, candidate.finger_thickness_px,
            perp_rad,
        )
        sub = overlay.copy()
        cv2.fillPoly(sub, [pts], color)
        cv2.addWeighted(sub, 0.5, overlay, 0.5, 0, overlay)
        cv2.polylines(overlay, [pts], True, color, 2)

    _draw_dashed_line(
        overlay,
        (int(round(candidate.jaw1_cx)), int(round(candidate.jaw1_cy))),
        (int(round(candidate.jaw2_cx)), int(round(candidate.jaw2_cy))),
        color,
    )
    return overlay


def rotate_image_around_center(image, angle_deg, center_px):
    """Rotate image by angle_deg around center_px."""
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((float(center_px[0]), float(center_px[1])),
                                 angle_deg, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def render_rotated_candidate_image(image_rgb, candidate, crop_padding=3.0):
    """Render candidate on image, rotate so gripper is horizontal, then crop+zoom.

    crop_padding: how many times the opening to include as context around the grasp.
    """
    annotated = render_single_candidate(image_rgb, candidate)
    center = (candidate.center_x, candidate.center_y)

    # Rotate so grasp axis is horizontal
    if abs(candidate.angle_deg % 180) >= 1.0:
        annotated = rotate_image_around_center(
            annotated, candidate.angle_deg, center,
        )

    # Crop around the grasp center with padding proportional to opening
    h, w = annotated.shape[:2]
    crop_half = max(candidate.opening_px * crop_padding, 60)
    cx, cy = int(round(center[0])), int(round(center[1]))

    # After rotation, center stays at the same pixel (rotation was around it)
    x0 = max(0, int(cx - crop_half))
    y0 = max(0, int(cy - crop_half))
    x1 = min(w, int(cx + crop_half))
    y1 = min(h, int(cy + crop_half))

    cropped = annotated[y0:y1, x0:x1]

    # Upscale to at least 400px on the short side for VLM visibility
    ch, cw = cropped.shape[:2]
    min_dim = min(ch, cw)
    if min_dim < 400 and min_dim > 0:
        scale = 400.0 / min_dim
        cropped = cv2.resize(
            cropped, (int(cw * scale), int(ch * scale)),
            interpolation=cv2.INTER_LINEAR,
        )

    return cropped


def render_all_candidates_overview(image_rgb, candidates, scores, reasons):
    """Draw all candidates on one image with scores for the final summary."""
    colors = [
        (255, 0, 0), (0, 200, 0), (0, 100, 255),
        (220, 220, 0), (200, 0, 200), (0, 200, 200),
        (255, 128, 0), (128, 0, 255),
    ]
    overlay = image_rgb.copy()
    for i, cand in enumerate(candidates):
        color = colors[i % len(colors)]
        perp_rad = math.radians(cand.angle_deg) + math.pi / 2.0

        for jcx, jcy in [(cand.jaw1_cx, cand.jaw1_cy),
                         (cand.jaw2_cx, cand.jaw2_cy)]:
            pts = _rotated_rect_pts(
                jcx, jcy, cand.finger_depth_px, cand.finger_thickness_px,
                perp_rad,
            )
            sub = overlay.copy()
            cv2.fillPoly(sub, [pts], color)
            cv2.addWeighted(sub, 0.4, overlay, 0.6, 0, overlay)
            cv2.polylines(overlay, [pts], True, color, 2)

        _draw_dashed_line(
            overlay,
            (int(round(cand.jaw1_cx)), int(round(cand.jaw1_cy))),
            (int(round(cand.jaw2_cx)), int(round(cand.jaw2_cy))),
            color,
        )

        label = f"{i+1}:{scores[i]}/10"
        lx = int(round(cand.jaw1_cx)) + 12
        ly = int(round(cand.jaw1_cy)) - 12
        cv2.putText(overlay, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best = candidates[best_idx]
    text = (f"Best: #{best_idx+1} ({best.angle_deg:.0f}deg, "
            f"{scores[best_idx]}/10) - {reasons[best_idx]}")
    cv2.putText(overlay, text, (10, HEIGHT - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    return overlay


def render_place_target(image_rgb, cx, cy, radius=30):
    """Draw green circle + crosshair at the place target center."""
    overlay = image_rgb.copy()
    color = (0, 220, 0)
    icx, icy = int(round(cx)), int(round(cy))
    cv2.circle(overlay, (icx, icy), radius, color, 3)
    cv2.circle(overlay, (icx, icy), 4, color, -1)
    arm_len = radius + 10
    cv2.line(overlay, (icx - arm_len, icy), (icx + arm_len, icy), color, 2)
    cv2.line(overlay, (icx, icy - arm_len), (icx, icy + arm_len), color, 2)
    return overlay


def format_grasp_result(candidate):
    """Convert candidate to {"p1":[x,y],"p2":[x,y]} in raw pixel coords."""
    result = {
        "p1": [int(round(candidate.jaw1_cx)),
               int(round(candidate.jaw1_cy))],
        "p2": [int(round(candidate.jaw2_cx)),
               int(round(candidate.jaw2_cy))],
    }
    if mask_bbox is not None:
        result["mask_bbox"] = [
            int(round(mask_bbox[0])),
            int(round(mask_bbox[1])),
            int(round(mask_bbox[2])),
            int(round(mask_bbox[3])),
        ]
    if command_ts:
        result["command_ts"] = command_ts
    return result


# --- State machine ---
STATE_IDLE = "IDLE"
STATE_LOCATE_PENDING = "LOCATE_PENDING"
STATE_MASK_PENDING = "MASK_PENDING"       # waiting for SAM2 mask (after VLM locate)
STATE_SAM3_VLM_LOCATE = "SAM3_VLM_LOCATE"    # waiting for VLM to locate object for SAM3 point prompt
STATE_SAM3_MASK_PENDING = "SAM3_MASK_PENDING"  # waiting for SAM3 mask
STATE_RATING_PENDING = "RATING_PENDING"
STATE_PLACE_VLM_LOCATE = "PLACE_VLM_LOCATE"      # waiting for VLM to locate container center
STATE_PLACE_SAM3_PENDING = "PLACE_SAM3_PENDING"  # waiting for SAM3 point-based mask of container

state = STATE_IDLE
latest_image = None
latest_depth = None
latest_intrinsics = None
pending_trigger = False
candidates = []
locate_center = None
rating_idx = 0
scores = []
reasons = []
# Place detection state
best_grasp_result = None  # stored grasp result dict while place detection runs
place_center = None       # (cx_px, cy_px) of the container centroid
place_vlm_point = None    # VLM-located point for connected component filtering
command_ts = 0            # timestamp from chat for KPI
mask_bbox = None          # (x_min, y_min, x_max, y_max) pixel bbox of segmented object
last_vlm_place_bbox = None  # VLM-returned bbox for place target
vlm_locate_retries = 0     # retry counter for VLM locate failures
retries_left = 0          # retry counter for segmentation/locate failures
MAX_RETRIES = 1           # how many times to retry before giving up

node = Node()
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Clean up old result images to avoid stale files from previous runs
import glob as _glob
for _old in _glob.glob(os.path.join(OUTPUT_DIR, "critic_*.png")):
    os.remove(_old)
frame_count = 0


def _emit_best_geo(node, img, cands, fc):
    """Skip VLM rating — pick the best geometric candidate and emit immediately."""
    global state, best_grasp_result, place_center, place_vlm_point, frame_count
    winner = max(cands, key=lambda c: c.geo_score)
    print(f"[Skip VLM] Using best geometric candidate "
          f"({winner.angle_deg:.0f}deg, geo={winner.geo_score:.2f})")
    node.send_output("status", pa.array([f"Grasp found ({winner.angle_deg:.0f}deg)"]))
    grasp = format_grasp_result(winner)

    if PLACE_CONTAINER:
        if place_center is not None:
            # VLM located the place target — segment with SAM3 for proper mask
            place_cx, place_cy = place_center
            best_grasp_result = grasp
            place_vlm_point = (place_cx, place_cy)
            print(f"[Place] Refining '{PLACE_CONTAINER}' at ({place_cx:.0f},{place_cy:.0f}) with SAM3...")
            node.send_output("status", pa.array([f"Segmenting '{PLACE_CONTAINER}'..."]))
            if last_vlm_place_bbox is not None:
                bx0, by0, bx1, by1 = last_vlm_place_bbox
                print(f"  [Place] Sending VLM bbox [{bx0:.0f},{by0:.0f},{bx1:.0f},{by1:.0f}] to SAM3")
                node.send_output("sam3_boxes", pa.array([bx0, by0, bx1, by1]),
                                 {"image_id": "image", "text": PLACE_CONTAINER})
            else:
                hw = 50
                bx0 = max(0, place_cx - hw)
                by0 = max(0, place_cy - hw)
                bx1 = min(WIDTH, place_cx + hw)
                by1 = min(HEIGHT, place_cy + hw)
                node.send_output("sam3_boxes", pa.array([bx0, by0, bx1, by1]),
                                 {"image_id": "image", "text": PLACE_CONTAINER})
            place_center = None
            state = STATE_PLACE_SAM3_PENDING
        else:
            # No pre-located coords — ask VLM to locate the container
            best_grasp_result = grasp
            print(f"\n[Place] Asking VLM to locate '{PLACE_CONTAINER}'...")
            node.send_output("status", pa.array([f"Locating '{PLACE_CONTAINER}'..."]))
            prompt = PLACE_LOCATE_PROMPT_TEMPLATE.format(container_name=PLACE_CONTAINER)
            send_vlm_request(node, img, prompt)
            state = STATE_PLACE_VLM_LOCATE
    else:
        grasp_json = json.dumps(grasp)
        print(f"  Grasp result: {grasp_json}")
        node.send_output("grasp_result", pa.array([grasp_json]))
        frame_count += 1
        state = STATE_IDLE


def _start_rating(node, img, cands, fc):
    """Begin the VLM rating loop for the first candidate."""
    global state, candidates, scores, reasons, rating_idx

    if SKIP_VLM_RATING:
        return _emit_best_geo(node, img, cands, fc)

    candidates = cands
    scores = []
    reasons = []
    rating_idx = 0

    cand = candidates[0]
    rotated = render_rotated_candidate_image(img, cand)
    rot_path = os.path.join(OUTPUT_DIR, f"critic_cand{rating_idx}_{fc:03d}.png")
    cv2.imwrite(rot_path, cv2.cvtColor(rotated, cv2.COLOR_RGB2BGR))
    n_cands = len(candidates)
    print(f"[Pass 2] Rating candidate {rating_idx+1}/{n_cands} "
          f"({cand.angle_deg:.0f}deg, geo={cand.geo_score:.2f})...")

    prompt = RATE_PROMPT_TEMPLATE.format(object_name=TARGET_OBJECT)
    send_vlm_request(node, rotated, prompt)
    state = STATE_RATING_PENDING

for event in node:
    if event["type"] == "INPUT":
        event_id = event["id"]

        if event_id == "image":
            latest_image = event["value"].to_numpy().tobytes()
            metadata = event["metadata"]
            if "focal_length" in metadata:
                fl = metadata["focal_length"]
                if isinstance(fl, str):
                    try:
                        fl = json.loads(fl)
                    except (json.JSONDecodeError, ValueError):
                        fl = None
                if isinstance(fl, list) and len(fl) >= 2:
                    latest_intrinsics = (float(fl[0]), float(fl[1]))
            # Process deferred trigger now that we have an image
            if pending_trigger:
                pending_trigger = False
                event_id = "trigger"

        elif event_id == "depth":
            raw = event["value"].to_numpy()
            latest_depth = raw.astype(np.uint16).reshape((HEIGHT, WIDTH))

        elif event_id == "command":
            # Dynamic command from chat node — override targets and trigger
            raw = event["value"][0].as_py()
            try:
                cmd = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                print(f"[command] Failed to parse JSON: {raw}")
                continue
            TARGET_OBJECT = cmd.get("pick", TARGET_OBJECT)
            PLACE_CONTAINER = cmd.get("place", "")
            command_ts = cmd.get("command_ts", 0)
            print(f"[command] pick='{TARGET_OBJECT}', place='{PLACE_CONTAINER}'")
            # Fall through to trigger logic below
            event_id = "trigger"

        if event_id == "trigger":
            if state != STATE_IDLE:
                print(f"Busy (state={state}), ignoring trigger")
                continue
            if latest_image is None:
                print("No image yet, deferring trigger until first frame")
                pending_trigger = True
                continue

            retries_left = MAX_RETRIES

            if USE_SAM3:
                # SAM3 mode: ask VLM to locate first, then send point to SAM3
                img = np.frombuffer(latest_image, dtype=np.uint8).reshape(
                    (HEIGHT, WIDTH, 3)
                )
                place_center = None
                locate_center = None
                # Always locate pick object alone (single-object prompt is reliable)
                # Place will be located separately after pick is found
                print(f"[SAM3] Asking VLM to locate '{TARGET_OBJECT}'...")
                node.send_output("status", pa.array([f"Locating '{TARGET_OBJECT}'..."]))
                prompt = LOCATE_PROMPT_TEMPLATE.format(object_name=TARGET_OBJECT)
                send_vlm_request(node, img, prompt)
                vlm_locate_retries = 0
                state = STATE_SAM3_VLM_LOCATE
            else:
                img = np.frombuffer(latest_image, dtype=np.uint8).reshape(
                    (HEIGHT, WIDTH, 3)
                )
                place_center = None
                if PLACE_CONTAINER:
                    print(f"[Pass 1] Asking VLM to locate '{TARGET_OBJECT}' + '{PLACE_CONTAINER}'...")
                    node.send_output("status", pa.array([f"VLM locating '{TARGET_OBJECT}' + '{PLACE_CONTAINER}'..."]))
                    prompt = LOCATE_BOTH_PROMPT_TEMPLATE.format(
                        pick_name=TARGET_OBJECT, place_name=PLACE_CONTAINER)
                else:
                    print(f"[Pass 1] Asking VLM to locate {TARGET_OBJECT}...")
                    node.send_output("status", pa.array([f"VLM locating '{TARGET_OBJECT}'..."]))
                    prompt = LOCATE_PROMPT_TEMPLATE.format(object_name=TARGET_OBJECT)
                send_vlm_request(node, img, prompt)
                state = STATE_LOCATE_PENDING

        elif event_id == "mask":
            if state == STATE_PLACE_SAM3_PENDING:
                # --- Place container mask ---
                mask_meta = event["metadata"]
                mask_w = int(mask_meta.get("width", WIDTH))
                mask_h = int(mask_meta.get("height", HEIGHT))
                mask_raw = event["value"].to_numpy(zero_copy_only=False)
                if mask_raw.size == 0 or mask_raw.size < mask_h * mask_w:
                    print("  [Place] SAM3 returned empty mask, falling back to grasp-only")
                    node.send_output("status", pa.array([f"Place target '{PLACE_CONTAINER}' not found, grasp-only"]))
                    grasp_json = json.dumps(best_grasp_result)
                    print(f"  Grasp result: {grasp_json}")
                    node.send_output("grasp_result", pa.array([grasp_json]))
                    best_grasp_result = None
                    frame_count += 1
                    state = STATE_IDLE
                    continue
                mask = (mask_raw.reshape((mask_h, mask_w)) > 0).astype(np.uint8) * 255
                n_mask_raw = np.count_nonzero(mask)

                # Crop place mask to VLM bbox
                if last_vlm_place_bbox is not None:
                    bx0 = max(0, int(last_vlm_place_bbox[0]))
                    by0 = max(0, int(last_vlm_place_bbox[1]))
                    bx1 = min(mask_w, int(last_vlm_place_bbox[2]) + 1)
                    by1 = min(mask_h, int(last_vlm_place_bbox[3]) + 1)
                    cropped = np.zeros_like(mask)
                    cropped[by0:by1, bx0:bx1] = mask[by0:by1, bx0:bx1]
                    n_cropped = np.count_nonzero(cropped)
                    if n_cropped > 50:
                        print(f"  [Place] Cropped mask to VLM bbox [{bx0},{by0},{bx1},{by1}]: "
                              f"{n_mask_raw} -> {n_cropped} pixels")
                        mask = cropped
                        n_mask_raw = n_cropped

                # Keep only the connected component closest to the VLM-located
                # center.  SAM3 often returns multiple objects in one mask.
                if place_vlm_point is not None and n_mask_raw > 0:
                    num_labels, labels = cv2.connectedComponents(mask)
                    if num_labels > 2:  # >1 component (label 0 = background)
                        pcx, pcy = int(round(place_vlm_point[0])), int(round(place_vlm_point[1]))
                        best_label, best_dist = -1, float("inf")
                        for lbl in range(1, num_labels):
                            ys_l, xs_l = np.where(labels == lbl)
                            cx_l, cy_l = xs_l.mean(), ys_l.mean()
                            dist = (cx_l - pcx)**2 + (cy_l - pcy)**2
                            if dist < best_dist:
                                best_dist = dist
                                best_label = lbl
                        if best_label > 0:
                            mask = ((labels == best_label).astype(np.uint8) * 255)
                            print(f"  [Place] Filtered to component {best_label}/{num_labels-1} "
                                  f"(was {n_mask_raw}px, now {np.count_nonzero(mask)}px)")

                n_mask = np.count_nonzero(mask)
                print(f"  [Place] Container mask: {n_mask} pixels")

                if n_mask < 100:
                    # Empty mask — send grasp-only result
                    print("  [Place] Container mask too small, falling back to grasp-only")
                    node.send_output("status", pa.array([f"Place target '{PLACE_CONTAINER}' not found, grasp-only"]))
                    grasp_json = json.dumps(best_grasp_result)
                    print(f"  Grasp result: {grasp_json}")
                    node.send_output("grasp_result", pa.array([grasp_json]))
                    best_grasp_result = None
                    frame_count += 1
                    state = STATE_IDLE
                    continue

                ys, xs = np.where(mask > 0)
                # Use bounding box center (more resilient than centroid for irregular shapes)
                place_cx = float((xs.min() + xs.max()) / 2.0)
                place_cy = float((ys.min() + ys.max()) / 2.0)
                place_center = (place_cx, place_cy)
                print(f"  [Place] Container bbox center from mask: ({place_cx:.0f}, {place_cy:.0f})"
                      f"  (bbox: x=[{xs.min()},{xs.max()}] y=[{ys.min()},{ys.max()}])")

                # Save place mask debug image
                mask_path = os.path.join(OUTPUT_DIR, f"critic_place_mask_{frame_count:03d}.png")
                cv2.imwrite(mask_path, mask)

                # Save place target visualization
                img = np.frombuffer(latest_image, dtype=np.uint8).reshape(
                    (HEIGHT, WIDTH, 3)
                )
                place_img = render_place_target(img, place_cx, place_cy)
                place_path = os.path.join(OUTPUT_DIR, f"critic_place_{frame_count:03d}.png")
                cv2.imwrite(place_path, cv2.cvtColor(place_img, cv2.COLOR_RGB2BGR))
                print(f"  Saved place target: {place_path}")

                # Add place target to grasp result and send
                grasp = best_grasp_result.copy() if best_grasp_result else {}
                grasp["place_px"] = [
                    round(place_cx, 1),
                    round(place_cy, 1),
                ]
                # Use VLM bbox if available (more accurate), fall back to SAM3 mask extent
                if last_vlm_place_bbox is not None:
                    bx0, by0, bx1, by1 = last_vlm_place_bbox
                    max_half = 200
                    bx0 = max(bx0, place_cx - max_half)
                    by0 = max(by0, place_cy - max_half)
                    bx1 = min(bx1, place_cx + max_half)
                    by1 = min(by1, place_cy + max_half)
                    grasp["place_mask_bbox"] = [int(bx0), int(by0), int(bx1), int(by1)]
                    print(f"  [Place] Using VLM bbox (clamped): {grasp['place_mask_bbox']}")
                else:
                    grasp["place_mask_bbox"] = [
                        int(xs.min()), int(ys.min()),
                        int(xs.max()), int(ys.max()),
                    ]
                grasp_json = json.dumps(grasp)
                print(f"  Grasp result: {grasp_json}")
                node.send_output("grasp_result", pa.array([grasp_json]))
                best_grasp_result = None
                place_center = None
                frame_count += 1
                state = STATE_IDLE
                continue

            if state not in (STATE_MASK_PENDING, STATE_SAM3_MASK_PENDING):
                print(f"Mask arrived in unexpected state {state}, ignoring")
                continue

            is_sam3 = (state == STATE_SAM3_MASK_PENDING)
            source_name = "SAM3" if is_sam3 else "SAM2"

            # Receive mask
            mask_meta = event["metadata"]
            mask_w = int(mask_meta.get("width", WIDTH))
            mask_h = int(mask_meta.get("height", HEIGHT))
            mask_raw = event["value"].to_numpy(zero_copy_only=False)
            if mask_raw.size == 0 or mask_raw.size < mask_h * mask_w:
                print(f"  {source_name} returned empty mask for '{TARGET_OBJECT}'")
                if retries_left > 0:
                    retries_left -= 1
                    print(f"  Retrying segmentation ({retries_left} retries left)...")
                    node.send_output("status", pa.array([f"Retrying segmentation..."]))
                    # Re-trigger: send VLM locate request again
                    img = np.frombuffer(latest_image, dtype=np.uint8).reshape(
                        (HEIGHT, WIDTH, 3)
                    )
                    place_center = None
                    locate_center = None
                    if PLACE_CONTAINER:
                        prompt = LOCATE_BOTH_PROMPT_TEMPLATE.format(
                            pick_name=TARGET_OBJECT, place_name=PLACE_CONTAINER)
                    else:
                        prompt = LOCATE_PROMPT_TEMPLATE.format(object_name=TARGET_OBJECT)
                    send_vlm_request(node, img, prompt)
                    state = STATE_SAM3_VLM_LOCATE
                else:
                    node.send_output("grasp_result", pa.array([json.dumps({
                        "status": "failed",
                        "reason": f"Could not segment '{TARGET_OBJECT}'",
                    })]))
                    state = STATE_IDLE
                continue
            mask = (mask_raw.reshape((mask_h, mask_w)) > 0).astype(np.uint8) * 255
            n_mask_raw = np.count_nonzero(mask)

            # Crop mask to VLM bounding box to remove over-segmented regions
            if last_vlm_bbox is not None:
                bx0 = max(0, int(last_vlm_bbox[0]))
                by0 = max(0, int(last_vlm_bbox[1]))
                bx1 = min(mask_w, int(last_vlm_bbox[2]) + 1)
                by1 = min(mask_h, int(last_vlm_bbox[3]) + 1)
                cropped = np.zeros_like(mask)
                cropped[by0:by1, bx0:bx1] = mask[by0:by1, bx0:bx1]
                n_cropped = np.count_nonzero(cropped)
                if n_cropped > 50:
                    print(f"  Cropped mask to VLM bbox [{bx0},{by0},{bx1},{by1}]: "
                          f"{n_mask_raw} -> {n_cropped} pixels")
                    mask = cropped
                else:
                    print(f"  VLM bbox crop too small ({n_cropped}px), keeping full mask")

            n_mask = np.count_nonzero(mask)
            total_px = mask_w * mask_h
            mask_ratio = n_mask / total_px
            print(f"  {source_name} mask received: {n_mask} pixels ({mask_ratio:.1%} of image)")

            # Compute mask bounding box for object-top-Z estimation in motion planner
            # Prefer VLM bbox (tighter) over SAM3 mask extent (may over-segment)
            if last_vlm_bbox is not None:
                mask_bbox = last_vlm_bbox
                print(f"  Mask bbox (VLM): x=[{mask_bbox[0]:.0f},{mask_bbox[2]:.0f}] "
                      f"y=[{mask_bbox[1]:.0f},{mask_bbox[3]:.0f}]")
            elif n_mask > 0:
                ys_m, xs_m = np.where(mask > 0)
                mask_bbox = (float(xs_m.min()), float(ys_m.min()),
                             float(xs_m.max()), float(ys_m.max()))
                print(f"  Mask bbox (SAM): x=[{mask_bbox[0]:.0f},{mask_bbox[2]:.0f}] "
                      f"y=[{mask_bbox[1]:.0f},{mask_bbox[3]:.0f}]")
            else:
                mask_bbox = None

            if is_sam3:
                if locate_center is not None:
                    # VLM-located point was used for SAM3 segmentation
                    print(f"  {source_name} using VLM locate center: ({locate_center[0]:.0f}, {locate_center[1]:.0f})")
                else:
                    # Text-prompt fallback: compute center from mask centroid
                    if n_mask > 0:
                        ys, xs = np.where(mask > 0)
                        cx_px = float(np.mean(xs))
                        cy_px = float(np.mean(ys))
                    else:
                        cx_px, cy_px = WIDTH / 2.0, HEIGHT / 2.0
                    locate_center = (cx_px, cy_px)
                    print(f"  {source_name} mask centroid: ({cx_px:.0f}, {cy_px:.0f})")
            else:
                # SAM2: use VLM-located center
                # If mask covers >30% of image, SAM2 likely segmented the background
                if mask_ratio > 0.3:
                    print(f"  WARNING: mask too large ({mask_ratio:.1%}), inverting")
                    mask = 255 - mask

                    cx_px, cy_px = locate_center
                    num_labels, labels = cv2.connectedComponents(mask)
                    ix = max(0, min(int(round(cx_px)), mask_w - 1))
                    iy = max(0, min(int(round(cy_px)), mask_h - 1))
                    center_label = labels[iy, ix]
                    if center_label == 0:
                        ys, xs = np.where(mask > 0)
                        if len(ys) > 0:
                            dists = (ys - iy) ** 2 + (xs - ix) ** 2
                            nearest = np.argmin(dists)
                            center_label = labels[ys[nearest], xs[nearest]]
                    if center_label > 0:
                        mask = (labels == center_label).astype(np.uint8) * 255
                    n_mask = np.count_nonzero(mask)
                    print(f"  After inversion + component filter: {n_mask} pixels")

            cx_px, cy_px = locate_center
            img = np.frombuffer(latest_image, dtype=np.uint8).reshape(
                (HEIGHT, WIDTH, 3)
            )
            mm_to_px = compute_gripper_pixel_scale(
                cx_px, cy_px, latest_depth, latest_intrinsics
            )

            if n_mask > 100:
                mask_path = os.path.join(OUTPUT_DIR, f"critic_mask_{frame_count:03d}.png")
                cv2.imwrite(mask_path, mask)
                print(f"  Saved mask: {mask_path}")
                candidates = generate_candidates_from_mask(
                    mask, mm_to_px, image_rgb=img,
                    output_dir=OUTPUT_DIR, fc=frame_count,
                )

            if not candidates:
                print(f"  {source_name} mask contour analysis failed, using fallback")
                candidates = generate_candidates_fallback(cx_px, cy_px, mm_to_px)

            _start_rating(node, img, candidates, frame_count)

        elif event_id == "vlm_response":
            text = event["value"][0].as_py()

            if state == STATE_SAM3_VLM_LOCATE:
                print(f"[SAM3] Raw VLM locate response: {text[:300]}")
                pick_coords = None
                result = parse_locate_response(text)
                if result is not None and result[0] >= 0:
                    pick_coords = result

                if pick_coords is None:
                    vlm_locate_retries += 1
                    if vlm_locate_retries <= 3:
                        print(f"[SAM3] VLM locate failed — retrying ({vlm_locate_retries}/3)")
                        prompt = LOCATE_PROMPT_TEMPLATE.format(object_name=TARGET_OBJECT)
                        send_vlm_request(node, img, prompt)
                        continue
                    else:
                        print(f"[SAM3] VLM locate failed after 3 retries, giving up")
                        vlm_locate_retries = 0
                        state = STATE_IDLE
                        continue
                else:
                    cx_px, cy_px = pick_coords
                    locate_center = (cx_px, cy_px)
                    if last_vlm_bbox is not None:
                        bx0, by0, bx1, by1 = last_vlm_bbox
                        print(f"[SAM3] VLM bbox [{bx0:.0f},{by0:.0f},{bx1:.0f},{by1:.0f}], sending box to SAM3")
                        node.send_output("status", pa.array([f"Segmenting at ({cx_px:.0f},{cy_px:.0f})..."]))
                        node.send_output("sam3_boxes", pa.array([bx0, by0, bx1, by1]),
                                         {"image_id": "image", "text": TARGET_OBJECT})
                    else:
                        print(f"[SAM3] VLM located at ({cx_px:.0f}, {cy_px:.0f}), sending box to SAM3")
                        node.send_output("status", pa.array([f"Segmenting at ({cx_px:.0f},{cy_px:.0f})..."]))
                        hw = 50
                        bx0 = max(0, cx_px - hw)
                        by0 = max(0, cy_px - hw)
                        bx1 = min(WIDTH, cx_px + hw)
                        by1 = min(HEIGHT, cy_px + hw)
                        node.send_output("sam3_boxes", pa.array([bx0, by0, bx1, by1]),
                                         {"image_id": "image", "text": TARGET_OBJECT})
                    state = STATE_SAM3_MASK_PENDING

            elif state == STATE_LOCATE_PENDING:
                pick_coords = None
                if PLACE_CONTAINER:
                    both = parse_locate_both_response(text)
                    if both is not None:
                        pick_px, place_px = both
                        if pick_px[0] >= 0 and pick_px[1] >= 0:
                            pick_coords = pick_px
                        if place_px is not None and place_px[0] >= 0 and place_px[1] >= 0:
                            place_center = place_px
                            print(f"[Pass 1] Place '{PLACE_CONTAINER}' at ({place_px[0]:.0f}, {place_px[1]:.0f})")
                if pick_coords is None:
                    result = parse_locate_response(text)
                    if result is not None and result[0] >= 0:
                        pick_coords = result
                if pick_coords is None:
                    print(f"[Pass 1] Failed to parse locate response: {text}")
                    state = STATE_IDLE
                    continue

                cx_px, cy_px = pick_coords
                locate_center = (cx_px, cy_px)
                print(f"[Pass 1] Located {TARGET_OBJECT} at pixel ({cx_px:.0f}, {cy_px:.0f})")

                if USE_SAM2:
                    # Send point to SAM2 for segmentation
                    print("  Sending point to SAM2 for segmentation...")
                    point_data = pa.array([float(cx_px), float(cy_px)])
                    node.send_output("sam_point", point_data, {"image_id": "image"})
                    state = STATE_MASK_PENDING
                else:
                    # Inline segmentation fallback (no SAM2)
                    img = np.frombuffer(latest_image, dtype=np.uint8).reshape(
                        (HEIGHT, WIDTH, 3)
                    )
                    mm_to_px = compute_gripper_pixel_scale(
                        cx_px, cy_px, latest_depth, latest_intrinsics
                    )
                    print(f"  mm_to_px={mm_to_px:.3f}")

                    mask = None
                    if latest_depth is not None:
                        print("  Segmenting with depth...")
                        mask = segment_object_depth(latest_depth, cx_px, cy_px)
                    if mask is None:
                        print("  Segmenting with color (fallback)...")
                        mask = segment_object_color(img, cx_px, cy_px)

                    if mask is not None and np.count_nonzero(mask) > 100:
                        mask_path = os.path.join(OUTPUT_DIR, f"critic_mask_{frame_count:03d}.png")
                        cv2.imwrite(mask_path, mask)
                        print(f"  Saved mask: {mask_path} ({np.count_nonzero(mask)} pixels)")
                        candidates = generate_candidates_from_mask(
                            mask, mm_to_px, image_rgb=img,
                            output_dir=OUTPUT_DIR, fc=frame_count,
                        )

                    if not candidates:
                        print("  Contour analysis failed, using fallback")
                        candidates = generate_candidates_fallback(cx_px, cy_px, mm_to_px)

                    _start_rating(node, img, candidates, frame_count)

            elif state == STATE_RATING_PENDING:
                result = parse_rate_response(text)
                cand = candidates[rating_idx]
                n_cands = len(candidates)

                if result is None:
                    print(f"  Failed to parse rating for candidate {rating_idx+1}: {text}")
                    scores.append(0)
                    reasons.append("parse_error")
                else:
                    score, reason = result
                    scores.append(score)
                    reasons.append(reason)
                    print(f"  Candidate {rating_idx+1} ({cand.angle_deg:.0f}deg): "
                          f"{score}/10 - {reason}")

                rating_idx += 1

                # Early stop: if this candidate scored high enough, skip the rest
                early_stop = (
                    EARLY_STOP_SCORE > 0
                    and len(scores) > 0
                    and scores[-1] >= EARLY_STOP_SCORE
                    and rating_idx < n_cands
                )
                if early_stop:
                    print(f"  Early stop: candidate scored {scores[-1]}/10 "
                          f"(>= {EARLY_STOP_SCORE}), skipping {n_cands - rating_idx} remaining")

                if rating_idx < n_cands and not early_stop:
                    img = np.frombuffer(latest_image, dtype=np.uint8).reshape(
                        (HEIGHT, WIDTH, 3)
                    )
                    cand = candidates[rating_idx]
                    rotated = render_rotated_candidate_image(img, cand)
                    rot_path = os.path.join(OUTPUT_DIR, f"critic_cand{rating_idx}_{frame_count:03d}.png")
                    cv2.imwrite(rot_path, cv2.cvtColor(rotated, cv2.COLOR_RGB2BGR))
                    print(f"[Pass 2] Rating candidate {rating_idx+1}/{n_cands} "
                          f"({cand.angle_deg:.0f}deg, geo={cand.geo_score:.2f})...")

                    prompt = RATE_PROMPT_TEMPLATE.format(object_name=TARGET_OBJECT)
                    send_vlm_request(node, rotated, prompt)
                else:
                    # Pick the best among rated candidates
                    rated = candidates[:len(scores)]
                    best_idx = max(range(len(scores)), key=lambda i: scores[i])
                    winner = rated[best_idx]
                    score_strs = [f'#{i+1}:{s}/10' for i, s in enumerate(scores)]
                    print(f"\n[Result] Scores: {score_strs} "
                          f"({len(scores)}/{n_cands} rated)")
                    print(f"[Result] Best: #{best_idx+1} ({winner.angle_deg:.0f}deg, "
                          f"{scores[best_idx]}/10)")

                    img = np.frombuffer(latest_image, dtype=np.uint8).reshape(
                        (HEIGHT, WIDTH, 3)
                    )
                    overview = render_all_candidates_overview(
                        img, rated, scores, reasons
                    )
                    ov_path = os.path.join(OUTPUT_DIR, f"critic_overview_{frame_count:03d}.png")
                    cv2.imwrite(ov_path, cv2.cvtColor(overview, cv2.COLOR_RGB2BGR))
                    print(f"  Saved overview: {ov_path}")

                    grasp = format_grasp_result(winner)

                    if PLACE_CONTAINER:
                        best_grasp_result = grasp
                        if place_center is not None:
                            # Have VLM-estimated place coords — refine with SAM3
                            place_cx, place_cy = place_center
                            place_vlm_point = (place_cx, place_cy)
                            print(f"[Place] Refining '{PLACE_CONTAINER}' at ({place_cx:.0f},{place_cy:.0f}) with SAM3...")
                            node.send_output("status", pa.array([f"Segmenting '{PLACE_CONTAINER}'..."]))
                            if last_vlm_place_bbox is not None:
                                bx0, by0, bx1, by1 = last_vlm_place_bbox
                                print(f"  [Place] Sending VLM bbox [{bx0:.0f},{by0:.0f},{bx1:.0f},{by1:.0f}] to SAM3")
                                node.send_output("sam3_boxes", pa.array([bx0, by0, bx1, by1]),
                                                 {"image_id": "image", "text": PLACE_CONTAINER})
                            else:
                                hw = 50
                                bx0 = max(0, place_cx - hw)
                                by0 = max(0, place_cy - hw)
                                bx1 = min(WIDTH, place_cx + hw)
                                by1 = min(HEIGHT, place_cy + hw)
                                node.send_output("sam3_boxes", pa.array([bx0, by0, bx1, by1]),
                                                 {"image_id": "image", "text": PLACE_CONTAINER})
                            place_center = None
                            state = STATE_PLACE_SAM3_PENDING
                        else:
                            # No pre-located coords — ask VLM to locate
                            print(f"\n[Place] No place coords, asking VLM to locate '{PLACE_CONTAINER}'...")
                            node.send_output("status", pa.array([f"Locating '{PLACE_CONTAINER}'..."]))
                            prompt = PLACE_LOCATE_PROMPT_TEMPLATE.format(container_name=PLACE_CONTAINER)
                            send_vlm_request(node, img, prompt)
                            state = STATE_PLACE_VLM_LOCATE
                    else:
                        grasp_json = json.dumps(grasp)
                        print(f"  Grasp result: {grasp_json}")
                        node.send_output("grasp_result", pa.array([grasp_json]))
                        frame_count += 1
                        state = STATE_IDLE

            elif state == STATE_PLACE_VLM_LOCATE:
                # VLM returns 0-1000 normalized coords — refine via SAM3
                print(f"  [Place] Raw VLM response: {text[:500]}")
                result = parse_locate_response(text)
                if result is not None:
                    place_cx, place_cy = result
                    if place_cx < 0 or place_cy < 0:
                        print(f"  [Place] VLM says '{PLACE_CONTAINER}' not found, falling back to grasp-only")
                        node.send_output("status", pa.array([f"Place target '{PLACE_CONTAINER}' not found, grasp-only"]))
                        grasp_json = json.dumps(best_grasp_result)
                        print(f"  Grasp result: {grasp_json}")
                        node.send_output("grasp_result", pa.array([grasp_json]))
                        best_grasp_result = None
                        frame_count += 1
                        state = STATE_IDLE
                        continue
                    # Refine VLM point with SAM3 segmentation → bbox center
                    place_vlm_point = (place_cx, place_cy)
                    print(f"  [Place] VLM located '{PLACE_CONTAINER}' at ({place_cx:.0f},{place_cy:.0f}), refining with SAM3...")
                    node.send_output("status", pa.array([f"Segmenting '{PLACE_CONTAINER}'..."]))
                    if last_vlm_bbox is not None:
                        bx0, by0, bx1, by1 = last_vlm_bbox
                        print(f"  [Place] Sending VLM bbox [{bx0:.0f},{by0:.0f},{bx1:.0f},{by1:.0f}] to SAM3")
                        node.send_output("sam3_boxes", pa.array([bx0, by0, bx1, by1]),
                                         {"image_id": "image", "text": PLACE_CONTAINER})
                    else:
                        # No bbox — create small box around point
                        hw = 50
                        bx0 = max(0, place_cx - hw)
                        by0 = max(0, place_cy - hw)
                        bx1 = min(WIDTH, place_cx + hw)
                        by1 = min(HEIGHT, place_cy + hw)
                        node.send_output("sam3_boxes", pa.array([bx0, by0, bx1, by1]),
                                         {"image_id": "image", "text": PLACE_CONTAINER})
                    state = STATE_PLACE_SAM3_PENDING
                else:
                    # VLM failed — retry VLM locate
                    print(f"  [Place] Failed to parse VLM location, retrying...")
                    node.send_output("status", pa.array([f"Locating '{PLACE_CONTAINER}'..."]))
                    prompt = PLACE_LOCATE_PROMPT_TEMPLATE.format(container_name=PLACE_CONTAINER)
                    send_vlm_request(node, img, prompt)
                    # Stay in STATE_PLACE_VLM_LOCATE to retry

            else:
                print(f"VLM response in unexpected state {state}, ignoring")

    elif event["type"] == "STOP":
        break
