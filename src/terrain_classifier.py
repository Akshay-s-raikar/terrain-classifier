"""
Terrain Classifier for Robot Navigation
Real-time terrain segmentation and traversability scoring using OpenCV.

Author: Akshay Shailesh Raikar
GitHub: https://github.com/Akshay-s-raikar
"""

import cv2
import numpy as np
import csv
import os
import time
from datetime import datetime


# ─── CONFIG ───────────────────────────────────────────────────────────────────

CAMERA_INDEX   = 0          # change to 1 if using external camera
FRAME_WIDTH    = 1280
FRAME_HEIGHT   = 720
LOG_ENABLED    = True
LOG_DIR        = "../logs"
SHOW_DEBUG     = False      # set True to show HSV mask window

# Terrain HSV ranges (H, S, V) — tuned for outdoor robotics
TERRAIN_PROFILES = {
    "grass": {
        "lower": np.array([25, 40, 40]),
        "upper": np.array([90, 255, 255]),
        "color": (34, 139, 34),       # BGR
        "traversable": True,
        "score_base": 80,
    },
    "concrete": {
        "lower": np.array([0, 0, 140]),
        "upper": np.array([179, 40, 230]),
        "color": (160, 160, 160),
        "traversable": True,
        "score_base": 95,
    },
    "gravel": {
        "lower": np.array([0, 0, 80]),
        "upper": np.array([179, 50, 140]),
        "color": (120, 100, 80),
        "traversable": True,
        "score_base": 70,
    },
    "sand": {
        "lower": np.array([15, 30, 150]),
        "upper": np.array([30, 150, 255]),
        "color": (50, 180, 220),
        "traversable": True,
        "score_base": 60,
    },
    "mud": {
        "lower": np.array([5, 50, 30]),
        "upper": np.array([20, 180, 100]),
        "color": (30, 80, 120),
        "traversable": False,
        "score_base": 25,
    },
    "obstacle": {
        "lower": np.array([0, 0, 0]),
        "upper": np.array([179, 255, 50]),
        "color": (0, 0, 200),
        "traversable": False,
        "score_base": 0,
    },
}

# ─── LOGGER ───────────────────────────────────────────────────────────────────

class TerrainLogger:
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(log_dir, f"terrain_{ts}.csv")
        self.file = open(self.path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["timestamp", "terrain", "traversability_score", "dominant_coverage_pct", "safe_zone_area_pct"])
        print(f"[LOG] Logging to {self.path}")

    def write(self, terrain, score, coverage, safe_area):
        self.writer.writerow([
            datetime.now().isoformat(),
            terrain,
            round(score, 2),
            round(coverage, 2),
            round(safe_area, 2)
        ])
        self.file.flush()

    def close(self):
        self.file.close()


# ─── TERRAIN ANALYSIS ─────────────────────────────────────────────────────────

def extract_ground_roi(frame):
    """Extract lower 40% of frame as ground ROI (robot forward view assumption)."""
    h, w = frame.shape[:2]
    roi_top = int(h * 0.55)
    roi = frame[roi_top:h, 0:w]
    return roi, roi_top


def classify_terrain(roi_hsv):
    """
    For each terrain profile, compute mask coverage.
    Returns sorted list of (terrain_name, coverage_pct).
    """
    total_pixels = roi_hsv.shape[0] * roi_hsv.shape[1]
    results = []

    for name, profile in TERRAIN_PROFILES.items():
        mask = cv2.inRange(roi_hsv, profile["lower"], profile["upper"])
        coverage = (np.count_nonzero(mask) / total_pixels) * 100
        results.append((name, coverage, mask))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def compute_traversability(terrain_results):
    """
    Weighted traversability score based on top terrain detections.
    Returns score 0-100 and dominant terrain name.
    """
    if not terrain_results:
        return 0, "unknown"

    dominant_name, dominant_cov, _ = terrain_results[0]

    # weighted average of top 3
    total_weight = 0
    weighted_score = 0
    for name, cov, _ in terrain_results[:3]:
        profile = TERRAIN_PROFILES[name]
        weight = cov
        weighted_score += profile["score_base"] * weight
        total_weight += weight

    score = (weighted_score / total_weight) if total_weight > 0 else 0
    score = min(100, max(0, score))

    return score, dominant_name


def compute_safe_zone(roi, terrain_results):
    """
    Build a safe zone mask by combining all traversable terrain masks.
    Returns the combined mask and safe area percentage.
    """
    total_pixels = roi.shape[0] * roi.shape[1]
    safe_mask = np.zeros(roi.shape[:2], dtype=np.uint8)

    for name, cov, mask in terrain_results:
        if TERRAIN_PROFILES[name]["traversable"] and cov > 5.0:
            safe_mask = cv2.bitwise_or(safe_mask, mask)

    # morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    safe_mask = cv2.morphologyEx(safe_mask, cv2.MORPH_CLOSE, kernel)
    safe_mask = cv2.morphologyEx(safe_mask, cv2.MORPH_OPEN, kernel)

    safe_area_pct = (np.count_nonzero(safe_mask) / total_pixels) * 100
    return safe_mask, safe_area_pct


# ─── OVERLAY DRAWING ──────────────────────────────────────────────────────────

def draw_safe_overlay(frame, safe_mask, roi_top):
    """Draw semi-transparent green overlay on safe zones."""
    overlay = frame.copy()
    full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    full_mask[roi_top:, :] = safe_mask

    green_layer = np.zeros_like(frame)
    green_layer[:] = (0, 200, 80)
    frame_with_color = cv2.bitwise_and(green_layer, green_layer, mask=full_mask)
    cv2.addWeighted(frame_with_color, 0.3, overlay, 0.7, 0, overlay)

    # draw contours of safe zones
    contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 2000:
            cv2.drawContours(overlay, [cnt], -1, (0, 230, 100), 2)

    return overlay


def draw_hud(frame, score, terrain, coverage, safe_area, fps):
    h, w = frame.shape[:2]

    # background panel
    cv2.rectangle(frame, (10, 10), (360, 230), (20, 20, 20), -1)
    cv2.rectangle(frame, (10, 10), (360, 230), (60, 60, 60), 1)

    # score bar background
    bar_x, bar_y, bar_w, bar_h = 20, 60, 330, 18
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)

    # score bar fill
    score_color = (
        (0, 220, 80)   if score >= 70 else
        (0, 180, 220)  if score >= 40 else
        (0, 60, 220)
    )
    fill_w = int(bar_w * score / 100)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), score_color, -1)

    # text
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(frame, "TRAVERSABILITY", (20, 50), font, 0.45, (160, 160, 160), 1)
    cv2.putText(frame, f"{score:.0f} / 100", (bar_x + bar_w - 70, bar_y + 14), font, 0.45, (220, 220, 220), 1)

    cv2.putText(frame, f"Terrain   : {terrain.upper()}", (20, 105), font, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"Coverage  : {coverage:.1f}%",  (20, 130), font, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"Safe zone : {safe_area:.1f}%", (20, 155), font, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"FPS       : {fps:.1f}",        (20, 180), font, 0.5, (200, 200, 200), 1)

    status_text  = "SAFE TO TRAVERSE" if score >= 60 else "CAUTION" if score >= 30 else "DO NOT TRAVERSE"
    status_color = (0, 220, 80) if score >= 60 else (0, 180, 220) if score >= 30 else (0, 60, 220)
    cv2.putText(frame, status_text, (20, 215), font, 0.55, status_color, 2)

    # ROI line
    roi_top = int(h * 0.55)
    cv2.line(frame, (0, roi_top), (w, roi_top), (80, 80, 80), 1)
    cv2.putText(frame, "ground ROI", (w - 120, roi_top - 6), font, 0.38, (100, 100, 100), 1)

    # keybinds
    cv2.putText(frame, "Q: quit   D: debug   S: screenshot", (10, h - 12), font, 0.38, (80, 80, 80), 1)

    return frame


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("[INFO] Starting terrain classifier...")
    print("[INFO] Press Q to quit, D to toggle debug, S to save screenshot")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {CAMERA_INDEX}")
        return

    logger = TerrainLogger(LOG_DIR) if LOG_ENABLED else None
    show_debug = SHOW_DEBUG

    prev_time = time.time()
    frame_count = 0
    screenshot_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame read failed")
            break

        # FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-9)
        prev_time = curr_time
        frame_count += 1

        # extract ROI
        roi, roi_top = extract_ground_roi(frame)
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # classify
        terrain_results = classify_terrain(roi_hsv)
        score, dominant = compute_traversability(terrain_results)
        safe_mask, safe_area = compute_safe_zone(roi, terrain_results)
        dominant_cov = terrain_results[0][1] if terrain_results else 0

        # draw
        output = draw_safe_overlay(frame, safe_mask, roi_top)
        output = draw_hud(output, score, dominant, dominant_cov, safe_area, fps)

        cv2.imshow("Terrain Classifier — Robot Navigation", output)

        # debug window
        if show_debug:
            debug_hsv = cv2.cvtColor(roi_hsv, cv2.COLOR_HSV2BGR)
            cv2.imshow("DEBUG: ROI HSV", debug_hsv)
            top_mask = terrain_results[0][2] if terrain_results else np.zeros(roi.shape[:2], dtype=np.uint8)
            cv2.imshow(f"DEBUG: {terrain_results[0][0]} mask", top_mask)

        # log every 10 frames
        if logger and frame_count % 10 == 0:
            logger.write(dominant, score, dominant_cov, safe_area)

        # keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            show_debug = not show_debug
            if not show_debug:
                cv2.destroyWindow("DEBUG: ROI HSV")
        elif key == ord('s'):
            path = f"../assets/screenshot_{screenshot_count:03d}.png"
            cv2.imwrite(path, output)
            print(f"[INFO] Screenshot saved: {path}")
            screenshot_count += 1

    cap.release()
    cv2.destroyAllWindows()
    if logger:
        logger.close()
    print("[INFO] Classifier stopped.")


if __name__ == "__main__":
    main()
