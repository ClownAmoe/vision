#!/usr/bin/env python3
"""Detailed diagnostic to show motion detection problems."""

import logging
import numpy as np
import cv2
from pathlib import Path
from drone_parser import DroneVideoDataset
from drone_feature_matching import FeatureMatcher, DetectorType
from drone_motion_estimation import PlanarFlowEstimator

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Setup
root = Path("dataset")
video_rel = "drone_footage/23-02-01_FR_F01_V01.MP4"
csv_rel = "drone_footage/23-02-01_FR_F01.csv"

dataset = DroneVideoDataset(
    dataset_path=str(root),
    video_path=video_rel,
    csv_path=csv_rel,
    normalize_time=True,
    frame_stride=1,
    max_frames=100,
)

logging.info(dataset.summary())
logging.info("")

# Setup motion estimator
frame0, _ = dataset[0]
h, w = frame0.shape[:2]
fx = fy = max(w, h)
cx, cy = w / 2.0, h / 2.0

K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
estimator = PlanarFlowEstimator(K, min_alt_m=1.0, flow_model="affine")

# Setup matcher
matcher = FeatureMatcher(DetectorType.OPTICAL_FLOW)

logging.info("=" * 80)
logging.info("MOTION ESTIMATION vs GROUND TRUTH COMPARISON")
logging.info("=" * 80)
print()

# ============================================================================
# Main loop
# ============================================================================
estimated_pos = np.array([0.0, 0.0, 0.0])
gt_pos = np.array([0.0, 0.0, 0.0])

altitude_rejection_start = None
altitude_ok_start = None

print(f"{'Fr':<4} {'Time':<8} {'GT Alt':<8} {'Alt OK':<7} {'Matches':<10} {'Motion Est':<20} {'GT Motion':<20} {'Position Est':<25} {'Position GT':<25}")
print("-" * 180)

for idx in range(len(dataset)):
    frame, tele_curr = dataset[idx]
    alt_ok = tele_curr.alt_m >= 1.0
    
    if alt_ok and altitude_rejection_start is None:
        altitude_rejection_start = idx - 1 if idx > 0 else 0
    if not alt_ok and altitude_ok_start is None and idx > 0:
        altitude_ok_start = idx
    
    # Get next frame for matching
    if idx == len(dataset) - 1:
        continue
    
    frame_next, tele_next = dataset[idx + 1]
    
    # Detect features
    result = matcher.match(frame, frame_next, min_matches=8)
    n_matches = len(result.pts_prev) if result is not None and result.pts_prev is not None else 0
    
    # Estimate motion
    if result is None or len(result.pts_prev) < 8:
        motion_est = np.array([np.nan, np.nan, np.nan])
        reason = "NO_MATCHES" if result is None else f"LOW_MATCHES({n_matches})"
    elif not alt_ok:
        motion_est = np.array([np.nan, np.nan, np.nan])
        reason = f"ALT_LOW({tele_curr.alt_m:.1f}m)"
    else:
        estimate, fail_reason = estimator.estimate_with_reason(result.pts_prev, result.pts_curr, tele_curr)
        if estimate is None:
            motion_est = np.array([np.nan, np.nan, np.nan])
            reason = fail_reason
        else:
            motion_est = estimate.delta_enu
            reason = f"OK({estimate.n_inliers})"
            estimated_pos += motion_est
    
    # Ground truth motion
    gt_motion = tele_curr.pos_enu - gt_pos
    gt_pos = np.array(tele_curr.pos_enu)
    
    # Format output
    motion_est_str = f"[{motion_est[0]:6.2f}, {motion_est[1]:6.2f}]" if np.all(np.isfinite(motion_est)) else "      [nan, nan]       "
    gt_motion_str = f"[{gt_motion[0]:6.2f}, {gt_motion[1]:6.2f}]"
    pos_est_str = f"[{estimated_pos[0]:8.1f}, {estimated_pos[1]:8.1f}]"
    pos_gt_str = f"[{gt_pos[0]:8.1f}, {gt_pos[1]:8.1f}]"
    
    print(
        f"{idx:<4} {tele_curr.time_s:<8.2f} {tele_curr.alt_m:<8.2f} "
        f"{'✓' if alt_ok else '✗':<7} {n_matches:<10} "
        f"{motion_est_str} {gt_motion_str} "
        f"{pos_est_str} {pos_gt_str}"
    )

print()
print("=" * 180)
print("SUMMARY:")
print(f"  Altitude rejection phase: frames 0-{altitude_rejection_start} (altitude < 1.0m)")
print(f"  Altitude OK phase starts at frame: {altitude_ok_start}")
print()
print("OBSERVATIONS:")
print("  • 'Alt OK' column shows if altitude threshold (1.0m) is satisfied")
print("  • 'Matches' shows number of optical flow feature matches")
print("  • 'Motion Est' shows estimated ENU motion delta (or 'nan' if rejected)")
print("  • 'GT Motion' shows actual motion from GPS coordinates")
print("  • 'Position Est' is cumulative estimated position")
print("  • 'Position GT' is cumulative ground truth position from GPS")
print()
print("DIAGNOSTICS:")
if altitude_rejection_start is not None:
    print(f"  ✓ First {altitude_rejection_start + 1} frames rejected for low altitude (expected during takeoff)")
print("  ? If Motion Est stays at [nan, nan], check:")
print("    - Are there enough feature matches?")
print("    - Is the flow estimation failing?")
print("    - Is the altitude filtering too strict?")
print()

dataset.close()
logging.info("✓ Diagnostic complete")
