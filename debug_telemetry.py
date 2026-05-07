#!/usr/bin/env python3
"""Debug script to visualize CSV parsing and telemetry alignment."""

import cv2
import numpy as np
from pathlib import Path
from drone_parser import DroneVideoDataset, TelemetryTimeline

# Test with first video
root = Path("dataset")
video_rel = "drone_footage/23-02-01_FR_F01_V01.MP4"
csv_rel = "drone_footage/23-02-01_FR_F01.csv"

video_path = root / video_rel
csv_path = root / csv_rel

print(f"Video: {video_path}")
print(f"CSV: {csv_path}")
print()

# ============ Check video properties ============
cap = cv2.VideoCapture(str(video_path))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
video_duration_s = frame_count / fps if fps > 0 else -1
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

print(f"VIDEO PROPERTIES:")
print(f"  Frames: {frame_count}")
print(f"  FPS: {fps}")
print(f"  Duration: {video_duration_s:.1f}s")
print(f"  Resolution: {w}x{h}")
print()

# ============ Check CSV properties ============
timeline = TelemetryTimeline(csv_path, normalize_time=True)
print(f"CSV TELEMETRY:")
print(f"  Start time: {timeline.start_time:.2f}s")
print(f"  End time: {timeline.end_time:.2f}s")
print(f"  Duration: {timeline.duration:.1f}s")
print(f"  Rows loaded: {len(timeline._times)}")
print()

# ============ Sample telemetry at key points ============
print("SAMPLE TELEMETRY VALUES:")
print(f"{'Frame':<8} {'Time(s)':<10} {'Alt(m)':<10} {'Lat':<18} {'Lon':<18} {'ENU':<30}")
print("-" * 100)

frame_indices = [0, frame_count//4, frame_count//2, 3*frame_count//4, frame_count-1]
for fidx in frame_indices:
    frame_time = fidx / fps if fps > 0 else 0
    
    # Get telemetry at this frame time
    try:
        tele = timeline.sample(frame_time)
        if tele is None:
            status = "NO_TELEMETRY"
        else:
            alt_m = tele.alt_m
            lat = tele.lat
            lon = tele.lon
            enu = tele.pos_enu
            status = f"✓ alt={alt_m:.2f}m  lat={lat:.8f}  lon={lon:.8f}  enu=[{enu[0]:7.2f}, {enu[1]:7.2f}, {enu[2]:7.2f}]"
    except Exception as e:
        status = f"ERROR: {e}"
    
    print(f"{fidx:<8} {frame_time:<10.2f} {status:<80}")

print()

# ============ Check for altitude evolution ============
print("ALTITUDE STATISTICS (meters):")
alts = timeline._alt_m
print(f"  Min: {np.min(alts):.2f}m")
print(f"  Max: {np.max(alts):.2f}m")
print(f"  Mean: {np.mean(alts):.2f}m")
print(f"  Frames with alt < 1.0m: {np.sum(alts < 1.0)}")
print(f"  Frames with alt >= 1.0m: {np.sum(alts >= 1.0)}")
print()

# ============ Show altitude over time ============
print("ALTITUDE OVER TIME (first 100 rows):")
print(f"{'Row':<6} {'Time(s)':<10} {'Alt(ft)':<10} {'Alt(m)':<10}")
print("-" * 40)
for i in range(min(100, len(timeline._times))):
    t = timeline._times[i]
    alt_m = timeline._alt_m[i]
    alt_ft = alt_m / 0.3048
    print(f"{i:<6} {t:<10.2f} {alt_ft:<10.1f} {alt_m:<10.2f}")

print()

# ============ Create dataset and check interpolation ============
print("CREATING DRONE DATASET FOR VIDEO...")
dataset = DroneVideoDataset(
    dataset_path=str(root),
    video_path=video_rel,
    csv_path=csv_rel,
    target_fps=None,
    frame_stride=1,
    normalize_time=True,
)
print(f"  Dataset length: {len(dataset)}")
print()

print("SAMPLE DATASET FRAMES:")
print(f"{'Idx':<6} {'Frame#':<8} {'Time(s)':<10} {'Alt(m)':<10} {'ENU Pos':<30}")
print("-" * 70)

test_indices = [0, len(dataset)//4, len(dataset)//2, 3*len(dataset)//4, len(dataset)-1]
for didx in test_indices:
    frame, tele = dataset[didx]
    alt = tele.alt_m
    enu = tele.pos_enu
    frame_num = dataset._indices[didx]
    print(f"{didx:<6} {frame_num:<8} {tele.time_s:<10.2f} {alt:<10.2f} [{enu[0]:7.2f}, {enu[1]:7.2f}, {enu[2]:7.2f}]")

dataset.close()
print("\n✓ Diagnostic complete")
