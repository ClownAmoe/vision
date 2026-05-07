#!/usr/bin/env python3
"""Diagnose time and altitude alignment issues."""

import numpy as np
from pathlib import Path
from drone_parser import DroneVideoDataset, TelemetryTimeline

root = Path("dataset")
video_rel = "drone_footage/23-02-01_FR_F01_V01.MP4"
csv_rel = "drone_footage/23-02-01_FR_F01.csv"

# Create dataset
dataset = DroneVideoDataset(
    dataset_path=str(root),
    video_path=video_rel,
    csv_path=csv_rel,
    normalize_time=True,
    frame_stride=1,
    max_frames=None,
)

print("DATASET TIME AND ALTITUDE ANALYSIS:")
print(f"Dataset frames: {len(dataset)}")
print(f"Video FPS: {dataset._fps}")
print(f"Time offset: {dataset.time_offset}")
print(f"Normalize time: {dataset._timeline.normalize_time}")
print(f"CSV start time: {dataset._timeline.start_time:.2f}s")
print(f"CSV end time: {dataset._timeline.end_time:.2f}s")
print(f"CSV duration: {dataset._timeline.duration:.2f}s")
print()

# Compute expected frame times
print(f"{'Frame#':<8} {'Frame Time (computed)':<22} {'CSV Time':<12} {'Altitude':<10} {'ENU Pos':<30}")
print("-" * 90)

frame_indices = [0, 30, 60, 90, 120, 150, 180, 210, dataset._frame_count - 1]
for fidx in frame_indices:
    if fidx >= len(dataset):
        continue
    frame, tele = dataset[fidx]
    computed_frame_time = fidx / dataset._fps + dataset.time_offset
    alt = tele.alt_m
    enu = tele.pos_enu
    print(f"{fidx:<8} {computed_frame_time:<22.4f} {tele.time_s:<12.4f} {alt:<10.2f} [{enu[0]:7.1f}, {enu[1]:7.1f}, {enu[2]:7.1f}]")

dataset.close()
print()

# Also check raw timeline sampling
print("\nDIRECT TIMELINE SAMPLING (from TelemetryTimeline):")
timeline = TelemetryTimeline(root / csv_rel, normalize_time=True)
print(f"{'Time(s)':<10} {'Altitude(m)':<12}")
print("-" * 22)
for t in [0, 10, 20, 30, 40, 50, 70, 100, 139]:
    tele = timeline.sample(t)
    print(f"{t:<10.1f} {tele.alt_m:<12.2f}")
