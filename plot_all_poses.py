#!/usr/bin/env python3
"""Plot all telemetry poses (ENU) from a CSV and save to an image file.

Usage:
  python plot_all_poses.py --csv dataset/drone_footage/23-02-01_FR_F01.csv --out results/poses_plot.png
"""
import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from drone_parser import TelemetryTimeline


def main():
    p = argparse.ArgumentParser(description="Plot telemetry ENU poses from CSV")
    p.add_argument("--csv", type=str, default="dataset/drone_footage/23-02-01_FR_F01.csv")
    p.add_argument("--out", type=str, default="results/poses_plot.png")
    args = p.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    timeline = TelemetryTimeline(csv_path, normalize_time=True)

    poses = timeline._pos_enu  # Nx3: east, north, up
    east = poses[:, 0]
    north = poses[:, 1]
    up = poses[:, 2]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(east, north, lw=1.2, color="#1f77b4")
    ax.scatter([east[0]], [north[0]], color="green", s=50, label="start")
    ax.scatter([east[-1]], [north[-1]], color="red", s=50, label="end")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.set_title(f"Ground Track — {csv_path.name}")
    ax.legend()
    ax.grid(True)
    ax.axis("equal")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)

    # Also save altitude over time for reference
    alt_out = out_path.with_name(out_path.stem + "_alt.png")
    fig2, ax2 = plt.subplots(figsize=(8, 3))
    times = timeline._times
    ax2.plot(times, up, color="#ff7f0e")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Altitude (m)")
    ax2.set_title("Altitude over time")
    ax2.grid(True)
    fig2.tight_layout()
    fig2.savefig(alt_out, dpi=150)

    print(f"Saved ground-track plot: {out_path}")
    print(f"Saved altitude plot: {alt_out}")


if __name__ == "__main__":
    main()
