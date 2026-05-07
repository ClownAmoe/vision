#!/usr/bin/env python3
"""Concatenate multiple MP4 files (same resolution/fps) into one MP4.

This uses OpenCV to read frames sequentially and write them to a single output file.
Usage:
  python combine_videos.py --dir dataset/drone_footage --pattern "23-02-01_FR_F01_V*.MP4" --out dataset/drone_footage/23-02-01_FR_F01_combined.MP4
"""
import argparse
from pathlib import Path
import cv2
import sys


def find_videos(directory: Path, pattern: str):
    return sorted(directory.glob(pattern))


def combine(videos, out_path: Path):
    if len(videos) == 0:
        raise FileNotFoundError("No input videos found")

    cap0 = cv2.VideoCapture(str(videos[0]))
    if not cap0.isOpened():
        raise IOError(f"Failed to open {videos[0]}")

    fps = cap0.get(cv2.CAP_PROP_FPS)
    w = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap0.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not out.isOpened():
        raise IOError(f"Failed to open output writer {out_path}")

    total_frames = 0
    for v in videos:
        cap = cv2.VideoCapture(str(v))
        if not cap.isOpened():
            print(f"Warning: failed to open {v}, skipping", file=sys.stderr)
            continue
        print(f"Appending {v}...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            total_frames += 1
        cap.release()

    out.release()
    print(f"Wrote {total_frames} frames to {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=str, default="dataset/drone_footage")
    p.add_argument("--pattern", type=str, default="*.MP4")
    p.add_argument("--out", type=str, default="dataset/drone_footage/combined.MP4")
    return p.parse_args()


def main():
    args = parse_args()
    d = Path(args.dir)
    videos = find_videos(d, args.pattern)
    if len(videos) == 0:
        print(f"No videos found in {d} matching {args.pattern}")
        return
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combine(videos, out_path)


if __name__ == "__main__":
    main()
