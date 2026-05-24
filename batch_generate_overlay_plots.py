from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional


DEFAULT_VIDEO_PATHS = [
    r"dataset/drone_footage/23-02-01_FR_F01_V01.MP4",
    r"dataset/drone_footage/23-02-01_FR_F01_V02.MP4",
    r"dataset/drone_footage/23-02-01_FR_F01_V03.MP4",
]


class PlotConfig:
    def __init__(self, output_dir: str, frame_skip: int, suffix: str = ""):
        self.output_dir = output_dir
        self.frame_skip = frame_skip
        self.suffix = suffix


PLOT_CONFIGS: List[PlotConfig] = [
    PlotConfig(output_dir=r"results/plots", frame_skip=1, suffix=""),
    PlotConfig(output_dir=r"results/plots/all_frames", frame_skip=1, suffix=""),
    PlotConfig(output_dir=r"results/plots/skip_1", frame_skip=1, suffix=""),
    PlotConfig(output_dir=r"results/plots/skip_3", frame_skip=3, suffix="_skip_3"),
    PlotConfig(output_dir=r"results/plots/skip_5", frame_skip=5, suffix="_skip_5"),
]


def run_motion_estimation(
    python_exe: str,
    background_map_path: str,
    drone_csv_path: str,
    drone_video_paths: Iterable[str],
    config: PlotConfig,
    detectors: Iterable[str],
) -> None:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        python_exe,
        "motion_estimation.py",
        "--dataset_type",
        "drone",
        "--drone_csv_path",
        drone_csv_path,
        "--drone_video_paths",
        *drone_video_paths,
        "--frame_skip",
        str(config.frame_skip),
        "--detectors",
        *detectors,
        "--background_map_path",
        background_map_path,
        "--save_plots_dir",
        str(output_dir),
        "--no_plot",
    ]

    subprocess.run(command, check=True)

    if config.suffix:
        for base_name in ("fps_histogram", "fps_timeseries", "trajectories_comparison"):
            source = output_dir / f"{base_name}.png"
            if source.exists():
                target = output_dir / f"{base_name}{config.suffix}.png"
                os.replace(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate all plot folders with a background map overlay."
    )
    parser.add_argument(
        "--background_map_path",
        required=True,
        help="Path to the screenshot or map image used as the background.",
    )
    parser.add_argument(
        "--drone_csv_path",
        default=r"dataset/drone_footage/23-02-01_FR_F01.csv",
        help="Path to the drone telemetry CSV.",
    )
    parser.add_argument(
        "--python_exe",
        default=sys.executable,
        help="Python interpreter to use for the child processes.",
    )
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=["ORB", "OPTICAL_FLOW"],
        help="Detectors to run.",
    )
    args = parser.parse_args()

    missing = [path for path in DEFAULT_VIDEO_PATHS if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing drone video files: " + ", ".join(missing))

    for config in PLOT_CONFIGS:
        run_motion_estimation(
            python_exe=args.python_exe,
            background_map_path=args.background_map_path,
            drone_csv_path=args.drone_csv_path,
            drone_video_paths=DEFAULT_VIDEO_PATHS,
            config=config,
            detectors=args.detectors,
        )


if __name__ == "__main__":
    main()
