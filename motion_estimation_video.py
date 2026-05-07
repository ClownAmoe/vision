"""
Stage 3: Motion estimation for video + CSV telemetry.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import time
import os
import argparse
import logging

import cv2
import numpy as np
import matplotlib.pyplot as plt

from kitti_parser_video import VideoTelemetryParser, TelemetrySample
from feature_matching_video import FeatureMatcher, DetectorType

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler()])


@dataclass
class PoseEstimate:
    """Pose estimate between two frames."""
    R: np.ndarray
    t: np.ndarray
    t_scaled: np.ndarray
    scale: float
    inliers_mask: np.ndarray
    n_inliers: int


@dataclass
class TrajectoryState:
    """Accumulated camera pose."""
    R_pos: np.ndarray = None
    t_pos: np.ndarray = None

    def __post_init__(self):
        if self.R_pos is None:
            self.R_pos = np.eye(3, dtype=np.float64)
        if self.t_pos is None:
            self.t_pos = np.zeros((3, 1), dtype=np.float64)


@dataclass
class PipelineMetrics:
    frame_times_sec: np.ndarray
    fps_per_frame: np.ndarray
    pose_success: np.ndarray
    inliers_per_frame: np.ndarray


@dataclass
class ExperimentResult:
    detector_name: str
    estimated_traj: np.ndarray
    gt_traj: np.ndarray
    metrics: PipelineMetrics
    avg_fps: float
    ate_rmse: float
    pose_success_rate: float
    turn_success_rate: float
    straight_success_rate: float


def compute_ate_rmse(estimated_traj: np.ndarray, gt_traj: np.ndarray, start_idx: int = 1) -> float:
    est = estimated_traj[start_idx:]
    gt = gt_traj[start_idx:]
    if est.size == 0 or gt.size == 0:
        return float("nan")
    errors = np.linalg.norm(est - gt, axis=1)
    return float(np.sqrt(np.mean(errors ** 2)))


def detect_turn_frames(gt_traj: np.ndarray, heading_threshold_deg: float = 1.5) -> np.ndarray:
    n = len(gt_traj)
    turn_mask = np.zeros(n, dtype=bool)
    if n < 3:
        return turn_mask

    dx = np.diff(gt_traj[:, 0])
    dz = np.diff(gt_traj[:, 2])
    heading = np.unwrap(np.arctan2(dz, dx))
    d_heading = np.diff(heading)
    threshold = np.deg2rad(heading_threshold_deg)

    turn_mask[2:] = np.abs(d_heading) > threshold
    return turn_mask


def _safe_rate(mask: np.ndarray, success: np.ndarray) -> float:
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(success[mask]))


class TelemetryPoseBuilder:
    """Build 4x4 poses from telemetry rows (lat/lon/alt + yaw/pitch/roll)."""

    EARTH_RADIUS_M = 6378137.0
    FT_TO_M = 0.3048

    def __init__(self):
        self._origin_lat = None
        self._origin_lon = None
        self._origin_alt_m = None
        self._warned_angles = False

    def _position_from_sample(self, sample: TelemetrySample) -> np.ndarray:
        lat = sample.lat
        lon = sample.lon
        alt_m = sample.alt_m

        if self._origin_lat is None:
            self._origin_lat = lat
            self._origin_lon = lon
            self._origin_alt_m = alt_m

        lat0_rad = np.deg2rad(self._origin_lat)
        d_lat = np.deg2rad(lat - self._origin_lat)
        d_lon = np.deg2rad(lon - self._origin_lon)

        east = d_lon * np.cos(lat0_rad) * self.EARTH_RADIUS_M
        north = d_lat * self.EARTH_RADIUS_M
        up = alt_m - self._origin_alt_m

        return np.array([east, up, north], dtype=np.float64)

    def _rotation_from_sample(self, sample: TelemetrySample) -> np.ndarray:
        if not np.isfinite([sample.yaw_deg, sample.pitch_deg, sample.roll_deg]).all():
            if not self._warned_angles:
                logging.warning("Missing yaw/pitch/roll; using identity rotation.")
                self._warned_angles = True
            return np.eye(3, dtype=np.float64)

        yaw_r = np.deg2rad(sample.yaw_deg)
        pitch_r = np.deg2rad(sample.pitch_deg)
        roll_r = np.deg2rad(sample.roll_deg)

        Rz = np.array(
            [
                [np.cos(yaw_r), -np.sin(yaw_r), 0.0],
                [np.sin(yaw_r), np.cos(yaw_r), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        Ry = np.array(
            [
                [np.cos(pitch_r), 0.0, np.sin(pitch_r)],
                [0.0, 1.0, 0.0],
                [-np.sin(pitch_r), 0.0, np.cos(pitch_r)],
            ],
            dtype=np.float64,
        )
        Rx = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(roll_r), -np.sin(roll_r)],
                [0.0, np.sin(roll_r), np.cos(roll_r)],
            ],
            dtype=np.float64,
        )

        return Rz @ Ry @ Rx

    def sample_to_pose(self, sample: TelemetrySample) -> np.ndarray:
        t = self._position_from_sample(sample)
        R = self._rotation_from_sample(sample)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T


class MotionEstimator:
    def __init__(
        self,
        K: np.ndarray,
        min_scale: float = 0.01,
        ransac_prob: float = 0.999,
        ransac_threshold: float = 2.0,
        scale_mode: str = "xy",
    ):
        self.K = K.astype(np.float64)
        self.min_scale = float(min_scale)
        self.ransac_prob = ransac_prob
        self.ransac_threshold = ransac_threshold
        self.scale_mode = scale_mode

        self._focal = float(K[0, 0])
        self._pp = (float(K[0, 2]), float(K[1, 2]))

    def estimate_with_reason(
        self,
        pts_prev: np.ndarray,
        pts_curr: np.ndarray,
        pose_prev: np.ndarray,
        pose_curr: np.ndarray,
    ) -> Tuple[Optional[PoseEstimate], str]:
        delta = pose_curr[:3, 3] - pose_prev[:3, 3]
        if self.scale_mode == "xy":
            scale = float(np.linalg.norm(delta[[0, 2]]))
        else:
            scale = float(np.linalg.norm(delta))
        if scale < self.min_scale:
            return None, "scale"

        E, mask = cv2.findEssentialMat(
            pts_prev,
            pts_curr,
            focal=self._focal,
            pp=self._pp,
            method=cv2.RANSAC,
            prob=self.ransac_prob,
            threshold=self.ransac_threshold,
        )
        if E is None:
            return None, "essential"

        n_inliers, R, t, mask_pose = cv2.recoverPose(
            E,
            pts_prev,
            pts_curr,
            focal=self._focal,
            pp=self._pp,
            mask=mask,
        )
        if n_inliers < 8:
            return None, "inliers"

        t_scaled = t * scale

        return PoseEstimate(
            R=R,
            t=t,
            t_scaled=t_scaled,
            scale=scale,
            inliers_mask=mask_pose.ravel().astype(bool),
            n_inliers=n_inliers,
        ), ""

    def estimate(
        self,
        pts_prev: np.ndarray,
        pts_curr: np.ndarray,
        pose_prev: np.ndarray,
        pose_curr: np.ndarray,
    ) -> Optional[PoseEstimate]:
        estimate, _ = self.estimate_with_reason(pts_prev, pts_curr, pose_prev, pose_curr)
        return estimate

    @staticmethod
    def update_trajectory(state: TrajectoryState, estimate: PoseEstimate) -> TrajectoryState:
        new_t = state.t_pos + state.R_pos @ estimate.t_scaled
        new_R = estimate.R @ state.R_pos
        return TrajectoryState(R_pos=new_R, t_pos=new_t)


def run_odometry_pipeline(
    parser,
    feature_matcher,
    estimator: MotionEstimator,
    max_frames: Optional[int] = None,
    min_matches: int = 8,
) -> Tuple[np.ndarray, np.ndarray, PipelineMetrics]:
    n = len(parser) if max_frames is None else min(max_frames, len(parser))

    estimated_traj = np.zeros((n, 3), dtype=np.float64)
    gt_traj = np.zeros((n, 3), dtype=np.float64)
    frame_times_sec = np.zeros(n, dtype=np.float64)
    fps_per_frame = np.zeros(n, dtype=np.float64)
    pose_success = np.zeros(n, dtype=bool)
    inliers_per_frame = np.zeros(n, dtype=np.int32)

    state = TrajectoryState()
    telemetry_builder = TelemetryPoseBuilder()

    fail_no_matches = 0
    fail_scale = 0
    fail_essential = 0
    fail_inliers = 0

    img0, tele0 = parser[0]
    pose0 = telemetry_builder.sample_to_pose(tele0)
    _ = img0
    gt_traj[0] = pose0[:3, 3]

    for idx in range(1, n):
        t_frame_start = time.perf_counter()
        img_prev, tele_prev = parser[idx - 1]
        img_curr, tele_curr = parser[idx]

        pose_prev = telemetry_builder.sample_to_pose(tele_prev)
        pose_curr = telemetry_builder.sample_to_pose(tele_curr)

        match_result = feature_matcher.match(img_prev, img_curr, min_matches=min_matches)
        if match_result is None:
            fail_no_matches += 1
            estimated_traj[idx] = estimated_traj[idx - 1]
            gt_traj[idx] = pose_curr[:3, 3]
            frame_times_sec[idx] = time.perf_counter() - t_frame_start
            fps_per_frame[idx] = 1.0 / max(frame_times_sec[idx], 1e-9)
            continue

        estimate, reason = estimator.estimate_with_reason(
            match_result.pts_prev,
            match_result.pts_curr,
            pose_prev,
            pose_curr,
        )

        if estimate is None:
            if reason == "scale":
                fail_scale += 1
            elif reason == "essential":
                fail_essential += 1
            elif reason == "inliers":
                fail_inliers += 1

        if estimate is not None:
            state = MotionEstimator.update_trajectory(state, estimate)
            pose_success[idx] = True
            inliers_per_frame[idx] = int(estimate.n_inliers)

        estimated_traj[idx] = state.t_pos.ravel()
        gt_traj[idx] = pose_curr[:3, 3]
        frame_times_sec[idx] = time.perf_counter() - t_frame_start
        fps_per_frame[idx] = 1.0 / max(frame_times_sec[idx], 1e-9)

        if idx % 100 == 0:
            err = np.linalg.norm(estimated_traj[idx] - gt_traj[idx])
            print(
                f"[{idx:04d}/{n}]  "
                f"pos=({state.t_pos[0,0]:7.1f}, {state.t_pos[2,0]:7.1f})  "
                f"err={err:.2f} m"
            )

    metrics = PipelineMetrics(
        frame_times_sec=frame_times_sec,
        fps_per_frame=fps_per_frame,
        pose_success=pose_success,
        inliers_per_frame=inliers_per_frame,
    )
    logging.info(
        "Failures: "
        f"no_matches={fail_no_matches} "
        f"scale={fail_scale} "
        f"essential={fail_essential} "
        f"inliers={fail_inliers}"
    )
    return estimated_traj, gt_traj, metrics


def run_detector_experiment(
    parser,
    estimator: MotionEstimator,
    detector_type: DetectorType,
    max_frames: Optional[int] = None,
    turn_heading_threshold_deg: float = 1.5,
    align_ate: bool = False,
    min_matches: int = 8,
    flow_params: Optional[dict] = None,
) -> Optional[ExperimentResult]:
    try:
        matcher_kwargs = flow_params or {}
        matcher = FeatureMatcher(detector_type, **matcher_kwargs)
    except cv2.error as e:
        logging.warning(f"Detector {detector_type.name} unavailable: {e}")
        return None

    logging.info(f"\n=== Experiment: {detector_type.name} ===")
    estimated_traj, gt_traj, metrics = run_odometry_pipeline(
        parser, matcher, estimator, max_frames=max_frames, min_matches=min_matches
    )

    if align_ate:
        estimated_traj = align_trajectories_umeyama(estimated_traj, gt_traj, with_scale=True)

    valid = np.arange(len(metrics.fps_per_frame)) > 0
    avg_fps = float(np.mean(metrics.fps_per_frame[valid]))
    ate_rmse = compute_ate_rmse(estimated_traj, gt_traj, start_idx=1)
    pose_success_rate = float(np.mean(metrics.pose_success[valid]))

    turn_mask = detect_turn_frames(gt_traj, heading_threshold_deg=turn_heading_threshold_deg)
    turn_success_rate = _safe_rate(turn_mask & valid, metrics.pose_success)
    straight_success_rate = _safe_rate((~turn_mask) & valid, metrics.pose_success)

    return ExperimentResult(
        detector_name=detector_type.name,
        estimated_traj=estimated_traj,
        gt_traj=gt_traj,
        metrics=metrics,
        avg_fps=avg_fps,
        ate_rmse=ate_rmse,
        pose_success_rate=pose_success_rate,
        turn_success_rate=turn_success_rate,
        straight_success_rate=straight_success_rate,
    )


def plot_fps_histogram(results: List[ExperimentResult]):
    names = [r.detector_name for r in results]
    fps_values = [r.avg_fps for r in results]

    fig = plt.figure(figsize=(8, 5))
    color_map = {
        "SIFT": "#2a9d8f",
        "SURF": "#e9c46a",
        "ORB": "#e76f51",
        "OPTICAL_FLOW": "#264653",
    }
    colors = [color_map.get(name, "#6c757d") for name in names]
    bars = plt.bar(names, fps_values, color=colors)
    for bar, fps in zip(bars, fps_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{fps:.1f}",
            ha="center",
            va="bottom",
        )
    plt.title("Average FPS by Detector")
    plt.ylabel("FPS")
    plt.grid(axis="y", alpha=0.3)
    return fig


def plot_trajectories(results: List[ExperimentResult]):
    fig = plt.figure(figsize=(10, 7))
    gt = results[0].gt_traj
    plt.plot(gt[:, 0], gt[:, 2], label="Ground Truth", color="black", linewidth=2.0)

    colors = {
        "SIFT": "#1d3557",
        "SURF": "#457b9d",
        "ORB": "#e63946",
        "OPTICAL_FLOW": "#2a9d8f",
    }
    for res in results:
        plt.plot(
            res.estimated_traj[:, 0],
            res.estimated_traj[:, 2],
            label=f"{res.detector_name} (ATE={res.ate_rmse:.2f}m)",
            linestyle="--",
            color=colors.get(res.detector_name, None),
        )

    plt.title("Ground Truth vs Estimated Trajectories")
    plt.xlabel("X Position (meters)")
    plt.ylabel("Z Position (meters)")
    plt.legend()
    plt.grid(True)
    return fig


def align_trajectories_umeyama(
    estimated_traj: np.ndarray,
    gt_traj: np.ndarray,
    with_scale: bool = True,
) -> np.ndarray:
    if len(estimated_traj) < 2 or len(gt_traj) < 2:
        return estimated_traj

    src = estimated_traj.astype(np.float64)
    dst = gt_traj.astype(np.float64)

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_centered = src - mu_src
    dst_centered = dst - mu_dst

    cov = (dst_centered.T @ src_centered) / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0

    R = U @ S @ Vt

    if with_scale:
        var_src = np.mean(np.sum(src_centered ** 2, axis=1))
        if var_src < 1e-12:
            scale = 1.0
        else:
            scale = float(np.sum(D * np.diag(S)) / var_src)
    else:
        scale = 1.0

    t = mu_dst - scale * (R @ mu_src)
    aligned = (scale * (R @ src.T)).T + t
    return aligned


if __name__ == "__main__":
    parser_args = argparse.ArgumentParser(description="Motion Estimation for Video + CSV")
    parser_args.add_argument(
        "--max_frames", type=int, default=None,
        help="Maximum number of frames to process",
    )
    parser_args.add_argument(
        "--detectors", nargs="+", default=["SIFT", "SURF", "ORB", "OPTICAL_FLOW"],
        help="Detectors to test",
    )
    parser_args.add_argument(
        "--turn_heading_threshold_deg", type=float, default=1.5,
        help="Heading change threshold in degrees",
    )
    parser_args.add_argument("--no_plot", action="store_true", help="Do not show plots")
    parser_args.add_argument(
        "--save_plots_dir", type=str, default=None, help="Output directory for plots"
    )
    parser_args.add_argument(
        "--video_path", type=str, default="drone_footage/23-02-01_FR_F01_V01.MP4",
        help="MP4 path relative to dataset/",
    )
    parser_args.add_argument(
        "--csv_path", type=str, default="drone_footage/23-02-01_FR_F01.csv",
        help="CSV path relative to dataset/",
    )

    parser_args.add_argument(
        "--time_column", type=str, default="OSD.flyTime [s]",
        help="CSV column name with time in seconds",
    )
    parser_args.add_argument(
        "--time_offset", type=float, default=0.0,
        help="Offset added to frame time (seconds)",
    )
    normalize_group = parser_args.add_mutually_exclusive_group()
    normalize_group.add_argument(
        "--normalize_time",
        dest="normalize_time",
        action="store_true",
        help="Normalize telemetry time to start at 0",
    )
    normalize_group.add_argument(
        "--no-normalize_time",
        dest="normalize_time",
        action="store_false",
        help="Use raw telemetry time values",
    )
    parser_args.set_defaults(normalize_time=True)
    parser_args.add_argument("--fx", type=float, default=None)
    parser_args.add_argument("--fy", type=float, default=None)
    parser_args.add_argument("--cx", type=float, default=None)
    parser_args.add_argument("--cy", type=float, default=None)
    parser_args.add_argument(
        "--min_scale", type=float, default=0.01,
        help="Minimum GT distance between frames in meters",
    )
    parser_args.add_argument(
        "--scale_mode",
        choices=["xy", "3d"],
        default="xy",
        help="Scale mode: xy for horizontal-only, 3d for full 3D",
    )
    parser_args.add_argument(
        "--min_matches", type=int, default=8,
        help="Minimum matches required between frames",
    )
    parser_args.add_argument(
        "--ransac_threshold", type=float, default=2.0,
        help="RANSAC reprojection threshold (pixels)",
    )
    parser_args.add_argument(
        "--ransac_prob", type=float, default=0.999,
        help="RANSAC success probability",
    )
    parser_args.add_argument(
        "--flow_max_corners", type=int, default=3000,
        help="Max corners for optical flow",
    )
    parser_args.add_argument(
        "--flow_quality_level", type=float, default=0.01,
        help="Quality level for GFTT",
    )
    parser_args.add_argument(
        "--flow_min_distance", type=float, default=7.0,
        help="Min distance between corners",
    )
    parser_args.add_argument(
        "--flow_block_size", type=int, default=7,
        help="Block size for GFTT",
    )
    parser_args.add_argument(
        "--flow_fb_max_error", type=float, default=1.5,
        help="Forward-backward error threshold (pixels)",
    )

    align_group = parser_args.add_mutually_exclusive_group()
    align_group.add_argument(
        "--align_ate",
        dest="align_ate",
        action="store_true",
        help="Align trajectories before ATE",
    )
    align_group.add_argument(
        "--no-align_ate",
        dest="align_ate",
        action="store_false",
        help="Do not align trajectories before ATE",
    )
    parser_args.set_defaults(align_ate=True)

    args = parser_args.parse_args()

    DATASET_PATH = "dataset/"

    parser = VideoTelemetryParser(
        DATASET_PATH,
        video_path=args.video_path,
        csv_path=args.csv_path,
        time_column=args.time_column,
        time_offset=args.time_offset,
        normalize_time=args.normalize_time,
    )
    logging.info(parser.summary())

    frame0, _ = parser[0]
    h, w = frame0.shape[:2]
    fx = args.fx if args.fx is not None else max(w, h)
    fy = args.fy if args.fy is not None else max(w, h)
    cx = args.cx if args.cx is not None else w / 2.0
    cy = args.cy if args.cy is not None else h / 2.0
    if args.fx is None or args.fy is None or args.cx is None or args.cy is None:
        logging.warning("Camera intrinsics not provided; using image-based defaults.")

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    logging.info("Camera matrix K:")
    logging.info(K)

    estimator = MotionEstimator(
        K,
        min_scale=args.min_scale,
        ransac_prob=args.ransac_prob,
        ransac_threshold=args.ransac_threshold,
        scale_mode=args.scale_mode,
    )
    max_frames = args.max_frames if args.max_frames is not None else len(parser)

    if args.align_ate:
        logging.info("Aligning trajectories for ATE/plots (Umeyama)")

    selected_detectors: List[DetectorType] = []
    for name in args.detectors:
        key = name.upper()
        if key not in DetectorType.__members__:
            logging.warning(f"Unknown detector '{name}', skipping")
            continue
        selected_detectors.append(DetectorType[key])

    if not selected_detectors:
        raise ValueError("No valid detectors selected")

    results: List[ExperimentResult] = []
    for detector in selected_detectors:
        flow_params = dict(
            flow_max_corners=args.flow_max_corners,
            flow_quality_level=args.flow_quality_level,
            flow_min_distance=args.flow_min_distance,
            flow_block_size=args.flow_block_size,
            flow_fb_max_error=args.flow_fb_max_error,
        )
        result = run_detector_experiment(
            parser=parser,
            estimator=estimator,
            detector_type=detector,
            max_frames=max_frames,
            turn_heading_threshold_deg=args.turn_heading_threshold_deg,
            align_ate=args.align_ate,
            min_matches=args.min_matches,
            flow_params=flow_params,
        )
        if result is not None:
            results.append(result)

    if not results:
        raise RuntimeError("No experiments completed successfully")

    logging.info("\n===== SUMMARY =====")
    for r in results:
        logging.info(
            f"{r.detector_name:>4s} | FPS(avg)={r.avg_fps:8.2f} | "
            f"ATE(RMSE)={r.ate_rmse:8.3f} m | "
            f"success={100.0*r.pose_success_rate:6.2f}% | "
            f"turn_success={100.0*r.turn_success_rate if not np.isnan(r.turn_success_rate) else float('nan'):6.2f}% | "
            f"straight_success={100.0*r.straight_success_rate if not np.isnan(r.straight_success_rate) else float('nan'):6.2f}%"
        )

    fps_fig = plot_fps_histogram(results)
    traj_fig = plot_trajectories(results)

    if args.save_plots_dir:
        os.makedirs(args.save_plots_dir, exist_ok=True)
        fps_path = os.path.join(args.save_plots_dir, "fps_histogram.png")
        traj_path = os.path.join(args.save_plots_dir, "trajectories_comparison.png")
        fps_fig.savefig(fps_path, dpi=150, bbox_inches="tight")
        traj_fig.savefig(traj_path, dpi=150, bbox_inches="tight")
        logging.info(f"Saved FPS plot: {fps_path}")
        logging.info(f"Saved trajectory plot: {traj_path}")

    if not args.no_plot:
        plt.show()
