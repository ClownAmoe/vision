"""Motion estimation for downward-facing drone video using telemetry scale."""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import argparse
import logging
import os
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt

from drone_parser import DroneVideoDataset, TelemetrySample
from drone_feature_matching import FeatureMatcher, DetectorType

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler()])


@dataclass
class FlowEstimate:
    delta_enu: np.ndarray
    n_inliers: int


@dataclass
class TrajectoryState:
    t_pos: np.ndarray = None

    def __post_init__(self):
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


def detect_turn_frames(gt_traj: np.ndarray, heading_threshold_deg: float = 2.0) -> np.ndarray:
    n = len(gt_traj)
    turn_mask = np.zeros(n, dtype=bool)
    if n < 3:
        return turn_mask

    dx = np.diff(gt_traj[:, 0])
    dy = np.diff(gt_traj[:, 1])
    heading = np.unwrap(np.arctan2(dy, dx))
    d_heading = np.diff(heading)
    threshold = np.deg2rad(heading_threshold_deg)

    turn_mask[2:] = np.abs(d_heading) > threshold
    return turn_mask


def _safe_rate(mask: np.ndarray, success: np.ndarray) -> float:
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(success[mask]))


class PlanarFlowEstimator:
    """Estimate planar translation from optical flow for a downward-facing camera."""

    def __init__(
        self,
        K: np.ndarray,
        min_alt_m: float = 1.0,
        ransac_threshold: float = 2.0,
        use_yaw: bool = True,
        yaw_offset_deg: float = 0.0,
        yaw_source: str = "gimbal",
        flow_model: str = "affine",
    ):
        self.K = K.astype(np.float64)
        self.min_alt_m = float(min_alt_m)
        self.ransac_threshold = float(ransac_threshold)
        self.use_yaw = bool(use_yaw)
        self.yaw_offset_deg = float(yaw_offset_deg)
        self.yaw_source = yaw_source
        self.flow_model = flow_model

        self.fx = float(K[0, 0])
        self.fy = float(K[1, 1])

    def estimate_with_reason(
        self,
        pts_prev: np.ndarray,
        pts_curr: np.ndarray,
        tele_curr: TelemetrySample,
    ) -> Tuple[Optional[FlowEstimate], str]:
        if tele_curr.alt_m < self.min_alt_m:
            return None, "alt"

        tx, ty, n_inliers = self._estimate_pixel_shift(pts_prev, pts_curr)
        if tx is None or ty is None:
            return None, "flow"

        dx_cam = -tx * tele_curr.alt_m / self.fx
        dy_cam = -ty * tele_curr.alt_m / self.fy

        d_body_x = -dy_cam
        d_body_y = dx_cam

        yaw_deg = tele_curr.yaw_deg
        if self.yaw_source == "gimbal":
            yaw_deg = tele_curr.gimbal_yaw_deg
        elif self.yaw_source == "none":
            yaw_deg = 0.0

        yaw = 0.0
        if self.use_yaw:
            yaw = np.deg2rad(yaw_deg + self.yaw_offset_deg)

        d_north = np.cos(yaw) * d_body_x - np.sin(yaw) * d_body_y
        d_east = np.sin(yaw) * d_body_x + np.cos(yaw) * d_body_y

        delta_enu = np.array([d_east, d_north, 0.0], dtype=np.float64)
        return FlowEstimate(delta_enu=delta_enu, n_inliers=n_inliers), ""

    def _estimate_pixel_shift(
        self,
        pts_prev: np.ndarray,
        pts_curr: np.ndarray,
    ) -> Tuple[Optional[float], Optional[float], int]:
        if self.flow_model == "median":
            flow = pts_curr - pts_prev
            tx = float(np.median(flow[:, 0]))
            ty = float(np.median(flow[:, 1]))
            return tx, ty, len(flow)

        M, inliers = cv2.estimateAffinePartial2D(
            pts_prev,
            pts_curr,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
        )
        if M is None:
            return None, None, 0

        tx = float(M[0, 2])
        ty = float(M[1, 2])
        n_inliers = int(inliers.sum()) if inliers is not None else len(pts_prev)
        return tx, ty, n_inliers


def run_odometry_pipeline(
    dataset: DroneVideoDataset,
    feature_matcher: FeatureMatcher,
    estimator: PlanarFlowEstimator,
    max_frames: Optional[int] = None,
    min_matches: int = 12,
    log_every: int = 100,
) -> Tuple[np.ndarray, np.ndarray, PipelineMetrics]:
    n = len(dataset) if max_frames is None else min(max_frames, len(dataset))

    estimated_traj = np.zeros((n, 3), dtype=np.float64)
    gt_traj = np.zeros((n, 3), dtype=np.float64)
    frame_times_sec = np.zeros(n, dtype=np.float64)
    fps_per_frame = np.zeros(n, dtype=np.float64)
    pose_success = np.zeros(n, dtype=bool)
    inliers_per_frame = np.zeros(n, dtype=np.int32)

    state = TrajectoryState()

    fail_no_matches = 0
    fail_alt = 0
    fail_flow = 0

    img0, tele0 = dataset[0]
    _ = img0
    gt_traj[0] = tele0.pos_enu

    for idx in range(1, n):
        t_start = time.perf_counter()
        img_prev, tele_prev = dataset[idx - 1]
        img_curr, tele_curr = dataset[idx]

        match_result = feature_matcher.match(img_prev, img_curr, min_matches=min_matches)
        if match_result is None:
            fail_no_matches += 1
            estimated_traj[idx] = estimated_traj[idx - 1]
            gt_traj[idx] = tele_curr.pos_enu
            frame_times_sec[idx] = time.perf_counter() - t_start
            fps_per_frame[idx] = 1.0 / max(frame_times_sec[idx], 1e-9)
            continue

        estimate, reason = estimator.estimate_with_reason(
            match_result.pts_prev,
            match_result.pts_curr,
            tele_curr,
        )

        if estimate is None:
            if reason == "alt":
                fail_alt += 1
            elif reason == "flow":
                fail_flow += 1
        else:
            state.t_pos = state.t_pos + estimate.delta_enu.reshape(3, 1)
            pose_success[idx] = True
            inliers_per_frame[idx] = int(estimate.n_inliers)

        estimated_traj[idx] = state.t_pos.ravel()
        gt_traj[idx] = tele_curr.pos_enu
        frame_times_sec[idx] = time.perf_counter() - t_start
        fps_per_frame[idx] = 1.0 / max(frame_times_sec[idx], 1e-9)

        if log_every > 0 and idx % log_every == 0:
            err = np.linalg.norm(estimated_traj[idx] - gt_traj[idx])
            logging.info(
                f"[{idx:04d}/{n}] pos=({state.t_pos[0,0]:7.1f}, {state.t_pos[1,0]:7.1f}) err={err:.2f} m"
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
        f"alt={fail_alt} "
        f"flow={fail_flow}"
    )
    return estimated_traj, gt_traj, metrics


def run_detector_experiment(
    dataset: DroneVideoDataset,
    estimator: PlanarFlowEstimator,
    detector_type: DetectorType,
    max_frames: Optional[int] = None,
    turn_heading_threshold_deg: float = 2.0,
    min_matches: int = 12,
    log_every: int = 100,
    align_ate: bool = True,
) -> Optional[ExperimentResult]:
    try:
        matcher = FeatureMatcher(detector_type)
    except cv2.error as e:
        logging.warning(f"Detector {detector_type.name} unavailable: {e}")
        return None

    logging.info(f"\n=== Experiment: {detector_type.name} ===")
    estimated_traj, gt_traj, metrics = run_odometry_pipeline(
        dataset,
        matcher,
        estimator,
        max_frames=max_frames,
        min_matches=min_matches,
        log_every=log_every,
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


def plot_trajectories(results: List[ExperimentResult]):
    fig = plt.figure(figsize=(10, 7))
    gt = results[0].gt_traj
    plt.plot(gt[:, 0], gt[:, 1], label="Ground Truth", color="black", linewidth=2.0)

    colors = {
        "ORB": "#e63946",
        "OPTICAL_FLOW": "#2a9d8f",
    }
    for res in results:
        plt.plot(
            res.estimated_traj[:, 0],
            res.estimated_traj[:, 1],
            label=f"{res.detector_name} (ATE={res.ate_rmse:.2f}m)",
            linestyle="--",
            color=colors.get(res.detector_name, None),
        )

    plt.title("Ground Truth vs Estimated Trajectories")
    plt.xlabel("East Position (meters)")
    plt.ylabel("North Position (meters)")
    plt.legend()
    plt.grid(True)
    return fig


def plot_fps_histogram(results: List[ExperimentResult]):
    names = [r.detector_name for r in results]
    fps_values = [r.avg_fps for r in results]

    fig = plt.figure(figsize=(8, 5))
    colors = ["#e63946" if n == "ORB" else "#2a9d8f" for n in names]
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
        scale = 1.0 if var_src < 1e-12 else float(np.sum(D * np.diag(S)) / var_src)
    else:
        scale = 1.0

    t = mu_dst - scale * (R @ mu_src)
    aligned = (scale * (R @ src.T)).T + t
    return aligned


def parse_args() -> argparse.Namespace:
    parser_args = argparse.ArgumentParser(description="Drone video motion estimation")
    parser_args.add_argument(
        "--video_path", type=str, default="drone_footage/23-02-01_FR_F01_combined.MP4",
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

    parser_args.add_argument(
        "--target_fps", type=float, default=5.0,
        help="Target processing FPS (used to compute stride)",
    )
    parser_args.add_argument(
        "--frame_stride", type=int, default=None,
        help="Override frame stride (integer)",
    )
    parser_args.add_argument(
        "--start_frame", type=int, default=0,
        help="Start frame index",
    )
    parser_args.add_argument(
        "--max_frames", type=int, default=None,
        help="Maximum frames to process",
    )

    parser_args.add_argument(
        "--detectors", nargs="+", default=["OPTICAL_FLOW"],
        help="Detectors to test: ORB OPTICAL_FLOW",
    )
    parser_args.add_argument(
        "--min_matches", type=int, default=12,
        help="Minimum matches required between frames",
    )

    parser_args.add_argument(
        "--ransac_threshold", type=float, default=2.0,
        help="RANSAC reprojection threshold (pixels)",
    )
    parser_args.add_argument(
        "--min_alt", type=float, default=1.0,
        help="Minimum altitude (m) to accept translation estimate",
    )
    parser_args.add_argument(
        "--flow_model",
        choices=["affine", "median"],
        default="median",
        help="Flow model used to estimate pixel shift",
    )
    yaw_group = parser_args.add_mutually_exclusive_group()
    yaw_group.add_argument(
        "--use_yaw",
        dest="no_yaw",
        action="store_false",
        help="Rotate optical flow estimates using telemetry yaw",
    )
    yaw_group.add_argument(
        "--no_yaw",
        dest="no_yaw",
        action="store_true",
        help="Ignore telemetry yaw when rotating to ENU",
    )
    parser_args.set_defaults(no_yaw=True)
    parser_args.add_argument(
        "--yaw_offset", type=float, default=0.0,
        help="Yaw offset (deg) added before rotating to ENU",
    )
    parser_args.add_argument(
        "--yaw_source",
        choices=["gimbal", "osd", "none"],
        default="gimbal",
        help="Which yaw source to use for image-to-ENU rotation",
    )
    align_group = parser_args.add_mutually_exclusive_group()
    align_group.add_argument(
        "--align_ate",
        dest="align_ate",
        action="store_true",
        help="Align trajectories before ATE/plots",
    )
    align_group.add_argument(
        "--no-align_ate",
        dest="align_ate",
        action="store_false",
        help="Do not align trajectories before ATE/plots",
    )
    parser_args.set_defaults(align_ate=True)

    parser_args.add_argument("--fx", type=float, default=None)
    parser_args.add_argument("--fy", type=float, default=None)
    parser_args.add_argument("--cx", type=float, default=None)
    parser_args.add_argument("--cy", type=float, default=None)

    parser_args.add_argument(
        "--log_every", type=int, default=100,
        help="Log progress every N frames",
    )
    parser_args.add_argument("--no_plot", action="store_true", help="Do not show plots")
    parser_args.add_argument(
        "--save_plots_dir", type=str, default=None, help="Output directory for plots"
    )

    return parser_args.parse_args()


def main() -> None:
    args = parse_args()

    dataset = DroneVideoDataset(
        "dataset",
        video_path=args.video_path,
        csv_path=args.csv_path,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        time_column=args.time_column,
        time_offset=args.time_offset,
        normalize_time=args.normalize_time,
        target_fps=args.target_fps,
        frame_stride=args.frame_stride,
    )
    logging.info(dataset.summary())
    if args.max_frames is not None:
        logging.info(f"Limiting to max_frames={args.max_frames}")
    if dataset.frame_stride > 1:
        logging.info(
            "Downsampling via frame_stride. Use --target_fps 30 or --frame_stride 1 for all frames."
        )

    frame0, _ = dataset[0]
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

    estimator = PlanarFlowEstimator(
        K,
        min_alt_m=args.min_alt,
        ransac_threshold=args.ransac_threshold,
        use_yaw=not args.no_yaw,
        yaw_offset_deg=args.yaw_offset,
        yaw_source=args.yaw_source,
        flow_model=args.flow_model,
    )

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
        result = run_detector_experiment(
            dataset,
            estimator,
            detector,
            max_frames=args.max_frames,
            min_matches=args.min_matches,
            log_every=args.log_every,
            align_ate=args.align_ate,
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


if __name__ == "__main__":
    main()
