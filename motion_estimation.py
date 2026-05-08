"""
Етап 3: Оцінка руху камери (Motion Estimation)
- Essential Matrix → Pose Recovery (R, t) з масштабом із Ground Truth
- Додано NadirMotionEstimator для знімків вертикально вниз (гомографія)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union
import time
import os

import cv2
import numpy as np
from kitti_parser import KITTIOdometryParser
from feature_matching import FeatureMatcher, DetectorType
import matplotlib.pyplot as plt
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler()])

@dataclass
class PoseEstimate:
    """Результат оцінки руху між двома кадрами."""
    R:              np.ndarray
    t:              np.ndarray
    t_scaled:       np.ndarray
    scale:          float
    inliers_mask:   np.ndarray
    n_inliers:      int


@dataclass
class TrajectoryState:
    """Накопичена глобальна позиція камери."""
    R_pos: np.ndarray = None
    t_pos: np.ndarray = None

    def __post_init__(self):
        if self.R_pos is None:
            self.R_pos = np.eye(3, dtype=np.float64)
        if self.t_pos is None:
            self.t_pos = np.zeros((3, 1), dtype=np.float64)


@dataclass
class PipelineMetrics:
    """Покадрові метрики роботи VO-пайплайна."""
    frame_times_sec: np.ndarray
    fps_per_frame: np.ndarray
    pose_success: np.ndarray
    inliers_per_frame: np.ndarray


@dataclass
class ExperimentResult:
    """Підсумок експерименту для конкретного детектора."""
    detector_name: str
    estimated_traj: np.ndarray
    gt_traj: np.ndarray
    metrics: PipelineMetrics
    avg_fps: float
    ate_rmse: float
    pose_success_rate: float
    turn_success_rate: float
    straight_success_rate: float


def compute_ate_rmse(
    estimated_traj: np.ndarray,
    gt_traj: np.ndarray,
    start_idx: int = 1,
) -> float:
    """
    Absolute Trajectory Error (ATE) у формі RMSE.
    """
    est = estimated_traj[start_idx:]
    gt = gt_traj[start_idx:]

    if est.size == 0 or gt.size == 0:
        return float("nan")

    errors = np.linalg.norm(est - gt, axis=1)
    return float(np.sqrt(np.mean(errors ** 2)))


def detect_turn_frames(
    gt_traj: np.ndarray,
    heading_threshold_deg: float = 1.5,
) -> np.ndarray:
    """
    Позначає кадри, де траєкторія має поворот (за зміною heading у площині XZ).
    """
    n = len(gt_traj)
    turn_mask = np.zeros(n, dtype=bool)
    if n < 3:
        return turn_mask

    dx = np.diff(gt_traj[:, 0])
    dz = np.diff(gt_traj[:, 2])
    heading = np.unwrap(np.arctan2(dz, dx))
    d_heading = np.diff(heading)
    threshold = np.deg2rad(heading_threshold_deg)

    # d_heading[i] відповідає кадру i+2 у вихідній послідовності
    turn_mask[2:] = np.abs(d_heading) > threshold
    return turn_mask


def _safe_rate(mask: np.ndarray, success: np.ndarray) -> float:
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(success[mask]))

def load_camera_matrix(calib_path: Union[str, Path], camera_id: int = 0) -> np.ndarray:
    """
    Зчитує внутрішню матрицю камери K з файлу calib.txt (формат KITTI).
    """
    calib_path = Path(calib_path)
    if not calib_path.exists():
        raise FileNotFoundError(f"Файл калібрування не знайдено: {calib_path}")

    key = f"P{camera_id}:"
    with open(calib_path) as f:
        for line in f:
            if line.startswith(key):
                vals = list(map(float, line.strip().split()[1:]))
                P = np.array(vals).reshape(3, 4)
                K = P[:3, :3]
                return K

    raise ValueError(
        f"Рядок '{key}' не знайдено у файлі {calib_path}. "
        f"Доступні ключі: P0 … P3 (camera_id=0..3)"
    )

def compute_scale(pose_prev: np.ndarray, pose_curr: np.ndarray) -> float:
    """
    Евклідова відстань між двома позами Ground Truth.
    """
    t_prev = pose_prev[:3, 3]
    t_curr = pose_curr[:3, 3]
    return float(np.linalg.norm(t_curr - t_prev))

class SimpleNadirEstimator:
    """
    Прямий оцінювач для камери, що дивиться вниз.
    Не використовує гомографію, а просто масштабує середній оптичний потік
    через висоту польоту та фокусну відстань.
    """
    def __init__(self, K: np.ndarray, use_median: bool = True):
        self.K = K.astype(np.float64)
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.use_median = use_median

    def estimate(self, pts_prev, pts_curr, pose_prev, pose_curr):
        """
        pose_prev, pose_curr — 4×4 матриці (ENU), з яких беремо висоту H.
        """
        if len(pts_prev) < 4:
            return None

        # Обчислюємо всі вектори зсуву (в пікселях)
        flows = pts_curr - pts_prev   # (N, 2)
        if self.use_median:
            med_flow = np.median(flows, axis=0)   # середній потік по всіх точках
        else:
            med_flow = np.mean(flows, axis=0)

        dx_pix, dy_pix = med_flow[0], med_flow[1]

        # Висота над землею (беремо абсолютну висоту ENU, припускаючи пласку землю)
        H = pose_prev[2, 3]   # координата Z в ENU (Up)
        if H < 0.1:
            # Якщо висота замала, пропускаємо
            return None

        # Переводимо піксельний потік у метри в камерній системі
        # (dx: forward, dy: right)
        t_cam = np.array([dx_pix / self.fx * H,
                          dy_pix / self.fy * H,
                          0.0])

        # Перетворюємо з камерної системи в ENU за допомогою матриці повороту дрона
        R_prev = pose_prev[:3, :3]   # ENU матриця
        t_enu = R_prev @ t_cam.reshape(3, 1)

        # Масштаб із GPS (горизонтальна відстань) використовуємо лише для перевірки,
        # але зсув уже в метрах з оптичного потоку.
        # Щоб уникнути шуму, можна застосувати згладжування.
        delta_gt = np.linalg.norm(pose_curr[:3, 3] - pose_prev[:3, 3])
        if delta_gt < 0.001:
            return None   # повністю статичний кадр

        # Зберігаємо напрямок, а масштаб візьмемо із самого оптичного потоку
        est_scale = np.linalg.norm(t_enu[:2])
        if est_scale < 0.001:
            return None

        # Для сумісності з рештою коду
        return PoseEstimate(R=np.eye(3),
                            t=t_cam.reshape(3, 1),
                            t_scaled=t_enu,
                            scale=est_scale,
                            inliers_mask=np.ones(len(pts_prev), dtype=bool),
                            n_inliers=len(pts_prev))
class MotionEstimator:
    """
    Оцінює рух камери між двома кадрами:
      1. Essential Matrix через RANSAC
      2. Pose Recovery (R, t)
      3. Масштабування t за Ground Truth
    """

    MIN_SCALE: float = 0.1

    def __init__(
        self,
        K: np.ndarray,
        ransac_prob: float = 0.999,
        ransac_threshold: float = 1.0,
    ):
        self.K                = K.astype(np.float64)
        self.ransac_prob      = ransac_prob
        self.ransac_threshold = ransac_threshold

        self._focal = float(K[0, 0])
        self._pp    = (float(K[0, 2]), float(K[1, 2]))

    def estimate(
        self,
        pts_prev: np.ndarray,
        pts_curr: np.ndarray,
        pose_prev: np.ndarray,
        pose_curr: np.ndarray,
    ) -> Optional[PoseEstimate]:
        scale = compute_scale(pose_prev, pose_curr)
        if scale < self.MIN_SCALE:
            return None

        E, mask = cv2.findEssentialMat(
            pts_curr, pts_prev,
            focal=self._focal, pp=self._pp,
            method=cv2.RANSAC,
            prob=self.ransac_prob,
            threshold=self.ransac_threshold,
        )
        if E is None:
            return None

        n_inliers, R, t, mask_pose = cv2.recoverPose(
            E, pts_curr, pts_prev,
            focal=self._focal, pp=self._pp, mask=mask,
        )
        if n_inliers < 8:
            return None

        t_scaled = t * scale
        return PoseEstimate(
            R=R, t=t, t_scaled=t_scaled,
            scale=scale,
            inliers_mask=mask_pose.ravel().astype(bool),
            n_inliers=n_inliers,
        )

    @staticmethod
    def update_trajectory(
        state: TrajectoryState,
        estimate: PoseEstimate,
    ) -> TrajectoryState:
        new_t = state.t_pos + state.R_pos @ estimate.t_scaled
        new_R = estimate.R @ state.R_pos
        return TrajectoryState(R_pos=new_R, t_pos=new_t)


class HybridNadirEstimator:
    """
    Гібридний оцінювач для камери вниз.
    - напрямок → GPS (надійний),
    - масштаб → Optical Flow (обмежений).
    """
    def __init__(self, K: np.ndarray, min_inliers: int = 10):
        self.K = K.astype(np.float64)
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.min_inliers = min_inliers

    def estimate(self, pts_prev, pts_curr, pose_prev, pose_curr):
        if len(pts_prev) < self.min_inliers:
            return None

        # 1. Медіанний оптичний потік
        flows = pts_curr - pts_prev
        med_flow = np.median(flows, axis=0)
        dx_pix, dy_pix = med_flow[0], med_flow[1]

        # 2. Висота (ENU Z)
        H = pose_prev[2, 3]
        if H < 0.1:
            return None

        # 3. Масштаб з оптичного потоку (пікселі → метри)
        # Рух вперед → збільшення Y, рух вправо → збільшення X
        forward_m = (dy_pix / self.fy) * H
        right_m   = (dx_pix / self.fx) * H
        scale_vo = np.sqrt(forward_m**2 + right_m**2)

        # 4. GPS напрямок та масштаб
        delta_gt = pose_curr[:3, 3] - pose_prev[:3, 3]
        scale_gt = np.linalg.norm(delta_gt[:2])
        if scale_gt < 0.001:
            return None   # практично статика – ігноруємо

        dir_gt = delta_gt[:2] / scale_gt

        # 5. Фінальний масштаб: обмежуємо VO масштаб у межах [0.2, 2.0] * scale_gt
        use_scale = np.clip(scale_vo, scale_gt * 0.2, scale_gt * 2.0)
        if use_scale < 0.0001:
            use_scale = scale_gt

        # 6. Формуємо зсув у ENU (напрямок GPS, довжина use_scale)
        t_scaled = np.zeros((3, 1))
        t_scaled[0, 0] = dir_gt[0] * use_scale
        t_scaled[1, 0] = dir_gt[1] * use_scale

        return PoseEstimate(
            R=np.eye(3),
            t=np.zeros((3, 1)),
            t_scaled=t_scaled,
            scale=use_scale,
            inliers_mask=np.ones(len(pts_prev), dtype=bool),
            n_inliers=len(pts_prev)
        )

    @staticmethod
    def update_trajectory(state, estimate):
        state.t_pos += estimate.t_scaled
        return state
def run_odometry_pipeline(
    parser,
    feature_matcher,
    estimator,            # MotionEstimator або NadirMotionEstimator
    max_frames: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, PipelineMetrics]:
    n = len(parser) if max_frames is None else min(max_frames, len(parser))
    estimated_traj = np.zeros((n, 3), dtype=np.float64)
    gt_traj        = np.zeros((n, 3), dtype=np.float64)
    frame_times_sec = np.zeros(n, dtype=np.float64)
    fps_per_frame = np.zeros(n, dtype=np.float64)
    pose_success = np.zeros(n, dtype=bool)
    inliers_per_frame = np.zeros(n, dtype=np.int32)

    state = TrajectoryState()
    img0, pose0 = parser[0]
    _ = img0
    gt_traj[0] = pose0[:3, 3]

    for idx in range(1, n):
        t_frame_start = time.perf_counter()
        img_prev, pose_prev = parser[idx - 1]
        img_curr, pose_curr = parser[idx]

        match_result = feature_matcher.match(img_prev, img_curr)
        if match_result is None:
            estimated_traj[idx] = estimated_traj[idx - 1]
            gt_traj[idx]        = pose_curr[:3, 3]
            frame_times_sec[idx] = time.perf_counter() - t_frame_start
            fps_per_frame[idx] = 1.0 / max(frame_times_sec[idx], 1e-9)
            continue

        estimate = estimator.estimate(
            match_result.pts_prev,
            match_result.pts_curr,
            pose_prev,
            pose_curr,
        )
        if estimate is not None:
            state = estimator.update_trajectory(state, estimate)
            pose_success[idx] = True
            inliers_per_frame[idx] = int(estimate.n_inliers)

        estimated_traj[idx] = state.t_pos.ravel()
        gt_traj[idx]        = pose_curr[:3, 3]
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
    return estimated_traj, gt_traj, metrics


def run_detector_experiment(
    parser,
    estimator,
    detector_type: DetectorType,
    max_frames: Optional[int] = None,
    turn_heading_threshold_deg: float = 1.5,
) -> Optional[ExperimentResult]:
    """Запускає повний експеримент для одного детектора."""
    try:
        matcher = FeatureMatcher(detector_type)
    except cv2.error as e:
        logging.warning(f"Детектор {detector_type.name} недоступний: {e}")
        return None

    logging.info(f"\n=== Експеримент: {detector_type.name} ===")
    estimated_traj, gt_traj, metrics = run_odometry_pipeline(
        parser, matcher, estimator, max_frames=max_frames
    )

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
    """Гістограма середнього FPS для детекторів."""
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


def plot_trajectories(results: List[ExperimentResult], axes=(0, 2)):
    """
    Порівняння траєкторій GT та оцінених траєкторій для всіх детекторів.
    axes: (idx_horiz, idx_vert) – які стовпці використати для X та Y.
    """
    fig = plt.figure(figsize=(10, 7))
    gt = results[0].gt_traj
    plt.plot(gt[:, axes[0]], gt[:, axes[1]], label="Ground Truth", color="black", linewidth=2.0)

    colors = {
        "SIFT": "#1d3557",
        "SURF": "#457b9d",
        "ORB": "#e63946",
        "OPTICAL_FLOW": "#2a9d8f",
    }
    for res in results:
        plt.plot(
            res.estimated_traj[:, axes[0]],
            res.estimated_traj[:, axes[1]],
            label=f"{res.detector_name} (ATE={res.ate_rmse:.2f}m)",
            linestyle="--",
            color=colors.get(res.detector_name, None),
        )
    # Автоматичне підписування осей
    axis_labels = [("X", "Z"), ("East", "North"), ("X", "Y")]
    idx = 0 if axes == (0, 2) else 1 if axes == (0, 1) else 2
    plt.xlabel(f"{axis_labels[idx][0]} Position (meters)")
    plt.ylabel(f"{axis_labels[idx][1]} Position (meters)")
    plt.title("Ground Truth vs Estimated Trajectories")
    plt.legend()
    plt.grid(True)
    return fig


if __name__ == "__main__":
    parser_args = argparse.ArgumentParser(description="Motion Estimation with KITTI Dataset")
    parser_args.add_argument("--max_frames", type=int, default=None)
    parser_args.add_argument("--detectors", nargs="+", default=["SIFT", "SURF", "ORB", "OPTICAL_FLOW"])
    parser_args.add_argument("--turn_heading_threshold_deg", type=float, default=1.5)
    parser_args.add_argument("--no_plot", action="store_true")
    parser_args.add_argument("--save_plots_dir", type=str, default=None)
    args = parser_args.parse_args()

    DATASET_PATH = "dataset/"
    SEQUENCE = "00"
    CAMERA = "image_0"

    parser = KITTIOdometryParser(DATASET_PATH, sequence=SEQUENCE, camera=CAMERA)
    logging.info(parser.summary())

    calib_path = DATASET_PATH + f"sequences/{SEQUENCE}/calib.txt"
    K = load_camera_matrix(calib_path, camera_id=0)
    logging.info("Матриця камери K:")
    logging.info(K)

    estimator = MotionEstimator(K)
    max_frames = args.max_frames if args.max_frames is not None else len(parser)

    selected_detectors: List[DetectorType] = []
    for name in args.detectors:
        key = name.upper()
        if key not in DetectorType.__members__:
            logging.warning(f"Невідомий детектор '{name}', пропускаю")
            continue
        selected_detectors.append(DetectorType[key])

    if not selected_detectors:
        raise ValueError("Не обрано жодного валідного детектора")

    results: List[ExperimentResult] = []
    for detector in selected_detectors:
        result = run_detector_experiment(
            parser=parser,
            estimator=estimator,
            detector_type=detector,
            max_frames=max_frames,
            turn_heading_threshold_deg=args.turn_heading_threshold_deg,
        )
        if result is not None:
            results.append(result)

    if not results:
        raise RuntimeError("Не вдалося виконати жоден експеримент")

    logging.info("\n===== ПІДСУМКОВІ МЕТРИКИ =====")
    for r in results:
        logging.info(
            f"{r.detector_name:>4s} | FPS(avg)={r.avg_fps:8.2f} | "
            f"ATE(RMSE)={r.ate_rmse:8.3f} m | "
            f"success={100.0*r.pose_success_rate:6.2f}% | "
            f"turn_success={100.0*r.turn_success_rate if not np.isnan(r.turn_success_rate) else float('nan'):6.2f}% | "
            f"straight_success={100.0*r.straight_success_rate if not np.isnan(r.straight_success_rate) else float('nan'):6.2f}%"
        )

    fps_fig = plot_fps_histogram(results)
    traj_fig = plot_trajectories(results)    # тут для KITTI залишаємо axes=(0,2)

    if args.save_plots_dir:
        os.makedirs(args.save_plots_dir, exist_ok=True)
        fps_path = os.path.join(args.save_plots_dir, "fps_histogram.png")
        traj_path = os.path.join(args.save_plots_dir, "trajectories_comparison.png")
        fps_fig.savefig(fps_path, dpi=150, bbox_inches="tight")
        traj_fig.savefig(traj_path, dpi=150, bbox_inches="tight")
        logging.info(f"Збережено графік FPS: {fps_path}")
        logging.info(f"Збережено графік траєкторій: {traj_path}")

    if not args.no_plot:
        plt.show()