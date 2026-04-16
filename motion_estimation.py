"""
Етап 3: Оцінка руху камери (Motion Estimation)
Essential Matrix → Pose Recovery (R, t) з масштабом із Ground Truth
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

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

def load_camera_matrix(calib_path: Union[str, Path], camera_id: int = 0) -> np.ndarray:
    """
    Зчитує внутрішню матрицю камери K з файлу calib.txt (формат KITTI).

    KITTI calib.txt містить рядки вигляду:
        P0: fx 0 cx 0  0 fy cy 0  0 0 1 0
        P1: ...

    Повертає матрицю K (3×3).
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

    Parameters
    ----------
    pose_prev, pose_curr : np.ndarray (4×4) — матриці трансформації з GT

    Returns
    -------
    float — реальна метрова відстань між кадрами
    """
    t_prev = pose_prev[:3, 3]
    t_curr = pose_curr[:3, 3]
    return float(np.linalg.norm(t_curr - t_prev))

class MotionEstimator:
    """
    Оцінює рух камери між двома кадрами:
      1. Essential Matrix через RANSAC
      2. Pose Recovery (R, t)
      3. Масштабування t за Ground Truth

    Приклад використання:
        K = load_camera_matrix("calib.txt")
        estimator = MotionEstimator(K)
        state = TrajectoryState()

        for idx in range(1, len(parser)):
            img_prev, pose_prev = parser[idx - 1]
            img_curr, pose_curr = parser[idx]

            match = feature_matcher.match(img_prev, img_curr)
            if match is None:
                continue

            estimate = estimator.estimate(
                match.pts_prev, match.pts_curr,
                pose_prev, pose_curr
            )
            if estimate is None:
                continue

            state = estimator.update_trajectory(state, estimate)
            x, z = state.t_pos[0, 0], state.t_pos[2, 0]
    """

    MIN_SCALE: float = 0.1

    def __init__(
        self,
        K: np.ndarray,
        ransac_prob: float = 0.999,
        ransac_threshold: float = 1.0,
    ):
        """
        Parameters
        ----------
        K                  : внутрішня матриця камери (3×3)
        ransac_prob        : ймовірність успіху RANSAC
        ransac_threshold   : порогова відстань (пікселі) для RANSAC-інлаєрів
        """
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
        """
        Оцінює відносний рух між кадрами.

        Parameters
        ----------
        pts_prev, pts_curr : (N, 2) координати збігів
        pose_prev, pose_curr : (4×4) пози Ground Truth для масштабу

        Returns
        -------
        PoseEstimate або None, якщо недостатньо інлаєрів / занадто малий рух
        """
        scale = compute_scale(pose_prev, pose_curr)

        if scale < self.MIN_SCALE:
            return None

        E, mask = cv2.findEssentialMat(
            pts_curr,
            pts_prev,
            focal     = self._focal,
            pp        = self._pp,
            method    = cv2.RANSAC,
            prob      = self.ransac_prob,
            threshold = self.ransac_threshold,
        )
        if E is None:
            return None

        n_inliers, R, t, mask_pose = cv2.recoverPose(
            E,
            pts_curr,
            pts_prev,
            focal = self._focal,
            pp    = self._pp,
            mask  = mask,
        )

        if n_inliers < 8:
            return None

        t_scaled = t * scale

        return PoseEstimate(
            R            = R,
            t            = t,
            t_scaled     = t_scaled,
            scale        = scale,
            inliers_mask = mask_pose.ravel().astype(bool),
            n_inliers    = n_inliers,
        )

    @staticmethod
    def update_trajectory(
        state: TrajectoryState,
        estimate: PoseEstimate,
    ) -> TrajectoryState:
        """
        Застосовує оцінений рух до накопиченої позиції.

        Формула:
            t_pos = t_pos + R_pos @ t_scaled
            R_pos = R @ R_pos

        Parameters
        ----------
        state    : поточний стан (R_pos, t_pos)
        estimate : результат estimate()

        Returns
        -------
        Оновлений TrajectoryState
        """
        new_t = state.t_pos + state.R_pos @ estimate.t_scaled
        new_R = estimate.R @ state.R_pos

        return TrajectoryState(R_pos=new_R, t_pos=new_t)

def run_odometry_pipeline(
    parser,
    feature_matcher,
    estimator: MotionEstimator,
    max_frames: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Повний прохід по датасету: видобуток ознак → оцінка руху → траєкторія.

    Returns
    -------
    estimated_traj : (N, 3) — обчислена траєкторія (x, y, z)
    gt_traj        : (N, 3) — Ground Truth траєкторія
    """
    n = len(parser) if max_frames is None else min(max_frames, len(parser))

    estimated_traj = np.zeros((n, 3))
    gt_traj        = np.zeros((n, 3))

    state = TrajectoryState()

    for idx in range(1, n):
        img_prev, pose_prev = parser[idx - 1]
        img_curr, pose_curr = parser[idx]

        match_result = feature_matcher.match(img_prev, img_curr)
        if match_result is None:
            estimated_traj[idx] = estimated_traj[idx - 1]
            gt_traj[idx]        = pose_curr[:3, 3]
            continue

        estimate = estimator.estimate(
            match_result.pts_prev,
            match_result.pts_curr,
            pose_prev,
            pose_curr,
        )

        if estimate is not None:
            state = MotionEstimator.update_trajectory(state, estimate)

        estimated_traj[idx] = state.t_pos.ravel()
        gt_traj[idx]        = pose_curr[:3, 3]

        if idx % 100 == 0:
            err = np.linalg.norm(estimated_traj[idx] - gt_traj[idx])
            print(
                f"[{idx:04d}/{n}]  "
                f"pos=({state.t_pos[0,0]:7.1f}, {state.t_pos[2,0]:7.1f})  "
                f"err={err:.2f} m"
            )

    return estimated_traj, gt_traj

if __name__ == "__main__":
    parser_args = argparse.ArgumentParser(description="Motion Estimation with KITTI Dataset")
    parser_args.add_argument(
        "--max_frames", type=int, default=None,
        help="Максимальна кількість кадрів для обробки (None для всіх кадрів)"
    )
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
    matcher = FeatureMatcher(DetectorType.SIFT)

    state = TrajectoryState()

    max_frames = args.max_frames if args.max_frames is not None else len(parser)

    for idx in range(1, max_frames):
        img_prev, pose_prev = parser[idx - 1]
        img_curr, pose_curr = parser[idx]

        match_result = matcher.match(img_prev, img_curr)
        if match_result is None:
            logging.warning(f"[{idx:04d}] Недостатньо збігів")
            continue

        estimate = estimator.estimate(
            match_result.pts_prev,
            match_result.pts_curr,
            pose_prev,
            pose_curr,
        )

        if estimate is not None:
            state = MotionEstimator.update_trajectory(state, estimate)

        x, z = state.t_pos[0, 0], state.t_pos[2, 0]
        logging.info(f"[{idx:04d}] Позиція: x={x:.2f}, z={z:.2f}")

    estimated_traj, gt_traj = run_odometry_pipeline(parser, matcher, estimator, max_frames=max_frames)

    plt.figure(figsize=(10, 6))
    plt.plot(gt_traj[:, 0], gt_traj[:, 2], label="Ground Truth", color="green")
    plt.plot(estimated_traj[:, 0], estimated_traj[:, 2], label="Estimated", color="blue", linestyle="--")
    plt.title("Ground Truth vs Estimated Trajectory")
    plt.xlabel("X Position (meters)")
    plt.ylabel("Z Position (meters)")
    plt.legend()
    plt.grid(True)
    plt.show()
