"""
Етап 3: Оцінка руху камери (Motion Estimation)
Essential Matrix → Pose Recovery (R, t) з масштабом із Ground Truth
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from parser import KITTIOdometryParser
from feature_matching import FeatureMatcher, DetectorType
import matplotlib.pyplot as plt
import argparse
import logging

# Configure logging to output to console
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.StreamHandler()])

# ══════════════════════════════════════════════════════════════════════════
# Типи
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class PoseEstimate:
    """Результат оцінки руху між двома кадрами."""
    R:              np.ndarray      # (3, 3) матриця обертання
    t:              np.ndarray      # (3, 1) вектор трансляції (одиничний)
    t_scaled:       np.ndarray      # (3, 1) вектор з реальним масштабом
    scale:          float           # евклідова відстань між позами GT
    inliers_mask:   np.ndarray      # маска RANSAC-інлаєрів
    n_inliers:      int             # кількість інлаєрів


@dataclass
class TrajectoryState:
    """Накопичена глобальна позиція камери."""
    R_pos: np.ndarray = None    # поточна орієнтація (3×3)
    t_pos: np.ndarray = None    # поточна позиція   (3×1)

    def __post_init__(self):
        if self.R_pos is None:
            self.R_pos = np.eye(3, dtype=np.float64)
        if self.t_pos is None:
            self.t_pos = np.zeros((3, 1), dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════
# Парсер матриці калібрування
# ══════════════════════════════════════════════════════════════════════════

def load_camera_matrix(calib_path: str | Path, camera_id: int = 0) -> np.ndarray:
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
                # vals — 12 чисел проєкційної матриці 3×4
                P = np.array(vals).reshape(3, 4)
                # Внутрішня матриця — перші 3 стовпці
                K = P[:3, :3]
                return K

    raise ValueError(
        f"Рядок '{key}' не знайдено у файлі {calib_path}. "
        f"Доступні ключі: P0 … P3 (camera_id=0..3)"
    )


# ══════════════════════════════════════════════════════════════════════════
# Обчислення масштабу з Ground Truth
# ══════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════
# Основний клас
# ══════════════════════════════════════════════════════════════════════════

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

    # Мінімальна відстань між кадрами, щоб оновлювати позу
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

        # Параметри для cv2.findEssentialMat
        self._focal = float(K[0, 0])
        self._pp    = (float(K[0, 2]), float(K[1, 2]))   # principal point

    # ── оцінка між парою кадрів ───────────────────────────────────────────

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

        # Пропускаємо статичні кадри
        if scale < self.MIN_SCALE:
            return None

        # ── Essential Matrix ─────────────────────────────────────────────
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

        # ── Pose Recovery ────────────────────────────────────────────────
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

    # ── оновлення глобальної траєкторії ───────────────────────────────────

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


# ══════════════════════════════════════════════════════════════════════════
# Збірний пайплайн: feature matching + motion estimation
# ══════════════════════════════════════════════════════════════════════════

def run_odometry_pipeline(
    parser,                     # KITTIOdometryParser
    feature_matcher,            # FeatureMatcher
    estimator: MotionEstimator,
    max_frames: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
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

        # Зіставлення ключових точок
        match_result = feature_matcher.match(img_prev, img_curr)
        if match_result is None:
            # Якщо збігів немає — залишаємо поточну позицію
            estimated_traj[idx] = estimated_traj[idx - 1]
            gt_traj[idx]        = pose_curr[:3, 3]
            continue

        # Оцінка руху
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


# ══════════════════════════════════════════════════════════════════════════
# Демонстрація (без реального датасету)
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Додано аргументи командного рядка
    parser_args = argparse.ArgumentParser(description="Motion Estimation with KITTI Dataset")
    parser_args.add_argument(
        "--max_frames", type=int, default=None,
        help="Максимальна кількість кадрів для обробки (None для всіх кадрів)"
    )
    args = parser_args.parse_args()

    # Налаштування датасету
    DATASET_PATH = "dataset/"  # Шлях до датасету
    SEQUENCE = "00"           # Номер послідовності
    CAMERA = "image_0"        # Камера (image_0 для grayscale)

    # Ініціалізація парсера
    parser = KITTIOdometryParser(DATASET_PATH, sequence=SEQUENCE, camera=CAMERA)
    logging.info(parser.summary())

    # Завантаження матриці камери
    calib_path = DATASET_PATH + f"sequences/{SEQUENCE}/calib.txt"
    K = load_camera_matrix(calib_path, camera_id=0)
    logging.info("Матриця камери K:")
    logging.info(K)

    # Ініціалізація оцінювача руху та зіставлення ознак
    estimator = MotionEstimator(K)
    matcher = FeatureMatcher(DetectorType.SIFT)  # Можна змінити на SURF або ORB

    # Ініціалізація траєкторії
    state = TrajectoryState()

    # Обмеження кількості кадрів
    max_frames = args.max_frames if args.max_frames is not None else len(parser)

    for idx in range(1, max_frames):
        img_prev, pose_prev = parser[idx - 1]
        img_curr, pose_curr = parser[idx]

        # Зіставлення ключових точок
        match_result = matcher.match(img_prev, img_curr)
        if match_result is None:
            logging.warning(f"[{idx:04d}] Недостатньо збігів")
            continue

        # Оцінка руху
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

    # Запуск пайплайну з обмеженням кількості кадрів
    estimated_traj, gt_traj = run_odometry_pipeline(parser, matcher, estimator, max_frames=max_frames)

    # Візуалізація траєкторій
    plt.figure(figsize=(10, 6))
    plt.plot(gt_traj[:, 0], gt_traj[:, 2], label="Ground Truth", color="green")
    plt.plot(estimated_traj[:, 0], estimated_traj[:, 2], label="Estimated", color="blue", linestyle="--")
    plt.title("Ground Truth vs Estimated Trajectory")
    plt.xlabel("X Position (meters)")
    plt.ylabel("Z Position (meters)")
    plt.legend()
    plt.grid(True)
    plt.show()
