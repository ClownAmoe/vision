"""
Дронове Visual Odometry (VO) БЕЗ залежності від GPS.
Використовує Optical Flow + Altitude для масштабування.
"""

import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple
from droneVideoParser import DroneVideoCSVParser
from feature_matching import FeatureMatcher, DetectorType

VIDEO_PATH = "dataset/drone_footage/23-02-01_FR_F01_V01.mp4"
CSV_PATH   = "dataset/drone_footage/23-02-01_FR_F01.csv"
MAX_FRAMES = 2000  # Більш репрезентативний відрізок для порівняння форми траєкторій
SHOW_VISUALIZATIONS = False
USE_KALMAN_FOR_STEP = False
USE_VIO_FUSION = True
VIO_VO_WEIGHT = 0.20  # частка VO-кроку у fused motion (0..1)

# ===== ПОКРАЩЕНИЙ ОЦІНЮВАЧ РУХУ ДРОНА (без GPS) =====

@dataclass
class TrajectoryState:
    """Накопичена позиція у локальній системі (ENU)."""
    position: np.ndarray = None
    altitude: float = 0.0
    
    def __post_init__(self):
        if self.position is None:
            self.position = np.array([0.0, 0.0, 0.0])

class ImprovedDroneNadirEstimator:
    """
    Оцінювач руху камери для дрона, що дивиться вниз.
    
    Принцип:
    - Optical Flow дає напрямок руху в пікселях
    - Висота (Z altitude) дрона дає масштаб
    - Результат: траєкторія у локальній ENU системі
    
    Для камери, що рухається вниз (nadir):
      - Зсув у Y пікселях → рух вперед (North)
      - Зсув у X пікселях → рух вправо (East)
      - Формула: t_north = (dy_pix / fy) * H
                 t_east  = (dx_pix / fx) * H
    """
    def __init__(self, K: np.ndarray, altitude_column: str = "OSD.height [m]",
                 use_median: bool = True, median_filter_size: int = 3):
        self.K = K.astype(np.float64)
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]
        self.use_median = use_median
        self.median_filter_size = median_filter_size
        
        # Буфер для медіанного фільтра
        self._flow_buffer = []
        
    def estimate(self, pts_prev: np.ndarray, pts_curr: np.ndarray,
                 altitude: float, min_flow_px: float = 0.02,
                 R_rel: Optional[np.ndarray] = None) -> Optional[Tuple[np.ndarray, int]]:
        """
        Обчислює рух камери на основі optical flow + altitude.
        
        Args:
            pts_prev: точки у попередньому кадрі (N, 2)
            pts_curr: точки у поточному кадрі (N, 2)
            altitude: висота над землею (м)
            min_flow_px: мінімальний рух потоку (пікселі) для використання
            R_rel: відносне обертання камера2<-камера1 (3x3) для derotation
            
        Returns:
            (t_translation, n_inliers) або None якщо оцінка неможлива
        """
        if len(pts_prev) < 4 or altitude < 0.1:
            return None
        
        # Обчислення оптичного потоку (із компенсацією повного обертання камери)
        pts_prev_comp = pts_prev
        if R_rel is not None:
            try:
                H_rot = self.K @ R_rel @ np.linalg.inv(self.K)
                pts_h = np.hstack([pts_prev, np.ones((len(pts_prev), 1), dtype=np.float64)])
                warped_h = (H_rot @ pts_h.T).T
                denom = np.clip(warped_h[:, 2:3], 1e-9, None)
                pts_prev_comp = warped_h[:, :2] / denom
            except np.linalg.LinAlgError:
                pts_prev_comp = pts_prev

        flows = pts_curr - pts_prev_comp  # (N, 2)
        magnitudes = np.linalg.norm(flows, axis=1)
        
        # Видалення екстремальних outliers (але не занадто агресивно)
        if len(magnitudes) > 10:
            # Видаляємо лише значення > 95-го перцентилю
            percentile_95 = np.percentile(magnitudes, 95)
            valid_mask = magnitudes <= percentile_95
        else:
            valid_mask = np.ones(len(magnitudes), dtype=bool)
        
        if np.sum(valid_mask) < 4:
            # Якщо видалили занадто багато, використовуємо оригінальні точки
            valid_mask = np.ones(len(magnitudes), dtype=bool)
            
        flows_valid = flows[valid_mask]
        pts_prev_valid = pts_prev[valid_mask]

        if len(flows_valid) == 0:
            return None

        # Після компенсації yaw оцінюємо чистий зсув через affine+RANSAC,
        # щоб уникнути зміщення від локальних outliers/паралаксу.
        pts_corr = pts_prev_valid + flows_valid
        M, inlier_mask_aff = cv2.estimateAffinePartial2D(
            pts_prev_valid,
            pts_corr,
            method=cv2.RANSAC,
            ransacReprojThreshold=1.5,
            maxIters=2000,
            confidence=0.99,
        )

        if M is not None:
            dx_pix = float(M[0, 2])
            dy_pix = float(M[1, 2])
            if inlier_mask_aff is not None:
                n_inliers = int(np.sum(inlier_mask_aff.ravel() > 0))
            else:
                n_inliers = len(flows_valid)
        else:
            med_flow = np.median(flows_valid, axis=0)
            dx_pix, dy_pix = float(med_flow[0]), float(med_flow[1])
            n_inliers = len(flows_valid)

        flow_magnitude = np.sqrt(dx_pix**2 + dy_pix**2)
        
        # Якщо потік дуже малий, ігноруємо
        if flow_magnitude < min_flow_px:
            return None
        
        # Пікселі -> метри у ЛОКАЛЬНИХ осях камери/дрона:
        #   dx -> right, dy -> forward (ще НЕ ENU)
        t_forward = (dy_pix / self.fy) * altitude
        t_right = (dx_pix / self.fx) * altitude

        # Z не оцінюємо оптичним потоком тут
        translation = np.array([t_right, t_forward, 0.0])
        
        return translation, n_inliers

class KalmanFilterSmooth:
    """Простий Kalman filter для згладжування траєкторії."""
    def __init__(self, process_variance=1e-4, measurement_variance=1e-2):
        self.Q = process_variance * np.eye(3)  # невизначеність моделі
        self.R = measurement_variance * np.eye(3)  # невизначеність вимірювання
        self.x = np.zeros(3)  # стан
        self.P = np.eye(3) * 1.0  # коваріанція помилки
        
    def update(self, z: np.ndarray) -> np.ndarray:
        z = z.ravel()  # Конвертуємо у 1D
        
        # Передбачення
        self.P = self.P + self.Q
        
        # Оновлення
        S = self.P + self.R
        try:
            K = self.P @ np.linalg.inv(S)
        except:
            K = self.P @ np.linalg.pinv(S)
            
        self.x = self.x + K @ (z - self.x)
        self.P = (np.eye(3) - K @ self.P)
        
        return self.x.copy()

# ===== ОСНОВНИЙ ЦИКЛ =====

print("=" * 70)
print("ДРОНОВЕ VISUAL ODOMETRY (VO) - БЕЗ GPS")
print("=" * 70)

parser = DroneVideoCSVParser(VIDEO_PATH, CSV_PATH)
print(parser.summary())
K = parser.K
print(f"\nМатриця камери K:\n{K}\n")

# Отримуємо висоту для масштабу VO (пріоритет: OSD.height -> OSD.altitude)
if 'OSD.height [m]' in parser.df.columns:
    altitude_from_csv = parser.df['OSD.height [m]'].astype(float).values
    altitude_col_used = 'OSD.height [m]'
elif 'OSD.height [ft]' in parser.df.columns:
    altitude_from_csv = parser.df['OSD.height [ft]'].astype(float).values * 0.3048
    altitude_col_used = 'OSD.height [ft] -> m'
elif 'OSD.altitude [m]' in parser.df.columns:
    altitude_from_csv = parser.df['OSD.altitude [m]'].astype(float).values
    altitude_col_used = 'OSD.altitude [m]'
elif 'OSD.altitude [ft]' in parser.df.columns:
    altitude_from_csv = parser.df['OSD.altitude [ft]'].astype(float).values * 0.3048
    altitude_col_used = 'OSD.altitude [ft] -> m'
else:
    raise ValueError("Не вдалося знайти OSD.height/OSD.altitude у CSV!")

print(
    f"Висота дрона з CSV ({altitude_col_used}): "
    f"min={np.min(altitude_from_csv):.1f}м, "
    f"max={np.max(altitude_from_csv):.1f}м, "
    f"mean={np.mean(altitude_from_csv):.1f}м\n"
)

estimator = ImprovedDroneNadirEstimator(K, use_median=True)
matcher = FeatureMatcher(DetectorType.OPTICAL_FLOW)

n = min(len(parser), MAX_FRAMES)
print(f"Обробляється {n} кадрів...\n")

# Масиви для накопичення результатів
vo_traj = np.zeros((n, 3), dtype=np.float64)  # VO траєкторія (без GPS)
gt_traj = np.zeros((n, 3), dtype=np.float64)  # Ground-truth траєкторія (ENU from CSV/poses)
altitude_arr = np.zeros(n, dtype=np.float64)  # Висота дрона
altitude_rate_arr = np.zeros(n, dtype=np.float64)  # Швидкість змін висоти
fps_arr = np.zeros(n, dtype=np.float64)
success_arr = np.zeros(n, dtype=bool)
inliers_arr = np.zeros(n, dtype=np.int32)
flow_magnitude_arr = np.zeros(n, dtype=np.float64)
yaw_arr = np.zeros(n, dtype=np.float64)  # DJI yaw (deg), 0=north, clockwise positive
speed_arr = np.zeros(n, dtype=np.float64)  # horizontal speed (m/s) from telemetry

# Ініціалізація
state = TrajectoryState()
kalman = KalmanFilterSmooth(process_variance=1e-3, measurement_variance=1e-1)

img0, pose0 = parser[0]
altitude_arr[0] = altitude_from_csv[0]
state.altitude = altitude_arr[0]
gt_traj[0] = pose0[:3, 3]

# Інтерпольований yaw для кожного кадру
for i in range(n):
    frame_time = (parser.start_frame + i) / parser.video_fps
    yaw_arr[i] = np.interp(frame_time, parser.csv_times, parser.euler_csv[:, 0])

# Інтерпольована горизонтальна швидкість з CSV (MPH -> m/s)
if 'OSD.hSpeed [MPH]' in parser.df.columns:
    hspeed_mps_csv = parser.df['OSD.hSpeed [MPH]'].astype(float).values * 0.44704
    for i in range(n):
        frame_time = (parser.start_frame + i) / parser.video_fps
        speed_arr[i] = np.interp(frame_time, parser.csv_times, hspeed_mps_csv)

if SHOW_VISUALIZATIONS:
    cv2.namedWindow("Drone VO - Optical Flow", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drone VO - Optical Flow", 1200, 600)

print(f"Frame |     Position (E, N)      | Altitude | Flow (px) | Inliers | Success")
print("-" * 80)

# Основний цикл обробки
for idx in range(1, n):
    t0 = time.perf_counter()
    
    img_prev, pose_prev = parser[idx-1]
    img_curr, pose_curr = parser[idx]
    # Ground-truth pose from parser (ENU translation)
    gt_traj[idx] = pose_curr[:3, 3]
    
    # Отримання абсолютної висоти дрона з CSV (не Z з ENU!)
    altitude_curr = altitude_from_csv[idx]
    altitude_arr[idx] = altitude_curr
    altitude_rate_arr[idx] = (altitude_curr - altitude_arr[idx-1]) if idx > 0 else 0.0
    
    # Feature matching
    match = matcher.match(img_prev, img_curr)
    if match is None:
        vo_traj[idx] = vo_traj[idx-1]
        fps_arr[idx] = 1.0 / max(time.perf_counter() - t0, 1e-9)
        
        if idx % 50 == 0:
            print(f"{idx:5d} | ({vo_traj[idx,0]:7.1f}, {vo_traj[idx,1]:7.1f}) | {altitude_curr:8.1f} | "
                  f"    -    |   -     |   N")
        continue
    
    # Motion estimation with full rotation derotation from telemetry orientation
    R_prev = pose_prev[:3, :3]
    R_curr = pose_curr[:3, :3]
    R_rel = R_curr.T @ R_prev

    result = estimator.estimate(
        match.pts_prev,
        match.pts_curr,
        altitude_arr[idx - 1],
        R_rel=R_rel,
    )
    
    if result is not None:
        t_translation_local, n_inliers = result
        success_arr[idx] = True
        inliers_arr[idx] = n_inliers
        
        # Згладження локального кроку [right, forward, 0]
        # Для маневрів (повороти/петлі) агресивне згладження часто "вирівнює" траєкторію,
        # тому за замовчуванням використовуємо сирий крок.
        if USE_KALMAN_FOR_STEP:
            smoothed_local = kalman.update(t_translation_local)
        else:
            smoothed_local = t_translation_local

        # Перехід у ENU через орієнтацію з pose (колонки матриці R: forward/right axes)
        forward_enu = R_curr[:2, 0].astype(np.float64)
        right_enu = R_curr[:2, 1].astype(np.float64)
        f_norm = np.linalg.norm(forward_enu)
        r_norm = np.linalg.norm(right_enu)
        if f_norm > 1e-9:
            forward_enu /= f_norm
        if r_norm > 1e-9:
            right_enu /= r_norm
        delta_en = smoothed_local[1] * forward_enu + smoothed_local[0] * right_enu

        # Обмеження нереалістичних стрибків за телеметричною швидкістю
        max_step_m = max(0.05, (speed_arr[idx] / parser.video_fps) * 2.0 + 0.02)
        step_mag = np.linalg.norm(delta_en)
        if step_mag > max_step_m and step_mag > 1e-9:
            delta_en = delta_en * (max_step_m / step_mag)

        # Visual-Inertial fusion: VO delta + speed/yaw prior delta.
        # Це стабілізує дрейф monocular VO на довгих відрізках.
        if USE_VIO_FUSION and speed_arr[idx] > 1e-6:
            dt = 1.0 / max(parser.video_fps, 1e-9)
            delta_pred = speed_arr[idx] * dt * forward_enu
            w_vo = np.clip(VIO_VO_WEIGHT, 0.0, 1.0)
            # Якщо VO не дуже надійний (мало inliers), сильніше довіряємо speed prior.
            if n_inliers < 1200:
                w_vo *= 0.5
            # Якщо VO-крок явно неузгоджений з телеметричною швидкістю, прибираємо його внесок.
            pred_step_mag = np.linalg.norm(delta_pred)
            vo_step_mag = np.linalg.norm(delta_en)
            if pred_step_mag > 1e-6 and vo_step_mag > 2.5 * pred_step_mag:
                w_vo = 0.0
            delta_fused = w_vo * delta_en + (1.0 - w_vo) * delta_pred
        else:
            delta_fused = delta_en

        state.position[0] += delta_fused[0]  # East
        state.position[1] += delta_fused[1]  # North

        flow_magnitude_arr[idx] = np.linalg.norm(t_translation_local[:2])
    else:
        success_arr[idx] = False
        inliers_arr[idx] = 0
        flow_magnitude_arr[idx] = 0.0
    
    vo_traj[idx] = state.position.copy()
    fps_arr[idx] = 1.0 / max(time.perf_counter() - t0, 1e-9)
    
    # Відображення оптичного потоку
    if SHOW_VISUALIZATIONS and match is not None:
        vis = matcher.draw_matches(img_prev, img_curr, match, max_draw=100)
        status_text = f"Frame: {idx:4d} | Inliers: {inliers_arr[idx]:3d} | Success: {success_arr[idx]}"
        cv2.putText(vis, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Drone VO - Optical Flow", vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Періодичний вивід
    if idx % 50 == 0:
        status_str = "✓" if success_arr[idx] else "✗"
        print(f"{idx:5d} | ({vo_traj[idx,0]:7.1f}, {vo_traj[idx,1]:7.1f}) | {altitude_curr:8.1f} | "
              f"{flow_magnitude_arr[idx]:9.2f} | {inliers_arr[idx]:7d} | {status_str}")

if SHOW_VISUALIZATIONS:
    cv2.destroyAllWindows()

print("-" * 80)

# ===== СТАТИСТИКА =====
valid = np.arange(n) > 0
avg_fps = np.mean(fps_arr[valid]) if np.sum(valid) > 0 else 0
success_rate = np.mean(success_arr[valid]) if np.sum(valid) > 0 else 0

print(f"\n{'='*70}")
print(f"РЕЗУЛЬТАТИ VO АНАЛІЗУ")
print(f"{'='*70}")
print(f"Кадрів оброблено:        {n}")
print(f"Успішних оцінок руху:    {np.sum(success_arr[valid])} ({success_rate*100:.1f}%)")
print(f"Середня FPS:             {avg_fps:.1f}")
print(f"Середнє число inliers:   {np.mean(inliers_arr[valid]):.0f}")
print(f"Макс. середній flow:     {np.max(flow_magnitude_arr):.2f} px")
print(f"\nТраєкторія VO:")
print(f"  - Кінцева позиція (East, North): ({vo_traj[-1,0]:.1f}, {vo_traj[-1,1]:.1f}) м")
total_distance = float(np.sum(np.linalg.norm(np.diff(vo_traj[:, :2], axis=0), axis=1)))
print(f"  - Загальна дистанція: {total_distance:.1f} м")
print(f"\nВисота дрона:")
print(f"  - Початкова: {altitude_arr[0]:.1f} м")
print(f"  - Кінцева: {altitude_arr[-1]:.1f} м")
print(f"  - Мін./макс.: {np.min(altitude_arr):.1f} / {np.max(altitude_arr):.1f} м")
print(f"{'='*70}\n")

# ===== ПОРІВНЯННЯ: Ground Truth vs VO =====
def align_trajectories_umeyama_2d(estimated_traj: np.ndarray, gt_traj: np.ndarray, with_scale: bool = True) -> np.ndarray:
    """
    Umeyama alignment in 2D (East, North) only. This avoids scaling/rotation distortions
    caused by the large vertical (Up) component in GT while VO Z is effectively 0.
    Returns aligned estimated_traj (2D -> 2D) embedded back into 3D (Z copied from original).
    """
    if len(estimated_traj) < 2 or len(gt_traj) < 2:
        return estimated_traj

    src2 = estimated_traj[:, :2].astype(np.float64)  # (N,2)
    dst2 = gt_traj[:, :2].astype(np.float64)

    mu_src = src2.mean(axis=0)
    mu_dst = dst2.mean(axis=0)
    src_centered = src2 - mu_src
    dst_centered = dst2 - mu_dst

    cov = (dst_centered.T @ src_centered) / src2.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1.0

    R = U @ S @ Vt  # 2x2

    if with_scale:
        var_src = np.mean(np.sum(src_centered ** 2, axis=1))
        if var_src < 1e-12:
            scale = 1.0
        else:
            scale = float(np.sum(D * np.diag(S)) / var_src)
    else:
        scale = 1.0

    t = mu_dst - scale * (R @ mu_src)
    aligned2 = (scale * (R @ src2.T)).T + t

    # Build full 3D aligned array: copy Z from original estimated_traj
    aligned = np.zeros_like(estimated_traj)
    aligned[:, :2] = aligned2
    aligned[:, 2] = estimated_traj[:, 2]
    return aligned


# Align VO to GT and compute RAW ATE (RMSE)
aligned_vo = align_trajectories_umeyama_2d(vo_traj, gt_traj, with_scale=False)

valid_idx = np.arange(len(gt_traj)) > 0
if valid_idx.sum() > 1:
    ate_errors_2d = np.linalg.norm(aligned_vo[valid_idx, :2] - gt_traj[valid_idx, :2], axis=1)
    ate_rmse = float(np.sqrt(np.mean(ate_errors_2d ** 2)))
else:
    ate_rmse = 0.0

print(f"ATE raw (aligned): {ate_rmse:.2f} m")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1)
ax.plot(gt_traj[:, 0], gt_traj[:, 1], color='black', linewidth=2.0, label='Ground Truth')
ax.plot(aligned_vo[:, 0], aligned_vo[:, 1], linestyle='--', color='blue', linewidth=2.0, label=f'VO (aligned) ATE={ate_rmse:.2f} m')
ax.scatter(gt_traj[0, 0], gt_traj[0, 1], color='green', s=80, marker='o', label='Start')
ax.scatter(gt_traj[-1, 0], gt_traj[-1, 1], color='red', s=80, marker='s', label='GT End')
ax.set_xlabel('East (м)', fontsize=11, fontweight='bold')
ax.set_ylabel('North (м)', fontsize=11, fontweight='bold')
ax.set_title(f'Ground Truth vs VO (aligned) — ATE RMSE = {ate_rmse:.2f} m', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
ax.axis('equal')

plt.tight_layout()
plt.savefig('drone_vo_analysis.png', dpi=150, bbox_inches='tight')
print(f"✓ Графік збережено: drone_vo_analysis.png (порівняння GT vs VO, ATE={ate_rmse:.2f} m)\\n")

plt.show()