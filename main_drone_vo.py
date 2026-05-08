import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from droneVideoParser import DroneVideoCSVParser
from feature_matching import FeatureMatcher, DetectorType
from motion_estimation import (
    HybridNadirEstimator, TrajectoryState,
    compute_ate_rmse, PipelineMetrics,
    ExperimentResult, plot_trajectories
)

VIDEO_PATH = "dataset/drone_footage/23-02-01_FR_F01_V01.mp4"
CSV_PATH   = "dataset/drone_footage/23-02-01_FR_F01.csv"
MAX_FRAMES = 500

parser = DroneVideoCSVParser(VIDEO_PATH, CSV_PATH)
print(parser.summary())
K = parser.K
print("K:\n", K)

estimator = HybridNadirEstimator(K, min_inliers=10)
matcher = FeatureMatcher(DetectorType.OPTICAL_FLOW)

n = min(len(parser), MAX_FRAMES)
est_traj = np.zeros((n, 3))
gt_traj = np.zeros((n, 3))
fps_arr = np.zeros(n)
success = np.zeros(n, dtype=bool)
inliers_arr = np.zeros(n, dtype=np.int32)

state = TrajectoryState()
img0, pose0 = parser[0]
gt_traj[0] = pose0[:3, 3]

cv2.namedWindow("Optical Flow", cv2.WINDOW_NORMAL)

for idx in range(1, n):
    t0 = time.perf_counter()
    img_prev, pose_prev = parser[idx-1]
    img_curr, pose_curr = parser[idx]

    match = matcher.match(img_prev, img_curr)
    if match is None:
        est_traj[idx] = est_traj[idx-1]
        gt_traj[idx] = pose_curr[:3, 3]
        fps_arr[idx] = 1.0 / max(time.perf_counter()-t0, 1e-9)
        continue

    vis = matcher.draw_matches(img_prev, img_curr, match, max_draw=100)
    cv2.imshow("Optical Flow", vis)

    est = estimator.estimate(match.pts_prev, match.pts_curr,
                             pose_prev, pose_curr)
    if est is not None:
        state.t_pos += est.t_scaled
        success[idx] = True
        inliers_arr[idx] = est.n_inliers

    est_traj[idx] = state.t_pos.ravel()
    gt_traj[idx] = pose_curr[:3, 3]
    fps_arr[idx] = 1.0 / max(time.perf_counter()-t0, 1e-9)

    if idx % 100 == 0:
        err = np.linalg.norm(est_traj[idx] - gt_traj[idx])
        print(f"[{idx:04d}/{n}] pos=({state.t_pos[0,0]:7.1f}, {state.t_pos[1,0]:7.1f}) err={err:.2f} m")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

valid = np.arange(len(fps_arr)) > 0
avg_fps = np.mean(fps_arr[valid])
ate = compute_ate_rmse(est_traj, gt_traj, start_idx=1)
success_rate = np.mean(success[valid])

print(f"\n=== РЕЗУЛЬТАТ ===")
print(f"Кадрів: {n}")
print(f"FPS: {avg_fps:.1f}  |  ATE: {ate:.3f} м  |  Успішність: {success_rate*100:.1f}%")

metrics = PipelineMetrics(fps_arr, fps_arr, success, inliers_arr)
res = ExperimentResult(
    detector_name="OPTICAL_FLOW",
    estimated_traj=est_traj,
    gt_traj=gt_traj,
    metrics=metrics,
    avg_fps=avg_fps,
    ate_rmse=ate,
    pose_success_rate=success_rate,
    turn_success_rate=0.0,
    straight_success_rate=0.0
)

fig = plot_trajectories([res], axes=(0,1))
plt.show()