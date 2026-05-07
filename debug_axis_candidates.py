#!/usr/bin/env python3
import numpy as np
from drone_parser import DroneVideoDataset
from drone_feature_matching import FeatureMatcher, DetectorType
from drone_motion_estimation import PlanarFlowEstimator, align_trajectories_umeyama, compute_ate_rmse


ds = DroneVideoDataset(
    'dataset',
    video_path='drone_footage/23-02-01_FR_F01_combined.MP4',
    csv_path='drone_footage/23-02-01_FR_F01.csv',
    frame_stride=1,
    normalize_time=True,
)
fm = FeatureMatcher(DetectorType.OPTICAL_FLOW)
frame0, _ = ds[0]
h, w = frame0.shape[:2]
K = np.array([[max(w, h), 0, w/2], [0, max(w, h), h/2], [0, 0, 1]], dtype=np.float64)
est = PlanarFlowEstimator(K, min_alt_m=1.0, use_yaw=False, flow_model='affine')

cases = {
    'current': lambda tx, ty, alt: np.array([-tx * alt / est.fx, -ty * alt / est.fy, 0.0]),
    'swap': lambda tx, ty, alt: np.array([-ty * alt / est.fy, -tx * alt / est.fx, 0.0]),
    'flipx': lambda tx, ty, alt: np.array([ tx * alt / est.fx, -ty * alt / est.fy, 0.0]),
    'flipy': lambda tx, ty, alt: np.array([-tx * alt / est.fx,  ty * alt / est.fy, 0.0]),
    'direct': lambda tx, ty, alt: np.array([ tx * alt / est.fx,  ty * alt / est.fy, 0.0]),
}

n = 1500
rows = {name: np.zeros((n, 3), dtype=np.float64) for name in cases}
gt = np.zeros((n, 3), dtype=np.float64)
prev_frame, prev_tele = ds[0]
for i in range(1, n):
    frame, tele = ds[i]
    mr = fm.match(prev_frame, frame, min_matches=12)
    if mr is not None:
        tx, ty, _ = est._estimate_pixel_shift(mr.pts_prev, mr.pts_curr)
        if tx is not None:
            for name, fn in cases.items():
                rows[name][i] = rows[name][i - 1] + fn(tx, ty, tele.alt_m)
        else:
            for name in cases:
                rows[name][i] = rows[name][i - 1]
    else:
        for name in cases:
            rows[name][i] = rows[name][i - 1]
    gt[i] = tele.pos_enu
    prev_frame, prev_tele = frame, tele

for name, traj in rows.items():
    aligned = align_trajectories_umeyama(traj, gt, with_scale=True)
    ate = compute_ate_rmse(aligned, gt)
    print(f'{name}: ATE={ate:.3f} m final=({traj[-1,0]:.1f},{traj[-1,1]:.1f})')

ds.close()
