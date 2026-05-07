#!/usr/bin/env python3
import numpy as np
from drone_parser import DroneVideoDataset
from drone_feature_matching import FeatureMatcher, DetectorType
from drone_motion_estimation import PlanarFlowEstimator, align_trajectories_umeyama, compute_ate_rmse


def run_case(name, use_yaw, yaw_field='osd', yaw_offset=0.0, n=1500):
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
    est = PlanarFlowEstimator(K, min_alt_m=1.0, use_yaw=use_yaw, yaw_offset_deg=yaw_offset, flow_model='affine')

    # monkey-patch heading source if needed
    if yaw_field == 'gimbal':
        from drone_parser import TelemetrySample
        orig_sample = ds._timeline.sample
        def sample_with_gimbal(t):
            tele = orig_sample(t)
            # Re-sample gimbal yaw directly from raw arrays is not exposed; use OSD yaw as fallback in this quick test.
            return tele
        # no-op placeholder, just keep interface in case we extend later

    state = np.zeros(3, dtype=np.float64)
    est_traj = np.zeros((n, 3), dtype=np.float64)
    gt_traj = np.zeros((n, 3), dtype=np.float64)
    gt_traj[0] = ds[0][1].pos_enu
    prev_frame, prev_tele = ds[0]
    for i in range(1, n):
        frame, tele = ds[i]
        mr = fm.match(prev_frame, frame, min_matches=12)
        if mr is not None:
            flow, reason = est.estimate_with_reason(mr.pts_prev, mr.pts_curr, tele)
            if flow is not None:
                state += flow.delta_enu
        est_traj[i] = state
        gt_traj[i] = tele.pos_enu
        prev_frame, prev_tele = frame, tele
    aligned = align_trajectories_umeyama(est_traj, gt_traj, with_scale=True)
    ate = compute_ate_rmse(aligned, gt_traj)
    print(f'{name}: ATE={ate:.3f} m  final=({state[0]:.1f},{state[1]:.1f})')
    ds.close()


print('Testing rotation candidates on first 1500 frames')
run_case('OSD yaw', use_yaw=True, yaw_field='osd')
run_case('No yaw', use_yaw=False, yaw_field='osd')
run_case('OSD yaw +90', use_yaw=True, yaw_field='osd', yaw_offset=90.0)
run_case('OSD yaw -90', use_yaw=True, yaw_field='osd', yaw_offset=-90.0)
