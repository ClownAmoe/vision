#!/usr/bin/env python3
import csv
from pathlib import Path
import numpy as np

csv_path = Path('dataset/drone_footage/23-02-01_FR_F01.csv')

osd_yaw = []
gimbal_yaw = []
gimbal_pitch = []
gimbal_roll = []

with open(csv_path, encoding='utf-8', newline='') as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i >= 500:
            break

        def fv(key):
            value = row.get(key)
            if value in (None, ''):
                return None
            try:
                return float(value)
            except ValueError:
                return None

        osd = fv('OSD.yaw [360]')
        if osd is None:
            osd = fv('OSD.yaw')
        gy = fv('GIMBAL.yaw [360]')
        if gy is None:
            gy = fv('GIMBAL.yaw')

        if osd is not None:
            osd_yaw.append(osd)
        if gy is not None:
            gimbal_yaw.append(gy)
        gp = fv('GIMBAL.pitch')
        gr = fv('GIMBAL.roll')
        if gp is not None:
            gimbal_pitch.append(gp)
        if gr is not None:
            gimbal_roll.append(gr)

for name, arr in [
    ('OSD yaw', np.array(osd_yaw, dtype=np.float64)),
    ('GIMBAL yaw', np.array(gimbal_yaw, dtype=np.float64)),
    ('GIMBAL pitch', np.array(gimbal_pitch, dtype=np.float64)),
    ('GIMBAL roll', np.array(gimbal_roll, dtype=np.float64)),
]:
    print(f'{name}: count={len(arr)} min={np.min(arr):.2f} max={np.max(arr):.2f} std={np.std(arr):.2f}')
    print(f'  first 12: {arr[:12]}')
