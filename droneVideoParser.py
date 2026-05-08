import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from pyproj import Transformer
from typing import Tuple
import math

class DroneVideoCSVParser:
    """Парсер відео + CSV‑логів DJI, синхронізований за часом."""

    def __init__(self, video_path: str, csv_path: str,
                 camera_matrix: np.ndarray = None,
                 start_frame: int = 0):
        self.video_path = Path(video_path)
        self.csv_path = Path(csv_path)
        self.start_frame = start_frame

        # Відео
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Не вдалося відкрити відео: {video_path}")
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # CSV
        self.df = self._load_csv()
        self._compute_poses()                 # poses_csv, csv_times

        # Довжина з урахуванням стартового кадру
        self._length = self.total_frames - start_frame
        if self._length <= 0:
            raise ValueError("start_frame перевищує кількість кадрів")

        # Камера
        self.K = camera_matrix if camera_matrix is not None else self._default_k()

        # Послідовне читання відео
        self._next_frame_idx = 0
        self._current_frame = None

    # ------------------------------------------------------------------ #
    #                         Завантаження CSV                             #
    # ------------------------------------------------------------------ #
    def _load_csv(self) -> pd.DataFrame:
        # low_memory=False усуває DtypeWarning
        df = pd.read_csv(self.csv_path, low_memory=False)

        # Парсинг мітки часу
        date_col = 'CUSTOM.date [local]'
        time_col = 'CUSTOM.updateTime [local]'
        df['timestamp'] = pd.to_datetime(
            df[date_col] + ' ' + df[time_col],
            format='%m/%d/%Y %I:%M:%S.%f %p'
        )
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Перетворення футів → метри
        ft_cols = [c for c in df.columns if c.endswith('[ft]')]
        for col in ft_cols:
            new_col = col.replace('[ft]', '[m]')
            df[new_col] = df[col].astype(float) * 0.3048

        return df

    # ------------------------------------------------------------------ #
    #                    Позиції й орієнтація (ENU)                       #
    # ------------------------------------------------------------------ #
    def _compute_poses(self):
        """Для кожного рядка CSV обчислює матрицю 4x4 у локальній ENU."""
        lla = "EPSG:4326"
        ecef = "EPSG:4978"
        self._transformer = Transformer.from_crs(lla, ecef, always_xy=True)

        lat0 = self.df['OSD.latitude'].iloc[0]
        lon0 = self.df['OSD.longitude'].iloc[0]
        # Використовуємо абсолютну висоту OSD.height [m] (якщо є) або altitude
        alt_col = 'OSD.height [m]' if 'OSD.height [m]' in self.df.columns else 'OSD.altitude [m]'
        alt0 = self.df[alt_col].iloc[0]

        self._origin_ecef = np.array(self._transformer.transform(lon0, lat0, alt0))
        self._R_ecef2enu = self._ecef2enu_matrix(lat0, lon0)

        n = len(self.df)
        positions = np.zeros((n, 3), dtype=np.float64)
        euler = np.zeros((n, 3), dtype=np.float64)
        times = np.zeros(n)

        t0 = self.df['timestamp'].iloc[0]
        for i, (_, row) in enumerate(self.df.iterrows()):
            times[i] = (row['timestamp'] - t0).total_seconds()
            lat = float(row['OSD.latitude'])
            lon = float(row['OSD.longitude'])
            alt = float(row[alt_col]) if alt_col in row else 0.0

            ecef_pt = np.array(self._transformer.transform(lon, lat, alt))
            enu = self._R_ecef2enu @ (ecef_pt - self._origin_ecef)
            positions[i] = enu

            euler[i] = [float(row.get('OSD.yaw', 0.0)),
                        float(row.get('OSD.pitch', 0.0)),
                        float(row.get('OSD.roll', 0.0))]

        self.csv_times = times
        self.pos_csv = positions
        self.euler_csv = self._unwrap_yaw(euler)

    @staticmethod
    def _ecef2enu_matrix(lat_deg, lon_deg):
        lat = math.radians(lat_deg)
        lon = math.radians(lon_deg)
        slat, clat = math.sin(lat), math.cos(lat)
        slon, clon = math.sin(lon), math.cos(lon)
        return np.array([
            [-slon,         clon,         0],
            [-slat*clon,   -slat*slon,   clat],
            [ clat*clon,    clat*slon,   slat]
        ])

    @staticmethod
    def _unwrap_yaw(euler_deg):
        yaw = np.unwrap(np.deg2rad(euler_deg[:, 0]))
        out = euler_deg.copy()
        out[:, 0] = np.rad2deg(yaw)
        return out

    @staticmethod
    def euler_to_rot_matrix(yaw_deg, pitch_deg, roll_deg) -> np.ndarray:
        """
        Правильна матриця повороту з DJI-кутів у ENU.
        yaw=0 → північ, yaw зростає за годинниковою.
        """
        yaw_rad = np.pi/2 - np.deg2rad(yaw_deg)   # перетворення в стандартний кут
        pitch = np.deg2rad(pitch_deg)
        roll = np.deg2rad(roll_deg)
        cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        return np.array([
            [ cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
            [ sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
            [ -sp,    cp*sr,             cp*cr]
        ])
    
    def _interpolate_pose(self, frame_time: float) -> np.ndarray:
        if frame_time <= self.csv_times[0]:
            t = self.pos_csv[0], self.euler_csv[0]
        elif frame_time >= self.csv_times[-1]:
            t = self.pos_csv[-1], self.euler_csv[-1]
        else:
            pos = np.array([np.interp(frame_time, self.csv_times, self.pos_csv[:, i])
                            for i in range(3)])
            ang = np.array([np.interp(frame_time, self.csv_times, self.euler_csv[:, i])
                            for i in range(3)])
            t = pos, ang

        R = self.euler_to_rot_matrix(*t[1])
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = R
        pose[:3, 3] = t[0]
        return pose

    # ------------------------------------------------------------------ #
    #                      Читання кадрів (послідовне)                     #
    # ------------------------------------------------------------------ #
    def _read_frame_at(self, target_idx: int) -> np.ndarray:
        if target_idx < 0 or target_idx >= self.total_frames:
            raise IndexError(f"Кадр {target_idx} поза межами [0, {self.total_frames})")
        if target_idx == self._next_frame_idx:
            ret, frame = self.cap.read()
            if not ret:
                raise IOError(f"Не вдалося прочитати кадр {target_idx}")
            self._current_frame = frame
            self._next_frame_idx += 1
            return frame
        elif target_idx > self._next_frame_idx:
            skip = target_idx - self._next_frame_idx
            for _ in range(skip):
                ret = self.cap.grab()
                if not ret:
                    raise IOError(f"Помилка при перемотуванні до кадру {target_idx}")
                self._next_frame_idx += 1
            ret, frame = self.cap.retrieve()
            if not ret:
                raise IOError(f"Не вдалося отримати кадр {target_idx}")
            self._current_frame = frame
            self._next_frame_idx += 1
            return frame
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            self._next_frame_idx = target_idx
            ret, frame = self.cap.read()
            if not ret:
                raise IOError(f"Не вдалося прочитати кадр {target_idx}")
            self._current_frame = frame
            self._next_frame_idx += 1
            return frame

    # ------------------------------------------------------------------ #
    #                          Інтерфейс                                  #
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Індекс {idx} поза межами [0, {self._length})")
        frame_idx = self.start_frame + idx
        frame = self._read_frame_at(frame_idx)
        frame_time = frame_idx / self.video_fps
        pose = self._interpolate_pose(frame_time)
        return frame, pose

    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, mat):
        self._K = np.array(mat, dtype=np.float64)

    def _default_k(self) -> np.ndarray:
        # Для 960x540: fx = fy = 1400
        fx = fy = 1400.0
        cx, cy = self.width / 2.0, self.height / 2.0
        return np.array([[fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]], dtype=np.float64)

    def summary(self) -> str:
        return (f"DroneVideoCSVParser\n"
                f"  Video     : {self.video_path.name} ({self.width}x{self.height} @ {self.video_fps:.1f} fps)\n"
                f"  CSV rows  : {len(self.df)}\n"
                f"  Frames    : {self._length} (start={self.start_frame})\n"
                f"  Start pos : {self.pos_csv[0]} ENU")