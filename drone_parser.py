import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class TelemetrySample:
    time_s: float
    lat: float
    lon: float
    alt_m: float
    pos_enu: np.ndarray
    yaw_deg: float
    gimbal_yaw_deg: float
    pitch_deg: float
    roll_deg: float


class TelemetryTimeline:
    """Interpolates telemetry to arbitrary times."""

    FT_TO_M = 0.3048
    EARTH_RADIUS_M = 6378137.0

    def __init__(
        self,
        csv_path: Path,
        time_column: str = "OSD.flyTime [s]",
        normalize_time: bool = True,
    ):
        self.time_column = time_column
        self.normalize_time = normalize_time

        rows = self._load_csv_rows(csv_path)
        self._init_series(rows)

    @staticmethod
    def _convert_value(value: str):
        if value is None:
            return None
        v = value.strip()
        if v == "":
            return None
        try:
            if "." in v:
                return float(v)
            return int(v)
        except ValueError:
            return v

    def _load_csv_rows(self, path: Path) -> List[dict]:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                converted = {k: self._convert_value(v) for k, v in row.items()}
                rows.append(converted)
        return rows

    def _init_series(self, rows: List[dict]) -> None:
        times = []
        lat = []
        lon = []
        alt_m = []
        yaw = []
        gimbal_yaw = []
        pitch = []
        roll = []

        for row in rows:
            t = row.get(self.time_column)
            if t is None:
                continue
            try:
                t_val = float(t)
            except (TypeError, ValueError):
                continue

            lat_val = row.get("OSD.latitude")
            lon_val = row.get("OSD.longitude")
            # Prefer the absolute altitude field first; fall back to VPS height then height
            alt_ft = self._first_non_null(
                row,
                ["OSD.altitude [ft]", "OSD.vpsHeight [ft]", "OSD.height [ft]"],
            )
            if lat_val is None or lon_val is None or alt_ft is None:
                continue

            times.append(t_val)
            lat.append(float(lat_val))
            lon.append(float(lon_val))
            alt_m.append(float(alt_ft) * self.FT_TO_M)

            yaw_val = row.get("OSD.yaw") if row.get("OSD.yaw") is not None else row.get("OSD.yaw [360]")
            gimbal_yaw_val = row.get("GIMBAL.yaw") if row.get("GIMBAL.yaw") is not None else row.get("GIMBAL.yaw [360]")
            pitch_val = row.get("OSD.pitch")
            roll_val = row.get("OSD.roll")

            yaw.append(float(yaw_val) if yaw_val is not None else float("nan"))
            gimbal_yaw.append(float(gimbal_yaw_val) if gimbal_yaw_val is not None else float("nan"))
            pitch.append(float(pitch_val) if pitch_val is not None else float("nan"))
            roll.append(float(roll_val) if roll_val is not None else float("nan"))

        if len(times) < 2:
            raise ValueError("Not enough telemetry rows with valid time/position data.")

        times_arr = np.array(times, dtype=np.float64)
        if self.normalize_time:
            times_arr = times_arr - times_arr[0]

        self._times = times_arr
        self._lat = np.array(lat, dtype=np.float64)
        self._lon = np.array(lon, dtype=np.float64)
        self._alt_m = np.array(alt_m, dtype=np.float64)
        self._yaw = np.array(yaw, dtype=np.float64)
        self._gimbal_yaw = np.array(gimbal_yaw, dtype=np.float64)
        self._pitch = np.array(pitch, dtype=np.float64)
        self._roll = np.array(roll, dtype=np.float64)

        self._pos_enu = self._latlon_to_enu(self._lat, self._lon, self._alt_m)

    @property
    def start_time(self) -> float:
        return float(self._times[0])

    @property
    def end_time(self) -> float:
        return float(self._times[-1])

    @property
    def duration(self) -> float:
        return float(self._times[-1] - self._times[0])

    @staticmethod
    def _first_non_null(row: dict, keys: List[str]) -> Optional[float]:
        for key in keys:
            val = row.get(key)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
        return None

    def _latlon_to_enu(
        self,
        lat_deg: np.ndarray,
        lon_deg: np.ndarray,
        alt_m: np.ndarray,
    ) -> np.ndarray:
        lat0 = np.deg2rad(lat_deg[0])
        lon0 = np.deg2rad(lon_deg[0])

        lat = np.deg2rad(lat_deg)
        lon = np.deg2rad(lon_deg)

        d_lat = lat - lat0
        d_lon = lon - lon0

        east = d_lon * np.cos(lat0) * self.EARTH_RADIUS_M
        north = d_lat * self.EARTH_RADIUS_M
        up = alt_m - alt_m[0]

        return np.stack([east, north, up], axis=1)

    def _interp(self, t: float, values: np.ndarray) -> float:
        t_clamped = float(np.clip(t, self._times[0], self._times[-1]))
        return float(np.interp(t_clamped, self._times, values))

    def _interp_angle(self, t: float, values_deg: np.ndarray) -> float:
        mask = np.isfinite(values_deg)
        if mask.sum() < 2:
            return 0.0
        times = self._times[mask]
        vals_rad = np.deg2rad(values_deg[mask])
        vals_unwrap = np.unwrap(vals_rad)
        t_clamped = float(np.clip(t, times[0], times[-1]))
        interp_rad = np.interp(t_clamped, times, vals_unwrap)
        return float(np.rad2deg(interp_rad))

    def sample(self, t: float) -> TelemetrySample:
        lat = self._interp(t, self._lat)
        lon = self._interp(t, self._lon)
        alt = self._interp(t, self._alt_m)

        pos_e = self._interp(t, self._pos_enu[:, 0])
        pos_n = self._interp(t, self._pos_enu[:, 1])
        pos_u = self._interp(t, self._pos_enu[:, 2])

        yaw = self._interp_angle(t, self._yaw)
        gimbal_yaw = self._interp_angle(t, self._gimbal_yaw)
        pitch = self._interp_angle(t, self._pitch)
        roll = self._interp_angle(t, self._roll)

        return TelemetrySample(
            time_s=float(t),
            lat=lat,
            lon=lon,
            alt_m=alt,
            pos_enu=np.array([pos_e, pos_n, pos_u], dtype=np.float64),
            yaw_deg=yaw,
            gimbal_yaw_deg=gimbal_yaw,
            pitch_deg=pitch,
            roll_deg=roll,
        )


class DroneVideoDataset:
    """Provides frames and interpolated telemetry samples."""

    def __init__(
        self,
        dataset_path: str,
        video_path: str,
        csv_path: str,
        start_frame: int = 0,
        max_frames: Optional[int] = None,
        time_column: str = "OSD.flyTime [s]",
        time_offset: float = 0.0,
        normalize_time: bool = True,
        target_fps: float = 5.0,
        frame_stride: Optional[int] = None,
    ):
        self.root = Path(dataset_path)
        self.video_path = (
            self.root / video_path if not Path(video_path).is_absolute() else Path(video_path)
        )
        self.csv_path = (
            self.root / csv_path if not Path(csv_path).is_absolute() else Path(csv_path)
        )
        self.start_frame = int(start_frame)
        self.time_offset = float(time_offset)

        self._validate_paths()

        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise IOError(f"Failed to open video: {self.video_path}")

        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        if self._fps <= 0:
            raise ValueError("Invalid FPS; cannot compute frame timestamps.")

        if frame_stride is None:
            if target_fps <= 0:
                stride = 1
            else:
                stride = int(round(self._fps / target_fps))
                stride = max(1, stride)
        else:
            stride = max(1, int(frame_stride))
        self.frame_stride = stride

        self._timeline = TelemetryTimeline(
            self.csv_path,
            time_column=time_column,
            normalize_time=normalize_time,
        )

        if start_frame < 0 or start_frame >= self._frame_count:
            raise ValueError(f"start_frame {start_frame} out of range [0, {self._frame_count})")

        indices = list(range(start_frame, self._frame_count, self.frame_stride))
        if max_frames is not None:
            indices = indices[: int(max_frames)]
        self._indices = indices
        self._video_duration = self._frame_count / self._fps

    def _validate_paths(self) -> None:
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

    def __len__(self) -> int:
        return len(self._indices)

    def _frame_time(self, frame_idx: int) -> float:
        return frame_idx / self._fps + self.time_offset

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, TelemetrySample]:
        if idx < 0 or idx >= len(self._indices):
            raise IndexError(f"Index {idx} out of range [0, {len(self._indices)})")

        frame_idx = self._indices[idx]
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise IOError(f"Failed to read frame {frame_idx} from {self.video_path}")

        t = self._frame_time(frame_idx)
        tele = self._timeline.sample(t)
        return frame, tele

    def __iter__(self) -> Iterator[Tuple[np.ndarray, TelemetrySample]]:
        for i in range(len(self)):
            yield self[i]

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def fps(self) -> float:
        return self._fps

    def summary(self) -> str:
        return (
            f"DroneVideoDataset\n"
            f"  Video       : {self.video_path}\n"
            f"  CSV         : {self.csv_path}\n"
            f"  Frames      : {len(self._indices)}\n"
            f"  FPS         : {self._fps:.2f}\n"
            f"  Frame stride: {self.frame_stride}\n"
            f"  Video dur   : {self._video_duration:.2f} s\n"
            f"  Tele dur    : {self._timeline.duration:.2f} s\n"
        )
