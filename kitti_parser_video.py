import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple, Optional, List, Any

import cv2
import numpy as np


@dataclass
class TelemetrySample:
    time_s: float
    lat: float
    lon: float
    alt_m: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float


class VideoTelemetryParser:
    """Video + telemetry parser with time-based interpolation."""

    FT_TO_M = 0.3048

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
    ):
        self.root = Path(dataset_path)
        self.video_path = (
            self.root / video_path if not Path(video_path).is_absolute() else Path(video_path)
        )
        self.csv_path = (
            self.root / csv_path if not Path(csv_path).is_absolute() else Path(csv_path)
        )
        self.start_frame = int(start_frame)
        self.time_column = time_column
        self.time_offset = float(time_offset)
        self.normalize_time = bool(normalize_time)

        self._validate_paths()

        self._cap = self._open_capture()
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        if self._fps <= 0:
            raise ValueError("Invalid FPS; cannot compute frame timestamps.")

        self._rows = self._load_csv_rows(self.csv_path)
        self._init_telemetry_series()

        if self.start_frame < 0 or self.start_frame >= self._frame_count:
            raise ValueError(f"start_frame {self.start_frame} out of range [0, {self._frame_count})")

        length = self._frame_count - self.start_frame
        if max_frames is not None:
            length = min(length, int(max_frames))
        self._length = length

    def _validate_paths(self) -> None:
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

    def _open_capture(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise IOError(f"Failed to open video: {self.video_path}")
        return cap

    @staticmethod
    def _convert_value(value: str) -> Any:
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
            rows: List[dict] = []
            for row in reader:
                converted = {k: self._convert_value(v) for k, v in row.items()}
                rows.append(converted)
        return rows

    def _init_telemetry_series(self) -> None:
        times = []
        lat = []
        lon = []
        alt_m = []
        yaw = []
        pitch = []
        roll = []

        for row in self._rows:
            t = row.get(self.time_column)
            if t is None:
                continue
            try:
                t_val = float(t)
            except (TypeError, ValueError):
                continue

            lat_val = row.get("OSD.latitude")
            lon_val = row.get("OSD.longitude")
            alt_ft = self._first_non_null(
                row,
                ["OSD.height [ft]", "OSD.altitude [ft]", "OSD.vpsHeight [ft]"],
            )
            yaw_val = row.get("OSD.yaw") if row.get("OSD.yaw") is not None else row.get("OSD.yaw [360]")
            pitch_val = row.get("OSD.pitch")
            roll_val = row.get("OSD.roll")

            if lat_val is None or lon_val is None or alt_ft is None:
                continue

            times.append(t_val)
            lat.append(float(lat_val))
            lon.append(float(lon_val))
            alt_m.append(float(alt_ft) * self.FT_TO_M)
            yaw.append(float(yaw_val) if yaw_val is not None else float("nan"))
            pitch.append(float(pitch_val) if pitch_val is not None else float("nan"))
            roll.append(float(roll_val) if roll_val is not None else float("nan"))

        if len(times) < 2:
            raise ValueError("Not enough telemetry rows with valid time/position data.")

        times_arr = np.array(times, dtype=np.float64)
        if self.normalize_time:
            times_arr = times_arr - times_arr[0]

        self._tele_times = times_arr
        self._lat_series = np.array(lat, dtype=np.float64)
        self._lon_series = np.array(lon, dtype=np.float64)
        self._alt_series = np.array(alt_m, dtype=np.float64)
        self._yaw_series = np.array(yaw, dtype=np.float64)
        self._pitch_series = np.array(pitch, dtype=np.float64)
        self._roll_series = np.array(roll, dtype=np.float64)

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

    def __len__(self) -> int:
        return self._length

    def _frame_time(self, frame_idx: int) -> float:
        t = frame_idx / self._fps
        return t + self.time_offset

    def _interp(self, t: float, times: np.ndarray, values: np.ndarray) -> float:
        t_clamped = float(np.clip(t, times[0], times[-1]))
        return float(np.interp(t_clamped, times, values))

    def _interp_angle_deg(self, t: float, times: np.ndarray, values_deg: np.ndarray) -> float:
        mask = np.isfinite(values_deg)
        if mask.sum() < 2:
            return 0.0
        times_valid = times[mask]
        vals_rad = np.deg2rad(values_deg[mask])
        vals_unwrap = np.unwrap(vals_rad)
        t_clamped = float(np.clip(t, times_valid[0], times_valid[-1]))
        interp_rad = np.interp(t_clamped, times_valid, vals_unwrap)
        return float(np.rad2deg(interp_rad))

    def get_telemetry(self, frame_idx: int) -> TelemetrySample:
        t = self._frame_time(frame_idx)
        lat = self._interp(t, self._tele_times, self._lat_series)
        lon = self._interp(t, self._tele_times, self._lon_series)
        alt_m = self._interp(t, self._tele_times, self._alt_series)
        yaw = self._interp_angle_deg(t, self._tele_times, self._yaw_series)
        pitch = self._interp_angle_deg(t, self._tele_times, self._pitch_series)
        roll = self._interp_angle_deg(t, self._tele_times, self._roll_series)

        return TelemetrySample(
            time_s=t,
            lat=lat,
            lon=lon,
            alt_m=alt_m,
            yaw_deg=yaw,
            pitch_deg=pitch,
            roll_deg=roll,
        )

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, TelemetrySample]:
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Index {idx} out of range [0, {self._length})")

        frame_idx = self.start_frame + idx
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise IOError(f"Failed to read frame {frame_idx} from {self.video_path}")

        telemetry = self.get_telemetry(frame_idx)
        return frame, telemetry

    def __iter__(self) -> Iterator[Tuple[np.ndarray, TelemetrySample]]:
        cap = self._open_capture()
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        for i in range(self._length):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_idx = self.start_frame + i
            telemetry = self.get_telemetry(frame_idx)
            yield frame, telemetry
        cap.release()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def summary(self) -> str:
        return (
            f"VideoTelemetryParser\n"
            f"  Video     : {self.video_path}\n"
            f"  CSV       : {self.csv_path}\n"
            f"  Start     : {self.start_frame}\n"
            f"  Frames    : {self._length}\n"
            f"  FPS       : {self._fps:.2f}\n"
            f"  Time col  : {self.time_column}\n"
        )


if __name__ == "__main__":
    DATASET_PATH = "dataset/"
    VIDEO_PATH = "drone_footage/23-02-01_FR_F01_V01.MP4"
    CSV_PATH = "drone_footage/23-02-01_FR_F01.csv"

    parser = VideoTelemetryParser(
        DATASET_PATH,
        video_path=VIDEO_PATH,
        csv_path=CSV_PATH,
        time_column="OSD.flyTime [s]",
        time_offset=0.0,
        normalize_time=True,
    )
    print(parser.summary())

    for idx, (frame, tele) in enumerate(parser):
        print(
            f"[{idx:04d}] frame shape: {frame.shape} | "
            f"lat={tele.lat:.6f} lon={tele.lon:.6f} alt_m={tele.alt_m:.2f}"
        )
        if idx == 4:
            print("  ...")
            break

    parser.close()
