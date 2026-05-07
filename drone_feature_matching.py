"""Feature matching for drone video frames."""

import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

import cv2
import numpy as np

from drone_parser import DroneVideoDataset

class DetectorType(Enum):
    ORB = auto()
    OPTICAL_FLOW = auto()


@dataclass
class MatchResult:
    pts_prev: np.ndarray
    pts_curr: np.ndarray
    kp_prev: list
    kp_curr: list
    good_matches: list
    elapsed_ms: float
    detector_name: str


def _build_bf_matcher() -> cv2.BFMatcher:
    return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


def _lowe_ratio_test(matches: list, ratio_threshold: float) -> list:
    good = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good.append(m)
    return good


def _matched_points(kp_prev: list, kp_curr: list, good_matches: list) -> Tuple[np.ndarray, np.ndarray]:
    pts_prev = np.float32([kp_prev[m.queryIdx].pt for m in good_matches])
    pts_curr = np.float32([kp_curr[m.trainIdx].pt for m in good_matches])
    return pts_prev, pts_curr


class FeatureMatcher:
    def __init__(
        self,
        detector_type: DetectorType = DetectorType.ORB,
        ratio_threshold: float = 0.75,
        orb_n_features: int = 2000,
        flow_max_corners: int = 2000,
        flow_quality_level: float = 0.01,
        flow_min_distance: float = 7.0,
        flow_block_size: int = 7,
        flow_fb_max_error: float = 2.0,
        lk_win_size: Tuple[int, int] = (21, 21),
        lk_max_level: int = 3,
        lk_criteria: Tuple[int, int, float] = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        ),
    ):
        self.detector_type = detector_type
        self.ratio_threshold = ratio_threshold

        self.flow_max_corners = int(flow_max_corners)
        self.flow_quality_level = float(flow_quality_level)
        self.flow_min_distance = float(flow_min_distance)
        self.flow_block_size = int(flow_block_size)
        self.flow_fb_max_error = float(flow_fb_max_error)
        self.lk_win_size = tuple(lk_win_size)
        self.lk_max_level = int(lk_max_level)
        self.lk_criteria = lk_criteria

        if detector_type == DetectorType.ORB:
            self.detector = cv2.ORB_create(nfeatures=orb_n_features)
            self.matcher = _build_bf_matcher()
        else:
            self.detector = None
            self.matcher = None

    def match(
        self,
        img_prev: np.ndarray,
        img_curr: np.ndarray,
        min_matches: int = 12,
    ) -> Optional[MatchResult]:
        t0 = time.perf_counter()

        gray_prev = self._to_gray(img_prev)
        gray_curr = self._to_gray(img_curr)

        if self.detector_type == DetectorType.OPTICAL_FLOW:
            pts_prev = cv2.goodFeaturesToTrack(
                gray_prev,
                maxCorners=self.flow_max_corners,
                qualityLevel=self.flow_quality_level,
                minDistance=self.flow_min_distance,
                blockSize=self.flow_block_size,
            )
            if pts_prev is None or len(pts_prev) < min_matches:
                return None

            pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(
                gray_prev,
                gray_curr,
                pts_prev,
                None,
                winSize=self.lk_win_size,
                maxLevel=self.lk_max_level,
                criteria=self.lk_criteria,
            )
            if pts_curr is None or status is None:
                return None

            pts_back, status_back, _ = cv2.calcOpticalFlowPyrLK(
                gray_curr,
                gray_prev,
                pts_curr,
                None,
                winSize=self.lk_win_size,
                maxLevel=self.lk_max_level,
                criteria=self.lk_criteria,
            )
            if pts_back is None or status_back is None:
                return None

            pts_prev_xy = pts_prev.reshape(-1, 2)
            pts_back_xy = pts_back.reshape(-1, 2)
            fb_err = np.linalg.norm(pts_prev_xy - pts_back_xy, axis=1)
            valid = (status.ravel() == 1) & (status_back.ravel() == 1)
            valid &= fb_err <= self.flow_fb_max_error

            pts_prev_valid = pts_prev_xy[valid]
            pts_curr_valid = pts_curr.reshape(-1, 2)[valid]

            if len(pts_prev_valid) < min_matches:
                return None

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            kp_prev, kp_curr, good_matches = self._build_visualization_payload(
                pts_prev_valid, pts_curr_valid
            )

            return MatchResult(
                pts_prev=pts_prev_valid.astype(np.float32),
                pts_curr=pts_curr_valid.astype(np.float32),
                kp_prev=kp_prev,
                kp_curr=kp_curr,
                good_matches=good_matches,
                elapsed_ms=elapsed_ms,
                detector_name=self.detector_type.name,
            )

        kp_prev, des_prev = self.detector.detectAndCompute(gray_prev, None)
        kp_curr, des_curr = self.detector.detectAndCompute(gray_curr, None)

        if des_prev is None or des_curr is None:
            return None
        if len(kp_prev) < min_matches or len(kp_curr) < min_matches:
            return None

        raw_matches = self.matcher.knnMatch(des_prev, des_curr, k=2)
        good_matches = _lowe_ratio_test(raw_matches, self.ratio_threshold)

        if len(good_matches) < min_matches:
            return None

        pts_prev, pts_curr = _matched_points(kp_prev, kp_curr, good_matches)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return MatchResult(
            pts_prev=pts_prev,
            pts_curr=pts_curr,
            kp_prev=kp_prev,
            kp_curr=kp_curr,
            good_matches=good_matches,
            elapsed_ms=elapsed_ms,
            detector_name=self.detector_type.name,
        )

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    @staticmethod
    def _build_visualization_payload(
        pts_prev: np.ndarray,
        pts_curr: np.ndarray,
    ) -> Tuple[list, list, list]:
        kp_prev = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in pts_prev]
        kp_curr = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in pts_curr]
        good_matches = [cv2.DMatch(i, i, 0.0) for i in range(len(pts_prev))]
        return kp_prev, kp_curr, good_matches

    def draw_matches(
        self,
        img_prev: np.ndarray,
        img_curr: np.ndarray,
        result: MatchResult,
        max_draw: int = 50,
    ) -> np.ndarray:
        vis = cv2.drawMatches(
            img_prev,
            result.kp_prev,
            img_curr,
            result.kp_curr,
            result.good_matches[:max_draw],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        label = (
            f"{result.detector_name} | matches: {len(result.good_matches)} | "
            f"{result.elapsed_ms:.1f} ms"
        )
        cv2.putText(
            vis,
            label,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        return vis


if __name__ == "__main__":
    DATASET_PATH = "dataset"
    VIDEO_PATH = "drone_footage/23-02-01_FR_F01_V01.MP4"
    CSV_PATH = "drone_footage/23-02-01_FR_F01.csv"

    dataset = DroneVideoDataset(
        DATASET_PATH,
        video_path=VIDEO_PATH,
        csv_path=CSV_PATH,
        target_fps=5.0,
        frame_stride=None,
    )
    print(dataset.summary())

    matcher = FeatureMatcher(DetectorType.OPTICAL_FLOW)

    for idx in range(len(dataset) - 1):
        img_prev, _ = dataset[idx]
        img_curr, _ = dataset[idx + 1]

        result = matcher.match(img_prev, img_curr)
        if result:
            print(
                f"[{idx:04d}] matches={len(result.good_matches)} "
                f"time={result.elapsed_ms:.1f} ms"
            )
            vis = matcher.draw_matches(img_prev, img_curr, result, max_draw=80)
            cv2.imshow("Drone Feature Matches", vis)
        else:
            print(f"[{idx:04d}] not enough matches")

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    dataset.close()
