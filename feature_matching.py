"""
Етап 2: Виявлення та зіставлення ключових точок
Feature Extraction & Matching — SIFT / SURF / ORB
"""

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import cv2
import numpy as np
import os
from parser import KITTIOdometryParser


# ══════════════════════════════════════════════════════════════════════════
# Типи
# ══════════════════════════════════════════════════════════════════════════

class DetectorType(Enum):
    SIFT = auto()
    SURF = auto()
    ORB  = auto()


@dataclass
class MatchResult:
    """Результат зіставлення між двома кадрами."""
    pts_prev:       np.ndarray          # (N, 2) — точки на кадрі t-1
    pts_curr:       np.ndarray          # (N, 2) — точки на кадрі t
    kp_prev:        list                # повний список keypoints (кадр t-1)
    kp_curr:        list                # повний список keypoints (кадр t)
    good_matches:   list                # відфільтровані DMatch-об'єкти
    elapsed_ms:     float = 0.0         # час обробки пари кадрів (мс)
    detector_name:  str   = ""


# ══════════════════════════════════════════════════════════════════════════
# Допоміжні функції
# ══════════════════════════════════════════════════════════════════════════

def _build_flann_matcher(detector: DetectorType) -> cv2.FlannBasedMatcher:
    """FLANN для SIFT / SURF (float-дескриптори)."""
    FLANN_INDEX_KDTREE = 1
    index_params  = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    return cv2.FlannBasedMatcher(index_params, search_params)


def _build_bf_matcher() -> cv2.BFMatcher:
    """Brute-Force для ORB (бінарні дескриптори)."""
    return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


def _lowe_ratio_test(
    matches: list,
    ratio_threshold: float = 0.75,
) -> list:
    """
    Тест Лоу (Lowe's ratio test).
    Залишає пари, де відстань до найкращого збігу
    значно менша, ніж до другого.
    """
    good = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good.append(m)
    return good


def _matched_points(
    kp_prev: list,
    kp_curr: list,
    good_matches: list,
) -> tuple[np.ndarray, np.ndarray]:
    """Перетворює DMatch-об'єкти на масиви координат."""
    pts_prev = np.float32([kp_prev[m.queryIdx].pt for m in good_matches])
    pts_curr = np.float32([kp_curr[m.trainIdx].pt for m in good_matches])
    return pts_prev, pts_curr


# ══════════════════════════════════════════════════════════════════════════
# Основний клас
# ══════════════════════════════════════════════════════════════════════════

class FeatureMatcher:
    """
    Уніфікований інтерфейс для SIFT, SURF та ORB.

    Приклад використання:
        matcher = FeatureMatcher(DetectorType.SIFT)
        result  = matcher.match(frame_prev, frame_curr)
        print(f"Знайдено {len(result.good_matches)} збігів")
    """

    def __init__(
        self,
        detector_type: DetectorType = DetectorType.ORB,
        ratio_threshold: float = 0.75,
        # ORB-specific
        orb_n_features: int = 3000,
        # SURF-specific
        surf_hessian_threshold: float = 400.0,
        # SIFT-specific
        sift_n_features: int = 0,           # 0 = без обмежень
    ):
        self.detector_type   = detector_type
        self.ratio_threshold = ratio_threshold
        self.detector        = self._init_detector(
            detector_type, orb_n_features, surf_hessian_threshold, sift_n_features
        )
        self.matcher         = self._init_matcher(detector_type)

    # ── ініціалізація ─────────────────────────────────────────────────────

    @staticmethod
    def _init_detector(
        det_type: DetectorType,
        orb_n: int,
        surf_thresh: float,
        sift_n: int,
    ):
        if det_type == DetectorType.SIFT:
            return cv2.SIFT_create(nfeatures=sift_n)
        elif det_type == DetectorType.SURF:
            # SURF вимагає opencv-contrib-python
            return cv2.xfeatures2d.SURF_create(hessianThreshold=surf_thresh)
        elif det_type == DetectorType.ORB:
            return cv2.ORB_create(nfeatures=orb_n)
        else:
            raise ValueError(f"Невідомий тип детектора: {det_type}")

    @staticmethod
    def _init_matcher(det_type: DetectorType):
        if det_type in (DetectorType.SIFT, DetectorType.SURF):
            return _build_flann_matcher(det_type)
        else:  # ORB
            return _build_bf_matcher()

    # ── основний метод ────────────────────────────────────────────────────

    def match(
        self,
        img_prev: np.ndarray,
        img_curr: np.ndarray,
        min_matches: int = 8,
    ) -> Optional[MatchResult]:
        """
        Знаходить та фільтрує збіги між двома сусідніми кадрами.

        Parameters
        ----------
        img_prev : np.ndarray   — кадр t-1 (grayscale або BGR)
        img_curr : np.ndarray   — кадр t   (grayscale або BGR)
        min_matches : int       — мінімальна кількість збігів;
                                  якщо менше — повертає None

        Returns
        -------
        MatchResult або None, якщо збігів недостатньо.
        """
        t0 = time.perf_counter()

        gray_prev = self._to_gray(img_prev)
        gray_curr = self._to_gray(img_curr)

        # 1. Виявлення ключових точок та дескрипторів
        kp_prev, des_prev = self.detector.detectAndCompute(gray_prev, None)
        kp_curr, des_curr = self.detector.detectAndCompute(gray_curr, None)

        if des_prev is None or des_curr is None:
            return None
        if len(kp_prev) < min_matches or len(kp_curr) < min_matches:
            return None

        # 2. Зіставлення (kNN з k=2 для ratio test)
        des_prev, des_curr = self._ensure_float(des_prev, des_curr)
        raw_matches = self.matcher.knnMatch(des_prev, des_curr, k=2)

        # 3. Фільтрація за тестом Лоу
        good_matches = _lowe_ratio_test(raw_matches, self.ratio_threshold)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if len(good_matches) < min_matches:
            return None

        pts_prev, pts_curr = _matched_points(kp_prev, kp_curr, good_matches)

        return MatchResult(
            pts_prev      = pts_prev,
            pts_curr      = pts_curr,
            kp_prev       = kp_prev,
            kp_curr       = kp_curr,
            good_matches  = good_matches,
            elapsed_ms    = elapsed_ms,
            detector_name = self.detector_type.name,
        )

    # ── утиліти ───────────────────────────────────────────────────────────

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        """BGR → grayscale (якщо вже grayscale — без змін)."""
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _ensure_float(
        self,
        des_prev: np.ndarray,
        des_curr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """FLANN вимагає float32; ORB повертає uint8 — але ми використовуємо BF."""
        if self.detector_type in (DetectorType.SIFT, DetectorType.SURF):
            return des_prev.astype(np.float32), des_curr.astype(np.float32)
        return des_prev, des_curr

    # ── візуалізація ──────────────────────────────────────────────────────

    def draw_matches(
        self,
        img_prev: np.ndarray,
        img_curr: np.ndarray,
        result: MatchResult,
        max_draw: int = 50,
    ) -> np.ndarray:
        """Повертає зображення з намальованими збігами (для налагодження)."""
        vis = cv2.drawMatches(
            img_prev, result.kp_prev,
            img_curr, result.kp_curr,
            result.good_matches[:max_draw],
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        label = (
            f"{result.detector_name} | "
            f"matches: {len(result.good_matches)} | "
            f"{result.elapsed_ms:.1f} ms"
        )
        cv2.putText(vis, label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return vis


# ══════════════════════════════════════════════════════════════════════════
# Швидкий тест без датасету
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Генеруємо два синтетичні кадри для демонстрації роботи модуля
    np.random.seed(42)
    dummy = (np.random.rand(480, 640) * 255).astype(np.uint8)

    for det in DetectorType:
        try:
            matcher = FeatureMatcher(det)
            result  = matcher.match(dummy, dummy)
            if result:
                print(f"{det.name:4s}: {len(result.good_matches):4d} збігів | "
                      f"{result.elapsed_ms:.1f} мс")
            else:
                print(f"{det.name:4s}: недостатньо збігів")
        except cv2.error as e:
            print(f"{det.name:4s}: недоступний ({e})")

    # Налаштування датасету
    DATASET_PATH = "dataset/"  # Шлях до датасету
    SEQUENCE = "00"           # Номер послідовності
    CAMERA = "image_0"        # Камера (image_0 для grayscale)

    # Ініціалізація парсера
    parser = KITTIOdometryParser(DATASET_PATH, sequence=SEQUENCE, camera=CAMERA)
    print(parser.summary())

    # Використання FeatureMatcher для кожної пари кадрів
    matcher = FeatureMatcher(DetectorType.SIFT)  # Можна змінити на SURF або ORB

    for idx in range(len(parser) - 1):
        img_prev, pose_prev = parser[idx]
        img_curr, pose_curr = parser[idx + 1]

        result = matcher.match(img_prev, img_curr)
        if result:
            print(f"[{idx:04d}] Знайдено {len(result.good_matches)} збігів | "
                  f"{result.elapsed_ms:.1f} мс")
        else:
            print(f"[{idx:04d}] Недостатньо збігів")
