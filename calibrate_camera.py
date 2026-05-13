"""
Калібрування камери дрона на основі видеопараметрів і стандартних DJI-специфікацій
"""

import cv2
import numpy as np
from pathlib import Path
from droneVideoParser import DroneVideoCSVParser

def estimate_focal_length_from_video(video_path, assume_fov_deg=84.0):
    """
    Оцінює фокусну відстань на основі розміру кадру та припущення про FOV.
    
    Більшість DJI дронів мають FOV ~84-90°.
    
    Parameters
    ----------
    video_path : шлях до відео
    assume_fov_deg : припустимий кут огляду (ступені)
    
    Returns
    -------
    K : матриця калібрування (3x3)
    width, height : роздільна здатність відео
    estimated_fx_fy : оцінена фокусна відстань
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Не вдалося відкрити: {video_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    
    # FOV = 2 * arctan(width / (2 * fx))
    # fx = width / (2 * tan(FOV/2))
    
    fov_rad = np.deg2rad(assume_fov_deg)
    fx = (width / 2.0) / np.tan(fov_rad / 2.0)
    fy = fx  # Припускаємо квадратні пікселі
    
    cx = width / 2.0
    cy = height / 2.0
    
    K = np.array([
        [fx,  0.0, cx],
        [0.0, fy,  cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    
    print(f"📹 Відео: {Path(video_path).name}")
    print(f"   Розлік: {width}x{height}")
    print(f"   Припущений FOV: {assume_fov_deg}°")
    print(f"   Розраховане fx/fy: {fx:.1f}")
    print(f"   Головна точка (cx, cy): ({cx:.1f}, {cy:.1f})")
    print()
    
    return K, width, height, fx


def test_multiple_fov_values(video_path):
    """Тестує різні FOV значення для порівняння."""
    print("=" * 60)
    print("ТЕСТУВАННЯ РІЗНИХ FOV ЗНАЧЕНЬ")
    print("=" * 60)
    print()
    
    fov_values = [70.0, 75.0, 80.0, 84.0, 88.0, 90.0, 95.0, 100.0]
    
    for fov in fov_values:
        K, w, h, fx = estimate_focal_length_from_video(video_path, assume_fov_deg=fov)
        print(f"FOV={fov:5.1f}° → fx/fy={fx:7.1f}   (поточне значення: 1400.0)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Калібрування камери дрона")
    parser.add_argument("--video", type=str, default="dataset/drone_footage/23-02-01_FR_F01.mp4",
                        help="Шлях до відеофайлу")
    parser.add_argument("--fov", type=float, default=84.0,
                        help="Припущений FOV (ступені)")
    parser.add_argument("--test-fov", action="store_true",
                        help="Тестувати різні FOV значення")
    
    args = parser.parse_args()
    
    if args.test_fov:
        test_multiple_fov_values(args.video)
    else:
        K, w, h, fx = estimate_focal_length_from_video(args.video, assume_fov_deg=args.fov)
        
        print("=" * 60)
        print("РЕКОМЕНДОВАНА МАТРИЦЯ КАЛІБРУВАННЯ")
        print("=" * 60)
        print(K)
        print()
        print("Скопіюйте цю матрицю у droneVideoParser.py, метод _default_k(),")
        print("замінивши fx=fy=1400.0 на fx=fy={:.1f}".format(fx))
