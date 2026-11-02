from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import numpy as np


def create_landmarker(model_path: str | Path) -> vision.PoseLandmarker:
    model_path = Path(model_path)

    if not model_path.is_file():
        raise FileNotFoundError(f"Pose model not found: {model_path}")

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )

    return vision.PoseLandmarker.create_from_options(options)


def detect_pose(landmarker: vision.PoseLandmarker, frame: np.ndarray, timestamp_ms: int) -> vision.PoseLandmarkerResult:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    return landmarker.detect_for_video(image, timestamp_ms)