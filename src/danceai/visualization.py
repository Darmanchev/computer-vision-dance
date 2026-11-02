import cv2
import numpy as np
from mediapipe.tasks.python import vision


def draw_pose_points(frame: np.ndarray, result: vision.PoseLandmarkerResult) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]

    for pose in result.pose_landmarks:
        for point in pose:
            if point.visibility is not None and point.visibility < 0.5:
                continue

            x = round(point.x * width)
            y = round(point.y * height)

            if 0 <= x < width and 0 <= y < height:
                cv2.circle(output, (x, y), 6, (0, 255, 0), -1)

    return output

