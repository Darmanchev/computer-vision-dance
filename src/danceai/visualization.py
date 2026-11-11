import cv2
import numpy as np
from mediapipe.tasks.python import vision

BODY_CONNECTIONS = (
    (11, 12),                       # Shoulders
    (11, 13), (13, 15), (15,19),    # Left arm
    (12, 14), (14, 16), (16,20),    # Right arm
    (11, 23), (12, 24), (23, 24),   # Torso
    (23, 25), (25, 27),             # Left leg
    (24, 26), (26, 28),             # Right leg
    (27, 29), (29, 31), (27, 31),   # Left foot
    (28, 30), (30, 32), (28, 32),   # Right foot
)

STANDALONE_POINTS = {0, 19, 20}

DRAWN_POINTS = {index for pair in BODY_CONNECTIONS for index in pair} | STANDALONE_POINTS


def draw_pose_points(frame: np.ndarray, result: vision.PoseLandmarkerResult) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]

    for pose in result.pose_landmarks:
        visible_points = {}

        for index, point in enumerate(pose):
            if index not in DRAWN_POINTS:
                continue


            if point.visibility is not None and point.visibility < 0.5:
                continue

            x = round(point.x * width)
            y = round(point.y * height)

            if 0 <= x < width and 0 <= y < height:
                visible_points[index] = (x, y)

        for start, end in  BODY_CONNECTIONS:
            if start in visible_points and end in visible_points:
                cv2.line(output, visible_points[start], visible_points[end], (0, 255, 255), 3)


        for position in visible_points.values():
            cv2.circle(output, position, 6, (0, 255, 0), -1)

    return output


