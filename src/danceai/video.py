import math
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


def read_frames(video_path: str | Path) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (timestamp_ms, BGR frame) for constant-frame-rate video."""

    capture = cv2.VideoCapture(str(video_path))

    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS)

        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Invalid video FPS: {fps}")

        frame_index = 0

        while True:
            success, frame = capture.read()

            if not success:
                if frame_index == 0:
                    raise ValueError("Video contains no readable frames")
                break

            timestamp_ms = round(frame_index * 1000 / fps)
            yield timestamp_ms, frame
            frame_index += 1

    finally:
        capture.release()
