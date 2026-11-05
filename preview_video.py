import argparse
import math
from contextlib import closing
from itertools import chain
from pathlib import Path

import cv2

from danceai.pose import create_landmarker, detect_pose
from danceai.video import read_frames
from danceai.visualization import draw_pose_points


def main():
    parser = argparse.ArgumentParser(description="Draw poses on a video")
    parser.add_argument("video", help="Path to the input video")
    args = parser.parse_args()

    source = Path(args.video)
    output_path = source.with_name(f"{source.stem}_pose.mp4")

    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {source}")
        fps = capture.get(cv2.CAP_PROP_FPS)
    finally:
        capture.release()

    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid video FPS: {fps}")

    with closing(read_frames(source)) as frames:
        first_timestamp, first_frame = next(frames)
        height, width = first_frame.shape[:2]

        with create_landmarker("assets/models/pose_landmarker_full.task") as model:
            writer = cv2.VideoWriter(
                str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )

            try:
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot create video:{output_path}")

                total = 0
                detected = 0

                for timestamp_ms, frame in chain([(first_timestamp, first_frame)], frames):
                    result = detect_pose(model, frame, timestamp_ms)
                    output = draw_pose_points(frame, result)
                    writer.write(output)

                    total += 1
                    detected += bool(result.pose_landmarks)

                    if total % 100 == 0:
                        print(f"Processed {total} frames")
            finally:
                writer.release()
    print(f"Pose detected in {detected}/{total} frames")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
