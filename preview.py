import argparse
from contextlib import closing

import cv2

from danceai.pose import create_landmarker, detect_pose
from danceai.processing import joint_angles
from danceai.video import read_frames
from danceai.visualization import draw_pose_points


def main():
    parser = argparse.ArgumentParser(description="Preview pose detection")
    parser.add_argument("video", help="Path to the input video")
    args = parser.parse_args()

    with closing(read_frames(args.video)) as frames:
        timestamp_ms, frame = next(frames)

    with create_landmarker("assets/models/pose_landmarker_full.task") as model:
        result = detect_pose(model, frame, timestamp_ms)

    output = draw_pose_points(frame, result)
    output_path = "data/pose_preview.jpg"

    if not cv2.imwrite(output_path, output):
        raise RuntimeError(f"Could not save {output_path}")

    print(f"People detected: {len(result.pose_landmarks)}")
    if result.pose_world_landmarks:
        for joint, angle in joint_angles(result.pose_world_landmarks[0]).items():
            measurement = f"{angle:.1f}°" if angle is not None else "unavailable"
            print(f"{joint}: {measurement}")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
