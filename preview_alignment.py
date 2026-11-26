import argparse
import os
import math
import tempfile
from pathlib import Path

import cv2
import numpy as np

from danceai.alignment import align_measurements
from danceai.comparison import compare_angles
from danceai.pipeline import load_measurements
from danceai.processing import JOINTS

from dotenv import load_dotenv

load_dotenv()

VIDEO_PATH1 = os.environ["VIDEO_PATH1"]
VIDEO_PATH2 = os.environ["VIDEO_PATH2"]


def draw_reference_arm(frame, reference_size, reference_pose, learner_pose):
    shoulder, elbow, wrist = JOINTS["left elbow"]
    if len(reference_pose) <= wrist or len(learner_pose) <= shoulder:
        return

    reference_arm = [reference_pose[index] for index in (shoulder, elbow, wrist)]
    learner_shoulder = learner_pose[shoulder]
    points = [*reference_arm, learner_shoulder]
    if any(point["visibility"] is None or not point["visibility"] >= 0.5 for point in points):
        return

    arm = np.array([(point["x"], point["y"]) for point in reference_arm]) * reference_size
    height, width = frame.shape[:2]
    anchor = np.array([learner_shoulder["x"] * width, learner_shoulder["y"] * height])
    if not np.isfinite(arm).all() or not np.isfinite(anchor).all():
        return

    shoulder_px, elbow_px, wrist_px = np.rint(arm - arm[0] + anchor).astype(int)
    cv2.line(frame, tuple(shoulder_px), tuple(elbow_px), (0, 255, 255), 3, cv2.LINE_AA)
    cv2.line(frame, tuple(elbow_px), tuple(wrist_px), (0, 255, 255), 3, cv2.LINE_AA)


def save_alignment_preview(reference_video, learner_video, reference, learner, path, output):
    """Save DTW pairs at 30 FPS; source timing is shown in labels, without audio."""
    output = Path(output)
    sources = [Path(reference_video), Path(learner_video)]
    if output.resolve() in [source.resolve() for source in sources]:
        raise ValueError("Output must differ from the source videos")
    if not path:
        raise ValueError("Alignment path is empty")

    captures = []
    writer = None
    temporary = None
    try:
        for source in sources:
            capture = cv2.VideoCapture(str(source))
            captures.append(capture)
            if not capture.isOpened():
                raise ValueError(f"Cannot open video: {source}")
        rates = [capture.get(cv2.CAP_PROP_FPS) for capture in captures]
        if any(not math.isfinite(fps) or fps <= 0 for fps in rates):
            raise ValueError("Invalid source video FPS")

        with tempfile.NamedTemporaryFile(suffix=".mp4", dir=output.parent, delete=False) as file:
            temporary = Path(file.name)
        writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), 30, (720, 520))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create video: {output}")

        positions = [-1, -1]
        cached = [None, None]
        for i, j in path:
            panels = []
            for side, (frames, index, label) in enumerate(
                ((reference, i, "Reference"), (learner, j, "Learner"))
            ):
                timestamp = frames[index]["timestamp_ms"]
                target = round(timestamp * rates[side] / 1000)
                if target < positions[side]:
                    raise ValueError("Alignment timestamps must move forward")
                while positions[side] < target:
                    success, frame = captures[side].read()
                    if not success:
                        raise RuntimeError(f"Cannot read {label} frame at {timestamp} ms")
                    cached[side] = frame
                    positions[side] += 1
                frame = cached[side]
                height, width = frame.shape[:2]
                scale = min(360 / width, 480 / height)
                resized = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
                if side == 0:
                    reference_size = (resized.shape[1], resized.shape[0])
                else:
                    draw_reference_arm(
                        resized, reference_size, reference[i].get("landmarks", []), learner[j].get("landmarks", [])
                    )

                panel = np.zeros((520, 360, 3), dtype=np.uint8)
                x = (360 - resized.shape[1]) // 2
                y = 40 + (480 - resized.shape[0]) // 2
                panel[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
                cv2.putText(panel, f"{label}: {timestamp / 1000:.2f}s", (10, 27),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
                panels.append(panel)
            writer.write(np.hstack(panels))

        writer.release()
        writer = None
        check = cv2.VideoCapture(str(temporary))
        try:
            count = 0
            while check.read()[0]:
                count += 1
            if count != len(path):
                raise RuntimeError(f"Output verification failed: {count}/{len(path)} frames")
        finally:
            check.release()
        temporary.replace(output)
    finally:
        if writer is not None:
            writer.release()
        for capture in captures:
            capture.release()
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Compare two dance videos using saved measurements")
    parser.add_argument("reference", nargs="?", default=VIDEO_PATH1)
    parser.add_argument("learner", nargs="?", default=VIDEO_PATH2)
    parser.add_argument("--output", default="data/alignment_preview.mp4")
    args = parser.parse_args()

    measurements = []
    for video in (args.reference, args.learner):
        source = Path(video)
        measurements.append(load_measurements(source.with_name(f"{source.stem}-measurement.json")))
    reference, learner, path, distance = align_measurements(*measurements)
    print(f"Usable frames: reference {len(reference)}/{len(measurements[0])}, "
          f"learner {len(learner)}/{len(measurements[1])}")
    print(f"DTW distance (alignment cost): {distance:.2f}")
    for joint, error in compare_angles(reference, learner, path).items():
        print(f"{joint}: {error:.2f} degrees mean error")
    save_alignment_preview(args.reference, args.learner, reference, learner, path, args.output)
    print(f"Saved {args.output} (one DTW pair per frame, 30 FPS, no audio)")


if __name__ == "__main__":
    main()
