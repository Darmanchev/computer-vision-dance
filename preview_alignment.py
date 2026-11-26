import argparse
import os
import math
import tempfile
from contextlib import closing
from pathlib import Path

import cv2
import numpy as np

from danceai.alignment import align_measurements
from danceai.comparison import compare_angles
from danceai.feedback import find_direction_feedback, find_joint_feedback
from danceai.pipeline import load_measurements
from danceai.processing import JOINTS, visible_image_points
from danceai.video import read_frames
from danceai.visualization import BODY_CONNECTIONS

from dotenv import load_dotenv

load_dotenv()

VIDEO_PATH1 = os.environ["VIDEO_PATH1"]
VIDEO_PATH2 = os.environ["VIDEO_PATH2"]

LIMB_JOINTS = {
    "left elbow": "Left arm",
    "right elbow": "Right arm",
    "left knee": "Left leg",
    "right knee": "Right leg",
}
REFERENCE_COLOR = (255, 190, 100)
CORRECTION_COLOR = (0, 255, 255)


def draw_reference_pose(frame, reference, learner, active_joints):
    anchors = [index for index in (11, 12, 23, 24) if index in reference and index in learner]
    if not anchors:
        return

    reference_torso = np.array([reference[index] for index in anchors])
    learner_torso = np.array([learner[index] for index in anchors])
    reference_center = reference_torso.mean(axis=0)
    learner_center = learner_torso.mean(axis=0)
    reference_spread = np.linalg.norm(reference_torso - reference_center)
    learner_spread = np.linalg.norm(learner_torso - learner_center)
    scale = learner_spread / reference_spread if reference_spread and learner_spread else 1
    aligned = {
        index: tuple(np.rint((point - reference_center) * scale + learner_center).astype(int))
        for index, point in reference.items()
    }

    highlighted = set()
    for joint in active_joints:
        start, middle, end = JOINTS[joint]
        highlighted.update(((start, middle), (middle, end)))

    for start, end in BODY_CONNECTIONS:
        if start in aligned and end in aligned:
            is_highlighted = (start, end) in highlighted
            color = CORRECTION_COLOR if is_highlighted else REFERENCE_COLOR
            cv2.line(frame, aligned[start], aligned[end], color, 3 if is_highlighted else 2, cv2.LINE_AA)


def save_alignment_preview(reference_video, learner_video, reference, learner, path, output, *, feedback=None):
    """Save DTW pairs at 30 FPS with source timestamps and active corrections."""
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
        writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), 30, (720, 608))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create video: {output}")

        positions = [-1, -1]
        cached = [None, None]
        for i, j in path:
            learner_time = learner[j]["timestamp_ms"]
            active_feedback = {}
            for joint, intervals in (feedback or {}).items():
                active = [(cue, gap) for start, end, cue, gap in intervals if start <= learner_time <= end]
                if active:
                    active_feedback[joint] = active
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
                points = visible_image_points(frames[index].get("landmarks", []), resized.shape[1::-1])
                if side == 0:
                    reference_points = points
                else:
                    draw_reference_pose(resized, reference_points, points, active_feedback)

                panel = np.zeros((520, 360, 3), dtype=np.uint8)
                x = (360 - resized.shape[1]) // 2
                y = 40 + (480 - resized.shape[0]) // 2
                panel[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
                cv2.putText(panel, f"{label}: {timestamp / 1000:.2f}s", (10, 27),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
                panels.append(panel)
            preview = np.zeros((608, 720, 3), dtype=np.uint8)
            preview[:520] = np.hstack(panels)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(preview, "Blue: reference pose", (10, 543), font, 0.55, REFERENCE_COLOR, 1, cv2.LINE_AA)
            cv2.putText(preview, "Yellow: correction below Learner", (10, 565), font, 0.55, CORRECTION_COLOR, 1, cv2.LINE_AA)
            ranked = sorted(active_feedback, key=lambda joint: max(gap for _, gap in active_feedback[joint]), reverse=True)
            for row, joint in enumerate(ranked):
                cues = [cue for cue, _ in active_feedback[joint]]
                label = LIMB_JOINTS[joint] if cues == ["adjust direction"] else joint.capitalize()
                message = f"{label}: {' / '.join(cues)}"
                cv2.putText(preview, message, (370, 540 + row * 20), font, 0.55, CORRECTION_COLOR, 1, cv2.LINE_AA)
            writer.write(preview)

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
    points_by_video = []
    for video, frames in ((args.reference, reference), (args.learner, learner)):
        with closing(read_frames(video)) as source_frames:
            _, image = next(source_frames)
            size = image.shape[1::-1]
        points_by_video.append([visible_image_points(frame.get("landmarks", []), size) for frame in frames])

    reference_points, learner_points = points_by_video
    learner_poses = [(frame["timestamp_ms"], points) for frame, points in zip(learner, learner_points)]
    feedback = {
        joint: find_joint_feedback(reference, learner, path, joint)
        + find_direction_feedback(reference_points, learner_poses, path, joint)
        for joint in LIMB_JOINTS
    }
    save_alignment_preview(args.reference, args.learner, reference, learner, path, args.output, feedback=feedback)
    print(f"Saved {args.output} (one DTW pair per frame, 30 FPS, no audio)")

    for joint, intervals in feedback.items():
        for start, end, cue, gap in sorted(intervals):
            print(f"{LIMB_JOINTS[joint]} {start / 1000:.2f}–{end / 1000:.2f}s: {cue} ({gap:.1f} degrees)")


if __name__ == "__main__":
    main()
