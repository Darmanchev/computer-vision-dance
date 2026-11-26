import json
from contextlib import closing
from pathlib import Path

from danceai.pose import create_landmarker, detect_pose
from danceai.processing import joint_angles
from danceai.video import read_frames

MODEL_PATH = "assets/models/pose_landmarker_full.task"
VIDEO_PATH1 = "data/dance2.mp4"
VIDEO_PATH2 = "data/my-video.MOV"


def main() -> None:
    extract_measurements(video_path=VIDEO_PATH1, model_path=MODEL_PATH)
    extract_measurements(video_path=VIDEO_PATH2, model_path=MODEL_PATH)


def extract_measurements(video_path: str | Path, model_path: str | Path) -> list[dict]:
    measurements = []
    video_path = Path(video_path)
    model_path = Path(model_path)

    if not model_path.is_file():
        raise FileNotFoundError(f"Pose model not found: {model_path}")

    with create_landmarker(model_path) as model:
        with closing(read_frames(video_path)) as frames:
            for timestamp_ms, frame in frames:
                result = detect_pose(model, frame, timestamp_ms)
                if result.pose_world_landmarks:
                    angles = joint_angles(result.pose_world_landmarks[0])
                    landmarks = [
                        {"x": point.x, "y": point.y, "visibility": point.visibility}
                        for point in result.pose_landmarks[0]
                    ]

                    measurements.append(
                        {
                            "timestamp_ms": timestamp_ms,
                            "angles": angles,
                            "landmarks": landmarks,
                        }
                    )

    output_path = video_path.with_name(f"{video_path.stem}-measurement.json")

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(measurements, file, indent=2)

    return measurements

def load_measurements(json_path: str | Path) -> list[dict]:
    with Path(json_path).open("r", encoding="utf-8") as file:
        return json.load(file)

    
if __name__ == "__main__":
    main()
