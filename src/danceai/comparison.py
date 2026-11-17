from danceai.pipeline import load_measurements

MODEL_PATH = "assets/models/pose_landmarker_full.task"
VIDEO_PATH1 = "data/dance2.mp4"
VIDEO_PATH2 = "data/my-video.MOV"
REFERENCE_PATH = "data/dance2-measurement.json"
LEARNER_PATH = "data/my-video-measurement.json"


def main() -> None:
    reference = load_measurements(json_path=REFERENCE_PATH)
    learner = load_measurements(json_path=LEARNER_PATH)
    print(compare_angles(reference, learner))


def compare_angles(reference: list[dict], learner: list[dict]) -> float:
    reference_angle = reference[0]["angles"]["left elbow"]
    learner_angle = learner[0]["angles"]["left elbow"]

    return abs(reference_angle - learner_angle)


if __name__ == "__main__":
    main()
