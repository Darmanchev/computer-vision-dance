import math

from danceai.alignment import align_measurements
from danceai.pipeline import load_measurements
from danceai.processing import JOINTS


REFERENCE_PATH = "data/dance2-measurement.json"
LEARNER_PATH = "data/my-video-measurement.json"

def compare_angles(reference, learner, path):
    if not path:
        raise ValueError("Alignment path is empty")

    errors = {}


    for joint in JOINTS:
        total_error = 0.0

        for i, j in path:
            reference_angle = reference[i]["angles"][joint]
            learner_angle = learner[j]["angles"][joint]
            total_error += abs(reference_angle - learner_angle)

        errors[joint] = total_error / len(path)

    return errors


def main() -> None:
    reference = load_measurements(REFERENCE_PATH)
    learner = load_measurements(LEARNER_PATH)

    reference, learner, path, _ = align_measurements(reference, learner)

    for joint, error in compare_angles(reference, learner, path).items():
        print(f"{joint}: {error:.2f} degrees mean error")



if __name__ == "__main__":
    main()
