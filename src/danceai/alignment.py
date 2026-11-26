import numpy as np
from tslearn.metrics import dtw_subsequence_path

from danceai.pipeline import load_measurements
from danceai.processing import JOINTS


def prepare_measurements(measurements):
    frames = [frame for frame in measurements if all(frame["angles"][joint] is not None for joint in JOINTS)]

    if not frames:
        raise ValueError("No frames with all required angles")

    values = np.array(
        [[frame["angles"][joint] for joint in JOINTS] for frame in frames],
        dtype=float,
    )

    if not np.isfinite(values).all():
        raise ValueError("Angles contain invalid numbers")

    return frames, values


def align_measurements(reference: list[dict], learner: list[dict]):
    reference, reference_values = prepare_measurements(reference)
    learner, learner_values = prepare_measurements(learner)
    path, distance = dtw_subsequence_path(reference_values, learner_values)
    return reference, learner, path, float(distance)


def main():
    reference, learner, path, distance = align_measurements(
        load_measurements("data/dance2-measurement.json"),
        load_measurements("data/my-video-measurement.json"),
    )
    print(f"DTW distance: {distance:.2f}")
    for i, j in path[:: max(1, len(path) // 10)]:
        print(f"Reference: {reference[i]['timestamp_ms'] / 1000:.2f}s -> Learner: {learner[j]['timestamp_ms'] / 1000:.2f}s")


if __name__ == "__main__":
    main()
