import numpy as np
from tslearn.metrics import dtw_subsequence_path

from danceai.pipeline import load_measurements
from danceai.processing import JOINTS


def prepare_measurements(measurements):
    frames = [
        frame
        for frame in measurements
        if all(frame["angles"][joint] is not None for joint in JOINTS)
    ]

    if not frames:
        raise ValueError("No frames with all required angles")