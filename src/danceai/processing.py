import numpy as np


JOINTS = {
    "left elbow": (11, 13, 15),
    "right elbow": (12, 14, 16),
    "left knee": (23, 25, 27),
    "right knee": (24, 26, 28),
}


def joint_angles(landmarks) -> dict[str, float | None]:
    """Measure angles in degrees from MediaPipe world landmarks."""
    angles = dict.fromkeys(JOINTS)
    for name, indices in JOINTS.items():
        points = [landmarks[index] for index in indices]
        if any(p.visibility is None or not p.visibility >= 0.5 for p in points):
            continue

        start, joint, end = np.array([(p.x, p.y, p.z) for p in points], dtype=float)
        a, b = start - joint, end - joint
        if not np.isfinite((a, b)).all() or not a.any() or not b.any():
            continue

        angles[name] = float(np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), a @ b)))
    return angles
