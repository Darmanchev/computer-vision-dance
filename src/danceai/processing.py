import numpy as np

JOINTS = {
    "left elbow": (11, 13, 15),
    "right elbow": (12, 14, 16),
    "left knee": (23, 25, 27),
    "right knee": (24, 26, 28),
    "left shoulder": (13, 11, 23),
    "right shoulder": (14, 12, 24),
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


def visible_image_points(landmarks, image_size):
    return {
        index: np.array([point["x"], point["y"]]) * image_size
        for index, point in enumerate(landmarks)
        if (point.get("visibility") or 0) >= 0.5 and np.isfinite((point["x"], point["y"])).all()
    }


def limb_directions(points, joint):
    """Return the two segment directions relative to the torso in the image plane."""
    if not all(index in points for index in (11, 12, 23, 24)):
        return None, None

    torso = points[11] + points[12] - points[23] - points[24]
    if not torso.any():
        return None, None

    directions = []
    start, middle, end = JOINTS[joint]
    for first, last in ((start, middle), (middle, end)):
        angle = None
        if first in points and last in points:
            segment = points[last] - points[first]
            if segment.any():
                cross = torso[0] * segment[1] - torso[1] * segment[0]
                angle = float(np.degrees(np.arctan2(cross, torso @ segment)))
        directions.append(angle)
    return tuple(directions)
