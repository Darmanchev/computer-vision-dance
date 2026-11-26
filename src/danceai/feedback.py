from itertools import groupby
from statistics import mean

from danceai.processing import limb_directions


def joints_with_direction_gap(reference, learner, joints):
    """Yield limbs with a visible segment over the 2D direction tolerance."""
    for joint in joints:
        directions = zip(limb_directions(reference, joint), limb_directions(learner, joint))
        if any(abs((actual - expected + 180) % 360 - 180) > 30
               for expected, actual in directions if expected is not None and actual is not None):
            yield joint


def find_joint_feedback(reference, learner, path, joint):
    """Return bend/straighten corrections with start and end times in the learner video."""
    intervals = []
    for learner_index, pairs in groupby(path, key=lambda pair: pair[1]):
        frame = learner[learner_index]
        timestamp = frame["timestamp_ms"]
        angle = frame["angles"][joint]
        difference = mean(angle - reference[i]["angles"][joint] for i, _ in pairs)

        direction = None
        if difference > 15:
            direction = "bend more"
        elif difference < -15:
            direction = "straighten more"

        if intervals:
            start, end, previous_direction = intervals[-1]
            if direction == previous_direction and timestamp - end <= 200:
                intervals[-1] = (start, timestamp, direction)
                continue
        intervals.append((timestamp, timestamp, direction))

    return [(start, end, direction) for start, end, direction in intervals if direction and end - start >= 300]
