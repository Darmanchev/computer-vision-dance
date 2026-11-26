from itertools import groupby
from statistics import mean, median

from danceai.processing import limb_directions


def sustained_intervals(samples):
    intervals = []
    for timestamp, cue, gap in samples:
        if intervals:
            start, end, previous_cue, gaps = intervals[-1]
            if cue == previous_cue and timestamp - end <= 200:
                gaps.append(gap)
                intervals[-1] = (start, timestamp, cue, gaps)
                continue
        intervals.append((timestamp, timestamp, cue, [gap]))
    return [(start, end, cue, median(gaps)) for start, end, cue, gaps in intervals if cue and end - start >= 300]


def find_direction_feedback(reference_points, learner_poses, path, joint):
    reference_directions = [limb_directions(points, joint) for points in reference_points]
    samples = []
    for learner_index, pairs in groupby(path, key=lambda pair: pair[1]):
        timestamp, points = learner_poses[learner_index]
        actual = limb_directions(points, joint)
        pair_gaps = []
        for i, _ in pairs:
            gaps = [
                abs((now - expected + 180) % 360 - 180)
                for expected, now in zip(reference_directions[i], actual)
                if expected is not None and now is not None
            ]
            if gaps:
                pair_gaps.append(max(gaps))
        gap = median(pair_gaps) if pair_gaps else 0
        samples.append((timestamp, "adjust direction" if gap > 30 else None, gap))
    return sustained_intervals(samples)


def find_joint_feedback(reference, learner, path, joint):
    """Return bend/straighten corrections with learner times and median angle gaps."""
    samples = []
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

        samples.append((timestamp, direction, abs(difference)))
    return sustained_intervals(samples)
