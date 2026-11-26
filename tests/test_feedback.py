import unittest

import numpy as np

from danceai.feedback import find_joint_feedback, joints_with_direction_gap


class FeedbackTest(unittest.TestCase):
    def test_detects_direction_change_in_one_segment(self):
        reference = {
            index: np.array(point)
            for index, point in {11: (0, 0), 12: (2, 0), 23: (0, 2), 24: (2, 2),
                                 26: (2, 4), 28: (2, 6)}.items()
        }
        learner = {**reference, 28: np.array([4, 4])}
        self.assertEqual(list(joints_with_direction_gap(reference, learner, ("right knee", "left knee"))), ["right knee"])
        self.assertEqual(list(joints_with_direction_gap(reference, reference, ("right knee",))), [])

    def test_groups_sustained_errors_without_counting_repeated_frames(self):
        reference = [{"angles": {"right knee": angle}} for angle in (90, 60, 120, 90)]
        learner = [
            {"timestamp_ms": time, "angles": {"right knee": angle}}
            for time, angle in (
                (0, 115), (100, 115), (300, 115), (400, 90),
                (500, 60), (600, 60), (800, 60), (1200, 60), (1300, 105),
            )
        ]
        path = [
            (0, 0), (0, 1), (0, 2), (1, 3), (2, 3),
            (3, 4), (3, 5), (3, 6), (3, 7), (3, 8),
        ]

        self.assertEqual(find_joint_feedback(reference, learner, path, "right knee"), [
            (0, 300, "bend more"),
            (500, 800, "straighten more"),
        ])
