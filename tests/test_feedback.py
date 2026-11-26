import unittest

import numpy as np

from danceai.feedback import find_direction_feedback, find_joint_feedback


class FeedbackTest(unittest.TestCase):
    def test_direction_uses_median_per_frame_and_requires_300_ms(self):
        reference = {
            index: np.array(point)
            for index, point in {11: (0, 0), 12: (2, 0), 23: (0, 2), 24: (2, 2),
                                 26: (2, 4), 28: (2, 6)}.items()
        }
        turned = {**reference, 28: np.array([4, 4])}
        learner = [(0, turned), (100, turned), (200, reference), (300, turned),
                   (400, turned), (500, turned), (600, turned), (700, turned), (800, reference)]
        path = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 3), (2, 4),
                (2, 5), (2, 6), (2, 7), (2, 8)]
        self.assertEqual(find_direction_feedback([reference, turned, reference], learner, path, "right knee"), [
            (300, 700, "adjust direction", 90.0),
        ])

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
            (0, 300, "bend more", 25.0),
            (500, 800, "straighten more", 30.0),
        ])
