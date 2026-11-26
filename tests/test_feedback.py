import unittest

from danceai.feedback import find_joint_feedback


class FeedbackTest(unittest.TestCase):
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
