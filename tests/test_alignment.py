import unittest

from danceai.alignment import align_measurements
from danceai.processing import JOINTS


class AlignmentTest(unittest.TestCase):
    def test_finds_delayed_copy_and_preserves_filtered_timestamps(self):
        reference = [
            {"timestamp_ms": i * 100, "angles": dict.fromkeys(JOINTS, value)}
            for i, value in enumerate((170, 140, 90, 150))
        ]
        learner = [{"timestamp_ms": 0, "angles": dict.fromkeys(JOINTS, None)}]
        learner += [
            {"timestamp_ms": (i + 1) * 100, "angles": dict.fromkeys(JOINTS, value)}
            for i, value in enumerate((40, 170, 140, 90, 150, 30))
        ]

        ref, attempt, path, distance = align_measurements(reference, learner)

        self.assertAlmostEqual(distance, 0)
        self.assertEqual(path[0], (0, 1))
        self.assertEqual(attempt[path[0][1]]["timestamp_ms"], 200)
        self.assertEqual(ref[-1]["timestamp_ms"], 300)
