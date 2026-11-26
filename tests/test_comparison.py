import unittest

from danceai.comparison import compare_angles
from danceai.processing import JOINTS


class ComparisonTest(unittest.TestCase):
    def test_uses_aligned_pairs_and_all_joints(self):
        reference = [{"angles": dict.fromkeys(JOINTS, value)} for value in (10, 30)]
        learner = [{"angles": dict.fromkeys(JOINTS, value)} for value in (100, 20, 35)]
        learner[2]["angles"]["left shoulder"] = 45

        errors = compare_angles(reference, learner, [(0, 1), (1, 2)])

        self.assertEqual(errors["left elbow"], 7.5)
        self.assertEqual(errors["left shoulder"], 12.5)
        self.assertEqual(set(errors), set(JOINTS))

    def test_empty_path_is_rejected(self):
        with self.assertRaises(ValueError):
            compare_angles([], [], [])
