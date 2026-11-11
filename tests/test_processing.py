import unittest
from types import SimpleNamespace

from danceai.processing import joint_angles


class JointAnglesTest(unittest.TestCase):
    def test_angle_and_unusable_landmarks(self):
        points = [SimpleNamespace(x=0, y=0, z=0, visibility=1) for _ in range(33)]
        points[11].x = 1
        points[15].z = 1

        angles = joint_angles(points)
        self.assertAlmostEqual(angles['left elbow'], 90)
        self.assertIsNone(angles['right elbow'])

        points[15].visibility = 0.1
        self.assertIsNone(joint_angles(points)['left elbow'])
