import unittest
from types import SimpleNamespace

from danceai.processing import joint_angles, limb_directions, visible_image_points


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

    def test_arm_direction_uses_image_aspect_ratio_and_ignores_arm_length(self):
        points = [{"x": 0.5, "y": 0.2, "visibility": 1.0} for _ in range(25)]
        points[23]["y"] = points[24]["y"] = 0.7
        points[13].update(x=0.6, y=0.4)
        points[15].update(x=0.6, y=0.3)
        upper_arm, forearm = limb_directions(visible_image_points(points, (200, 100)), "left elbow")
        self.assertAlmostEqual(upper_arm, 135)
        self.assertAlmostEqual(forearm, 0)

        points[13].update(x=0.55, y=0.3)
        self.assertAlmostEqual(limb_directions(visible_image_points(points, (200, 100)), "left elbow")[0], 135)

        points[13]["visibility"] = 0.1
        self.assertEqual(limb_directions(visible_image_points(points, (200, 100)), "left elbow"), (None, None))
