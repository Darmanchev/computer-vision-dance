import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from preview_alignment import save_alignment_preview


class PreviewAlignmentTest(unittest.TestCase):
    def test_writes_each_pair_and_highlights_only_the_limb_with_feedback(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "input.mp4"
            output = Path(folder) / "output.mp4"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
            self.assertTrue(writer.isOpened())
            for value in (0, 80, 160):
                writer.write(np.full((48, 64, 3), value, np.uint8))
            writer.release()
            reference = [{"timestamp_ms": ms} for ms in (0, 100, 200)]
            learner = [{"timestamp_ms": ms} for ms in (0, 100, 200)]
            reference_arm = [{}] * 16
            for index, (x, y) in ((11, (0.25, 0.25)), (12, (0.1, 0.25)), (13, (0.5, 0.25)), (15, (0.5, 0.5))):
                reference_arm[index] = {"x": x, "y": y, "visibility": 1.0}
            learner_pose = reference_arm.copy()
            learner_pose[11] = {"x": 0.5, "y": 0.5, "visibility": 1.0}
            learner_pose[12] = {"x": 0.35, "y": 0.5, "visibility": 1.0}
            for index in (1, 2):
                reference[index]["landmarks"] = reference_arm
            for index in (0, 2):
                learner[index]["landmarks"] = learner_pose

            save_alignment_preview(source, source, reference, learner,
                                   [(0, 0), (1, 0), (2, 2)], output,
                                   feedback={"left elbow": [(0, 100, "bend more")]})

            capture = cv2.VideoCapture(str(output))
            try:
                for expected_color in (None, "yellow", "blue"):
                    success, frame = capture.read()
                    self.assertTrue(success)
                    footer = frame[520:, 360:]
                    self.assertTrue(footer.size)
                    blue, green, red = cv2.split(footer)
                    shows_correction = np.any((blue < 100) & (green > 180) & (red > 180))
                    self.assertEqual(shows_correction, expected_color != "blue")
                    for x, y in ((580, 280), (630, 320)):
                        blue, green, red = frame[y, x]
                        self.assertEqual(blue < 100 and green > 180 and red > 180, expected_color == "yellow")
                        self.assertEqual(int(blue) - int(red) > 60, expected_color == "blue")
                    blue, _, red = frame[280, 510]
                    self.assertEqual(int(blue) - int(red) > 60, expected_color is not None)
                self.assertFalse(capture.read()[0])
            finally:
                capture.release()

    def test_rejects_output_that_would_replace_input(self):
        with self.assertRaises(ValueError):
            save_alignment_preview("same.mp4", "other.mp4", [], [], [(0, 0)], "same.mp4")
