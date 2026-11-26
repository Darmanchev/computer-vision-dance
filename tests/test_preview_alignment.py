import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from preview_alignment import save_alignment_preview


class PreviewAlignmentTest(unittest.TestCase):
    def test_writes_one_output_frame_per_pair(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "input.mp4"
            output = Path(folder) / "output.mp4"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
            self.assertTrue(writer.isOpened())
            for value in (0, 80, 160):
                writer.write(np.full((48, 64, 3), value, np.uint8))
            writer.release()
            measurements = [{"timestamp_ms": ms} for ms in (0, 100, 200)]

            save_alignment_preview(source, source, measurements, measurements,
                                   [(0, 0), (1, 0), (2, 2)], output)

            capture = cv2.VideoCapture(str(output))
            try:
                decoded = 0
                while capture.read()[0]:
                    decoded += 1
                self.assertEqual(decoded, 3)
            finally:
                capture.release()

    def test_rejects_output_that_would_replace_input(self):
        with self.assertRaises(ValueError):
            save_alignment_preview("same.mp4", "other.mp4", [], [], [(0, 0)], "same.mp4")
