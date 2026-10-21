import os
import tempfile
import unittest

import cv2
import numpy as np

from src.video import sample_frames


class VideoTests(unittest.TestCase):
    def test_samples_every_n_seconds(self):
        path = os.path.join(tempfile.mkdtemp(), "t.avi")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
        if not writer.isOpened():
            self.skipTest("No video writer available in this environment")
        for i in range(60):  # 6 seconds at 10 fps
            writer.write(np.full((48, 64, 3), i * 4 % 255, dtype=np.uint8))
        writer.release()
        frames = list(sample_frames(path, every_sec=2.0))
        self.assertEqual(len(frames), 3)
        self.assertEqual(frames[0][1].shape, (48, 64, 3))
        self.assertAlmostEqual(frames[1][0], 2.0, places=1)

    def test_max_frames(self):
        path = os.path.join(tempfile.mkdtemp(), "t.avi")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
        if not writer.isOpened():
            self.skipTest("No video writer available in this environment")
        for _ in range(60):
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
        writer.release()
        self.assertEqual(len(list(sample_frames(path, every_sec=1.0, max_frames=2))), 2)


if __name__ == "__main__":
    unittest.main()
