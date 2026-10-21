import unittest

import numpy as np

from src.pipeline import analyze_frames
from src.risk import RiskSmoother


class PipelineTests(unittest.TestCase):
    def test_alert_raised_when_level_changes(self):
        counts = iter([50, 50, 500, 500, 500])  # area 100 m2: Low, Low, then Critical
        frames = [(float(i), np.zeros((16, 16, 3), dtype=np.uint8)) for i in range(5)]

        def fake_predict(_frame):
            return next(counts), np.zeros((2, 2), dtype=np.float32)

        smoother = RiskSmoother(window=1, confirm_up=2)
        result = analyze_frames(frames, 100.0, predict_fn=fake_predict, smoother=smoother)
        self.assertEqual([h["level"] for h in result["history"]], ["Low", "Low", "Low", "Critical", "Critical"])
        self.assertEqual(len(result["alerts"]), 1)
        self.assertEqual(result["alerts"][0]["from"], "Low")
        self.assertEqual(result["alerts"][0]["to"], "Critical")
        self.assertEqual(result["alerts"][0]["time_s"], 3.0)

    def test_empty_input(self):
        result = analyze_frames([], 100.0)
        self.assertEqual(result["history"], [])
        self.assertIsNone(result["last_frame"])

    def test_runs_end_to_end_with_stub_backend(self):
        frames = [(0.0, (np.random.default_rng(0).random((64, 96, 3)) * 255).astype(np.uint8))]
        result = analyze_frames(frames, 200.0)
        self.assertEqual(len(result["history"]), 1)
        self.assertIn(result["history"][0]["level"], ["Low", "Medium", "High", "Critical"])


if __name__ == "__main__":
    unittest.main()
