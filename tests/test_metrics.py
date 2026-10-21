import unittest

from src.metrics import count_metrics, mae, rmse


class MetricTests(unittest.TestCase):
    def test_mae_rmse(self):
        pred, gt = [10, 20, 30], [12, 18, 33]
        self.assertAlmostEqual(mae(pred, gt), (2 + 2 + 3) / 3)
        self.assertAlmostEqual(rmse(pred, gt), ((4 + 4 + 9) / 3) ** 0.5)

    def test_perfect_prediction(self):
        m = count_metrics([5, 6], [5, 6])
        self.assertEqual(m["mae"], 0.0)
        self.assertEqual(m["rmse"], 0.0)


if __name__ == "__main__":
    unittest.main()
