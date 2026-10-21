import unittest

import numpy as np

from src.density import downsample_density_sum, make_density_map


class DensityTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.pts = np.column_stack([rng.uniform(0, 200, 60), rng.uniform(0, 160, 60)])

    def test_sum_equals_number_of_heads_adaptive(self):
        d = make_density_map((160, 200), self.pts, adaptive=True)
        self.assertAlmostEqual(float(d.sum()), 60.0, places=2)

    def test_sum_equals_number_of_heads_fixed(self):
        d = make_density_map((160, 200), self.pts, adaptive=False, fixed_sigma=8)
        self.assertAlmostEqual(float(d.sum()), 60.0, places=2)

    def test_points_on_border_keep_count(self):
        pts = np.array([[0, 0], [199, 159], [0, 159], [199, 0]])
        d = make_density_map((160, 200), pts, adaptive=False, fixed_sigma=10)
        self.assertAlmostEqual(float(d.sum()), 4.0, places=3)

    def test_out_of_bounds_points_dropped(self):
        pts = np.array([[10, 10], [500, 10], [-5, 20], [30, 400]])
        d = make_density_map((160, 200), pts, adaptive=False)
        self.assertAlmostEqual(float(d.sum()), 1.0, places=3)

    def test_empty_and_single_point(self):
        self.assertEqual(float(make_density_map((40, 40), np.empty((0, 2))).sum()), 0.0)
        d = make_density_map((40, 40), np.array([[20, 20]]), adaptive=True)
        self.assertAlmostEqual(float(d.sum()), 1.0, places=3)

    def test_peak_is_at_the_head(self):
        d = make_density_map((100, 100), np.array([[70, 30]]), adaptive=False, fixed_sigma=5)
        y, x = np.unravel_index(d.argmax(), d.shape)
        self.assertEqual((x, y), (70, 30))

    def test_downsample_preserves_count_and_shape(self):
        d = make_density_map((160, 200), self.pts, adaptive=True)
        small = downsample_density_sum(d, 8)
        self.assertEqual(small.shape, (20, 25))
        self.assertAlmostEqual(float(small.sum()), float(d.sum()), places=3)

    def test_downsample_crops_non_multiples(self):
        small = downsample_density_sum(np.ones((17, 33), dtype=np.float32), 8)
        self.assertEqual(small.shape, (2, 4))
        self.assertAlmostEqual(float(small.sum()), 16 * 32, places=3)


if __name__ == "__main__":
    unittest.main()
