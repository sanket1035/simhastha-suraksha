import unittest

import numpy as np

from src.heatmap import overlay_heatmap


class HeatmapTests(unittest.TestCase):
    def setUp(self):
        self.img = np.full((80, 120, 3), 100, dtype=np.uint8)

    def test_shape_and_dtype_preserved(self):
        out = overlay_heatmap(self.img, np.random.rand(10, 15))
        self.assertEqual(out.shape, self.img.shape)
        self.assertEqual(out.dtype, np.uint8)

    def test_zero_density_returns_unchanged_copy(self):
        out = overlay_heatmap(self.img, np.zeros((10, 15)))
        self.assertTrue(np.array_equal(out, self.img))
        self.assertIsNot(out, self.img)

    def test_hot_spot_tinted_far_area_untouched(self):
        dens = np.zeros((10, 15), dtype=np.float32)
        dens[5, 12] = 1.0
        out = overlay_heatmap(self.img, dens, min_fraction=0.2)
        self.assertFalse(np.array_equal(out[44:46, 96:98], self.img[44:46, 96:98]))
        self.assertTrue(np.array_equal(out[:10, :10], self.img[:10, :10]))


if __name__ == "__main__":
    unittest.main()
