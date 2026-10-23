import math
import unittest

import numpy as np

from src.geo import GRID_COLS, GRID_ROWS, coarse_grid, default_geo, footprint, map_cells
from src.risk import DEFAULT_THRESHOLDS


def metres(a, b):
    lat0 = (a[0] + b[0]) / 2
    return ((b[0] - a[0]) * 111320.0, (b[1] - a[1]) * 111320.0 * math.cos(math.radians(lat0)))


class GeoTests(unittest.TestCase):
    def test_coarse_grid_keeps_total_and_skips_ignored_rows(self):
        d = np.zeros((40, 60), dtype=np.float32)
        d[:10] = 5.0   # ignored band
        d[10:] = 0.01
        g = coarse_grid(d, ignore_rows=10)
        self.assertEqual(g.shape, (GRID_ROWS, GRID_COLS))
        self.assertAlmostEqual(float(g.sum()), 30 * 60 * 0.01, places=3)

    def test_footprint_is_a_rectangle_of_the_requested_area_and_width(self):
        geo = {"lat": 20.0077, "lon": 73.7921, "bearing": 0.0, "width_m": 20.0}
        c = footprint(geo, 300.0)
        w = math.hypot(*metres(c[0], c[1]))
        d = math.hypot(*metres(c[1], c[2]))
        self.assertAlmostEqual(w, 20.0, delta=0.1)
        self.assertAlmostEqual(d, 15.0, delta=0.1)

    def test_bearing_rotates_the_depth_axis(self):
        geo = {"lat": 20.0, "lon": 73.8, "bearing": 90.0, "width_m": 10.0}  # camera looks east
        c = footprint(geo, 200.0)  # depth 20 m runs east-west, width 10 m north-south
        dn, de = metres(c[1], c[2])
        self.assertAlmostEqual(abs(de), 20.0, delta=0.1)
        self.assertAlmostEqual(abs(dn), 0.0, delta=0.1)

    def test_cells_cover_grid_and_levels_follow_density(self):
        grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)
        grid[0, 0] = 10.0       # 280 cells of 100 m2 -> 0.357 m2 each: 10 people = 28 per m2
        cells = map_cells(grid, default_geo(20.0, 73.8, 100.0), 100.0, DEFAULT_THRESHOLDS)
        self.assertEqual(len(cells), GRID_ROWS * GRID_COLS)
        self.assertEqual(cells[0]["level"], "Critical")
        self.assertEqual(cells[1]["level"], "Low")
        self.assertEqual(len(cells[0]["poly"]), 4)

    def test_far_edge_is_the_top_row(self):
        geo = {"lat": 20.0, "lon": 73.8, "bearing": 0.0, "width_m": 10.0}
        grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)
        cells = map_cells(grid, geo, 100.0, DEFAULT_THRESHOLDS)
        top_lat = cells[0]["poly"][0][0]
        bottom_lat = cells[-1]["poly"][0][0]
        self.assertGreater(top_lat, bottom_lat)  # facing north: far (top of photo) is further north


if __name__ == "__main__":
    unittest.main()
