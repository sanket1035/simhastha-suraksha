import csv
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import savemat

from src.data_prep import find_gt_dir, find_part_dir, load_points, process_split


def _write_gt(path, pts):
    info = np.empty((1, 1), dtype=object)
    inner = np.empty((1, 1), dtype=[("location", object), ("number", object)])
    inner[0, 0]["location"] = pts
    inner[0, 0]["number"] = np.array([[len(pts)]])
    info[0, 0] = inner
    savemat(str(path), {"image_info": info})


class DataPrepTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        part = self.root / "part_A_final" / "train_data"
        (part / "images").mkdir(parents=True)
        (part / "ground_truth").mkdir()
        self.pts = np.array([[20.0, 30.0], [60.0, 40.0], [100.0, 90.0], [150.0, 20.0]])
        Image.fromarray(np.zeros((120, 200, 3), dtype=np.uint8)).save(part / "images" / "IMG_1.jpg")
        _write_gt(part / "ground_truth" / "GT_IMG_1.mat", self.pts)

    def test_load_points(self):
        pts = load_points(self.root / "part_A_final" / "train_data" / "ground_truth" / "GT_IMG_1.mat")
        self.assertTrue(np.allclose(pts, self.pts))

    def test_find_part_dir(self):
        self.assertEqual(find_part_dir(self.root, "A").name, "part_A_final")
        with self.assertRaises(FileNotFoundError):
            find_part_dir(self.root, "B")

    def test_process_split_writes_density_and_index(self):
        out = Path(tempfile.mkdtemp())
        rows = process_split(find_part_dir(self.root, "A"), "train", out, adaptive=True)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["gt_count"], 4)
        dens = np.load(rows[0]["density_path"])
        self.assertEqual(dens.shape, (120, 200))
        self.assertAlmostEqual(float(dens.sum()), 4.0, places=2)
        with open(out / "train_index.csv") as f:
            index = list(csv.DictReader(f))
        self.assertEqual(index[0]["gt_count"], "4")

    def test_hyphenated_ground_truth_folder(self):
        part = self.root / "part_A_final" / "train_data"
        (part / "ground_truth").rename(part / "ground-truth")
        self.assertEqual(find_gt_dir(part).name, "ground-truth")
        out = Path(tempfile.mkdtemp())
        rows = process_split(find_part_dir(self.root, "A"), "train", out, adaptive=True)
        self.assertEqual(rows[0]["gt_count"], 4)


if __name__ == "__main__":
    unittest.main()
