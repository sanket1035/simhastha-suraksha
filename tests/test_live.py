import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.ingest import grab_frame, load_cameras, run_once
from src.live import LiveStore, build_record, zone_summary
from src.risk import DEFAULT_THRESHOLDS


def fake_predict(count):
    def fn(img):
        return float(count), np.ones((4, 4), dtype=np.float32) * count / 16.0
    return fn


class LiveStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = LiveStore(self.tmp.name)
        self.img = np.full((64, 96, 3), 120, dtype=np.uint8)

    def tearDown(self):
        self.tmp.cleanup()

    def rec(self, zone, count, camera="Camera 1", area=100.0):
        return build_record(self.img, zone, area, DEFAULT_THRESHOLDS, name="x",
                            predict_fn=fake_predict(count), camera=camera)

    def add(self, *a, **k):
        self.store.add(self.rec(*a, **k), DEFAULT_THRESHOLDS)

    def test_round_trip_and_latest_per_camera(self):
        self.add("A", 50)
        self.add("B", 300)
        self.add("A", 120)
        rows, _, cams = self.store.snapshot()
        self.assertEqual(len(rows), 3)
        self.assertEqual(set(cams), {("A", "Camera 1"), ("B", "Camera 1")})
        self.assertEqual(cams[("A", "Camera 1")]["count"], 120)
        self.assertEqual(cams[("A", "Camera 1")]["heat"].shape, self.img.shape)

    def test_angles_of_one_zone_keep_their_own_area_and_are_combined(self):
        self.add("A", 10, camera="Steps", area=100.0)
        self.add("A", 300, camera="Bridge", area=50.0)
        cams = self.store.snapshot()[2]
        self.assertEqual(len(cams), 2)
        z = zone_summary(cams, DEFAULT_THRESHOLDS)["A"]
        self.assertEqual(z["count"], 310)
        self.assertEqual(z["area"], 150.0)
        self.assertAlmostEqual(z["density"], 310 / 150.0)
        self.assertEqual(z["level"], "Medium")  # 2.07 per m2; the busy angle alone would be Critical
        self.assertEqual(z["cameras"][0]["camera"], "Bridge")  # busiest first

    def test_ignore_top_removes_counts_from_masked_rows(self):
        def fp(img):  # 8x8 density map: all mass in the top half, 4 in the bottom
            d = np.zeros((8, 8), dtype=np.float32)
            d[:4] = 100.0 / 32
            d[4:] = 4.0 / 32
            return float(d.sum()), d
        full = build_record(self.img, "A", 100.0, DEFAULT_THRESHOLDS, predict_fn=fp)
        masked = build_record(self.img, "A", 100.0, DEFAULT_THRESHOLDS, predict_fn=fp, ignore_top=0.5)
        self.assertAlmostEqual(full["count"], 104.0, places=3)
        self.assertAlmostEqual(masked["count"], 4.0, places=3)
        self.assertEqual(masked["ignore_top"], 0.5)
        self.assertEqual(masked["overlay"].shape, self.img.shape)

    def test_new_photo_replaces_only_its_own_angle(self):
        self.add("A", 10, camera="Steps")
        self.add("A", 20, camera="Bridge")
        self.add("A", 30, camera="Steps")
        z = zone_summary(self.store.snapshot()[2], DEFAULT_THRESHOLDS)["A"]
        self.assertEqual(z["count"], 50)

    def test_alert_only_when_zone_level_changes(self):
        self.add("A", 50)
        self.add("A", 60)
        self.assertEqual(self.store.snapshot()[1], [])
        self.add("A", 300)
        alerts = self.store.snapshot()[1]
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["Change"], "Low → High")

    def test_clear(self):
        self.add("A", 50)
        self.store.clear()
        self.assertEqual(self.store.snapshot()[2], {})

    def test_two_store_objects_share_state(self):
        self.add("A", 50)
        other = LiveStore(self.tmp.name)
        self.assertIn(("A", "Camera 1"), other.snapshot()[2])


class IngestTests(unittest.TestCase):
    def test_folder_source_uses_newest_image_and_failures_are_isolated(self):
        with tempfile.TemporaryDirectory() as d:
            cv2.imwrite(str(Path(d) / "a.jpg"), np.full((40, 60, 3), 90, dtype=np.uint8))
            frame = grab_frame(d)
            self.assertEqual(frame.shape, (40, 60, 3))
            store = LiveStore(Path(d) / "live")
            cams = [{"zone": "Bad", "source": "/nonexistent/x.mp4", "area_m2": 100},
                    {"zone": "Good", "source": d, "area_m2": 100}]
            logs = []
            done = run_once(cams, store, DEFAULT_THRESHOLDS, log=logs.append, predict_fn=fake_predict(10))
            self.assertEqual(done, 1)
            self.assertEqual(zone_summary(store.snapshot()[2], DEFAULT_THRESHOLDS)["Good"]["count"], 10)
            self.assertTrue(any("Bad: skipped" in m for m in logs))

    def test_config_validation(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.yaml"
            p.write_text("cameras:\n  - zone: A\n    source: x\n")
            with self.assertRaises(ValueError):
                load_cameras(p)
        self.assertEqual(len(load_cameras("config/cameras.yaml")), 4)


if __name__ == "__main__":
    unittest.main()
