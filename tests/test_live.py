import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.ingest import grab_frame, load_cameras, run_once
from src.live import LiveStore, build_record
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

    def rec(self, zone, count):
        return build_record(self.img, zone, 100.0, DEFAULT_THRESHOLDS, name="x", predict_fn=fake_predict(count))

    def test_round_trip_and_latest_per_zone(self):
        self.store.add(self.rec("A", 50))
        self.store.add(self.rec("B", 300))
        self.store.add(self.rec("A", 120))
        rows, _, latest = self.store.snapshot()
        self.assertEqual(len(rows), 3)
        self.assertEqual(set(latest), {"A", "B"})
        self.assertEqual(latest["A"]["count"], 120)
        self.assertEqual(latest["A"]["level"], "Medium")
        self.assertEqual(latest["A"]["heat"].shape, self.img.shape)

    def test_alert_only_on_level_change(self):
        self.store.add(self.rec("A", 50))
        self.store.add(self.rec("A", 60))
        self.assertEqual(self.store.snapshot()[1], [])
        self.store.add(self.rec("A", 300))
        alerts = self.store.snapshot()[1]
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["Change"], "Low → High")

    def test_clear(self):
        self.store.add(self.rec("A", 50))
        self.store.clear()
        self.assertEqual(self.store.snapshot()[2], {})

    def test_two_store_objects_share_state(self):
        self.store.add(self.rec("A", 50))
        other = LiveStore(self.tmp.name)
        self.assertIn("A", other.snapshot()[2])


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
            self.assertEqual(store.snapshot()[2]["Good"]["count"], 10)
            self.assertTrue(any("Bad: skipped" in m for m in logs))

    def test_config_validation(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.yaml"
            p.write_text("cameras:\n  - zone: A\n    source: x\n")
            with self.assertRaises(ValueError):
                load_cameras(p)
        self.assertEqual(len(load_cameras("config/cameras.yaml")), 3)


if __name__ == "__main__":
    unittest.main()
