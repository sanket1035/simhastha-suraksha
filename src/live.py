"""Shared live state: one folder that both the dashboard and the camera-ingest worker use.

Each processed photo becomes one record (numbers in records.jsonl, images as JPEGs).
Because the state lives on disk, a separate ingest process and the Streamlit app see
the same data. Point both at the same folder with SS_LIVE_DIR.
"""
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .heatmap import colorize_density, overlay_heatmap
from .predict import predict
from .risk import density_from_count, level_for_density, load_thresholds

MAX_RECORDS = 300
IMAGE_KINDS = ("original", "overlay", "heat")


def _heat_lut():
    """Aubergine -> marigold -> pale gold colour ramp (BGR, 256 entries) for OpenCV."""
    stops = [(0.0, (0x34, 0x0F, 0x3A)), (0.45, (0x9A, 0x2F, 0x7A)), (0.75, (0x00, 0xA0, 0xEA)),
             (1.0, (0xC8, 0xF0, 0xFF))]
    xs = np.linspace(0, 1, 256)
    bgr = np.stack([np.interp(xs, [p for p, _ in stops], [c[i] for _, c in stops]) for i in range(3)], axis=1)
    return bgr.astype(np.uint8).reshape(256, 1, 3)


HEAT_LUT = _heat_lut()


def _shrink(img, max_w=900):
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    return cv2.resize(img, (max_w, int(h * max_w / w)), interpolation=cv2.INTER_AREA)


def build_record(img_rgb, zone, area, thresholds, name="", predict_fn=None, ts=None, camera="Camera 1"):
    """Run the model on one RGB frame and return a record (with image arrays)."""
    count, dens_map = (predict_fn or predict)(img_rgb)
    density = density_from_count(count, area)
    small = _shrink(img_rgb)
    return {
        "zone": zone, "camera": camera, "name": name, "ts": ts or datetime.now(), "area": float(area),
        "count": float(count), "density": density, "level": level_for_density(density, thresholds),
        "original": small,
        "overlay": overlay_heatmap(small, dens_map, colormap=HEAT_LUT, alpha=0.8, min_fraction=0.0,
                                   percentile=99, gamma=0.7),
        "heat": colorize_density(dens_map, (small.shape[1], small.shape[0]), colormap=HEAT_LUT,
                                 percentile=99, gamma=0.7),
    }


def default_dir():
    return Path(os.environ.get("SS_LIVE_DIR") or Path(tempfile.gettempdir()) / "simhastha_live")


def _write_jpeg(path, rgb):
    tmp = path.with_suffix(".tmp.jpg")
    cv2.imwrite(str(tmp), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    os.replace(tmp, path)


def _read_jpeg(path):
    bgr = cv2.imread(str(path))
    return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _slug(text):
    return "".join(c if c.isalnum() else "-" for c in str(text).strip()).strip("-").lower() or "cam"


def _numbers(rec):
    return {"zone": rec["zone"], "camera": rec["camera"], "name": rec["name"],
            "ts": rec["ts"].isoformat(timespec="seconds"), "area": rec["area"], "count": rec["count"],
            "density": rec["density"], "level": rec["level"]}


def zone_summary(cams, thresholds=None):
    """Combine the latest photo of every camera angle of a zone.

    `cams` maps (zone, camera) -> record. Angles are assumed to look at different ground, so the
    zone total is the sum of people and the sum of areas; density = total people / total area.
    Returns zone -> {count, area, density, level, ts, cameras: [records, busiest first]}.
    """
    thresholds = thresholds or load_thresholds()
    zones = {}
    for (zone, _cam), rec in cams.items():
        zones.setdefault(zone, []).append(rec)
    out = {}
    for zone, recs in zones.items():
        count = sum(r["count"] for r in recs)
        area = sum(r["area"] for r in recs)
        density = density_from_count(count, area)
        out[zone] = {"zone": zone, "count": count, "area": area, "density": density,
                     "level": level_for_density(density, thresholds), "ts": max(r["ts"] for r in recs),
                     "cameras": sorted(recs, key=lambda r: -r["density"])}
    return out


class LiveStore:
    def __init__(self, root=None):
        self.root = Path(root) if root else default_dir()
        self.root.mkdir(parents=True, exist_ok=True)
        self._records = self.root / "records.jsonl"
        self._alerts = self.root / "alerts.jsonl"

    def _read_lines(self, path, limit):
        if not path.exists():
            return []
        out = []
        for line in path.read_text().splitlines()[-limit:]:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue  # half-written line from a concurrent writer
        return out

    def _append(self, path, obj):
        with open(path, "a") as f:  # one short line per write, appended atomically on POSIX
            f.write(json.dumps(obj) + "\n")

    def _img_path(self, zone, camera, kind):
        return self.root / ("%s__%s_%s.jpg" % (_slug(zone), _slug(camera), kind))

    def _latest_by_camera(self):
        latest = {}
        for r in self._read_lines(self._records, MAX_RECORDS):
            r.setdefault("camera", "Camera 1")
            latest[(r["zone"], r["camera"])] = r
        return latest

    def add(self, rec, thresholds=None):
        thresholds = thresholds or load_thresholds()
        rec.setdefault("camera", "Camera 1")
        self.root.mkdir(parents=True, exist_ok=True)  # the folder may have been cleaned up
        before = zone_summary(self._latest_by_camera(), thresholds).get(rec["zone"])
        for kind in IMAGE_KINDS:
            _write_jpeg(self._img_path(rec["zone"], rec["camera"], kind), rec[kind])
        self._append(self._records, _numbers(rec))
        after = zone_summary(self._latest_by_camera(), thresholds)[rec["zone"]]
        if before and before["level"] != after["level"]:
            self._append(self._alerts, {"Time": rec["ts"].strftime("%H:%M:%S"), "Zone": rec["zone"],
                                        "Change": "%s → %s" % (before["level"], after["level"]),
                                        "People": round(after["count"])})
        lines = self._records.read_text().splitlines()
        if len(lines) > MAX_RECORDS * 2:
            self._records.write_text("\n".join(lines[-MAX_RECORDS:]) + "\n")

    def snapshot(self):
        """Return (records, alerts, cams). `cams` maps (zone, camera) -> latest record with image arrays."""
        rows = self._read_lines(self._records, MAX_RECORDS)
        for r in rows:
            r["ts"] = datetime.fromisoformat(r["ts"])
            r.setdefault("camera", "Camera 1")
        cams = {}
        for r in rows:
            cams[(r["zone"], r["camera"])] = r
        for (zone, cam), r in cams.items():
            for kind in IMAGE_KINDS:
                r[kind] = _read_jpeg(self._img_path(zone, cam, kind))
        return rows, self._read_lines(self._alerts, 100), cams

    def clear(self):
        for p in self.root.glob("*"):
            if p.suffix in (".jpg", ".jsonl"):
                p.unlink()
