"""Camera-ingest worker: poll each camera, run the model, update the shared live store.

Run next to the dashboard (same SS_LIVE_DIR):
    python -m src.ingest --config config/cameras.yaml
    python -m src.ingest --config config/cameras.yaml --once     # one pass, then exit

A camera `source` can be an http(s) snapshot URL (JPEG), an RTSP/HTTP stream URL, a video file,
or a folder of images (newest image is used). Failures are logged and retried next round,
so one dead camera never stops the others.
"""
import argparse
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .live import LiveStore, build_record
from .risk import load_thresholds

IMG_EXT = {".jpg", ".jpeg", ".png"}


def load_cameras(path):
    import yaml
    with open(path) as f:
        cams = (yaml.safe_load(f) or {}).get("cameras", [])
    for c in cams:
        if not {"zone", "source", "area_m2"} <= set(c):
            raise ValueError("camera entry needs zone, source, area_m2: %r" % (c,))
    return cams


def grab_frame(source, timeout=10):
    """Return one RGB frame (HxWx3 uint8) from a snapshot URL, stream, video file or image folder."""
    if source.startswith(("http://", "https://")) and Path(source.split("?")[0]).suffix.lower() in IMG_EXT | {""}:
        try:
            data = urllib.request.urlopen(source, timeout=timeout).read()
            bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if bgr is not None:
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            pass  # fall through: it may be a video stream URL
    p = Path(source)
    if p.is_dir():
        imgs = sorted((f for f in p.iterdir() if f.suffix.lower() in IMG_EXT), key=lambda f: f.stat().st_mtime)
        if not imgs:
            raise RuntimeError("no images in %s" % source)
        source = str(imgs[-1])
    if Path(source).suffix.lower() in IMG_EXT and Path(source).exists():
        bgr = cv2.imread(source)
    else:
        cap = cv2.VideoCapture(source)
        ok, bgr = cap.read()
        cap.release()
        if not ok:
            bgr = None
    if bgr is None:
        raise RuntimeError("could not read a frame from %s" % source)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def run_once(cameras, store, thresholds, grab=grab_frame, log=print, predict_fn=None):
    done = 0
    for c in cameras:
        try:
            rec = build_record(grab(c["source"]), c["zone"], c["area_m2"], thresholds, name=c["source"],
                               predict_fn=predict_fn)
            store.add(rec)
            done += 1
            log("%s: %d people, %s" % (c["zone"], round(rec["count"]), rec["level"]))
        except Exception as exc:
            log("%s: skipped (%s)" % (c["zone"], exc))
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/cameras.yaml")
    ap.add_argument("--interval", type=float, default=10.0, help="seconds between rounds")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--live-dir", default="")
    args = ap.parse_args()
    from .predict import ensure_model
    ensure_model()
    cameras, thresholds = load_cameras(args.config), load_thresholds()
    store = LiveStore(args.live_dir or None)
    print("ingest -> %s, %d camera(s)" % (store.root, len(cameras)))
    while True:
        run_once(cameras, store, thresholds)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
