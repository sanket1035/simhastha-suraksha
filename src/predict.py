"""predict(image) -> (count, density_map): the single entry point the app uses.

Backends:
  onnx : trained CSRNet exported to ONNX (models/csrnet.onnx or $SS_MODEL_PATH)
  stub : deterministic fake output, so the dashboard can be built and demoed
         before a model exists. backend_name() tells you which one is active.

If the model file is missing and $SS_MODEL_URL is set, it is downloaded once
(e.g. a GitHub Release asset), which keeps big weights out of git.
"""
import hashlib
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

MAX_SIDE = 1024
FACTOR = 8
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_session = None
_session_key = None


def _model_path():
    default = Path(__file__).resolve().parent.parent / "models" / "csrnet.onnx"
    return Path(os.environ.get("SS_MODEL_PATH", default))


def ensure_model():
    """Return True if a model file is available (downloading it if a URL is set)."""
    path = _model_path()
    if path.exists():
        return True
    url = os.environ.get("SS_MODEL_URL", "")
    if url:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, str(path))
        except Exception as exc:  # network errors should not crash the app
            print("Could not download model: %s" % exc)
            if path.exists():
                path.unlink()
    return path.exists()


def _get_session():
    """Load the ONNX session once; return None when no model is available."""
    global _session, _session_key
    path = _model_path()
    if not path.exists():
        return None
    key = (str(path), path.stat().st_mtime)
    if _session is None or _session_key != key:
        import onnxruntime as ort
        _session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        _session_key = key
    return _session


def backend_name():
    return "onnx" if _model_path().exists() else "stub"


def to_rgb_array(image):
    """Accept a path, PIL image or HxWx3 uint8 RGB array; return an RGB uint8 array."""
    if isinstance(image, (str, Path)):
        bgr = cv2.imread(str(image))
        if bgr is None:
            raise ValueError("Could not read image: %s" % image)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("Expected an image with 3 channels, got shape %r" % (arr.shape,))
    return np.ascontiguousarray(arr[:, :, :3]).astype(np.uint8)


def preprocess(image_rgb, max_side=MAX_SIDE):
    """Resize (longest side <= max_side), crop to multiples of 8, normalise.

    Returns a float32 array of shape (1, 3, H, W).
    """
    h, w = image_rgb.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    if scale < 1.0:
        image_rgb = cv2.resize(image_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = image_rgb.shape[:2]
    h8, w8 = max(FACTOR, h // FACTOR * FACTOR), max(FACTOR, w // FACTOR * FACTOR)
    x = image_rgb[:h8, :w8].astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)


def _stub(image_rgb):
    """Deterministic fake density map (same image -> same output)."""
    seed = int.from_bytes(hashlib.md5(image_rgb.tobytes()).digest()[:4], "little")
    rng = np.random.default_rng(seed)
    h, w = max(1, image_rgb.shape[0] // FACTOR), max(1, image_rgb.shape[1] // FACTOR)
    yy, xx = np.mgrid[0:h, 0:w]
    density = np.zeros((h, w), dtype=np.float64)
    for _ in range(int(rng.integers(3, 6))):
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        s = rng.uniform(0.08, 0.2) * max(h, w)
        density += rng.uniform(0.5, 1.5) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))
    target = float(rng.uniform(150, 900))
    density *= target / density.sum()
    return float(density.sum()), density.astype(np.float32)


def predict(image):
    """Return (count, density_map). density_map is at 1/8 of the processed image size."""
    rgb = to_rgb_array(image)
    session = _get_session()
    if session is None:
        return _stub(rgb)
    x = preprocess(rgb)
    out = session.run(None, {session.get_inputs()[0].name: x})[0]
    density = np.clip(out[0, 0], 0, None).astype(np.float32)
    return float(density.sum()), density
