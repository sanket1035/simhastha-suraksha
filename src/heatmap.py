"""Overlay a density map on a camera frame as a colour heatmap."""
import cv2
import numpy as np


def _normalise(density, size, percentile=100.0, gamma=1.0):
    """Resize to (w, h) and scale to 0..1. A percentile below 100 stops one hot pixel from washing
    out the rest; gamma below 1 lifts weak areas so they stay visible. Returns None for an empty map."""
    w, h = size
    d = cv2.resize(np.asarray(density, dtype=np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    d = np.clip(d, 0, None)
    peak = float(np.percentile(d, percentile)) if percentile < 100 else float(d.max())
    if peak <= 0:
        return None
    return np.clip(d / peak, 0, 1) ** gamma


def colorize_density(density, size, colormap=cv2.COLORMAP_JET, percentile=100.0, gamma=1.0):
    """Pure heatmap (no photo underneath) at `size` = (w, h), as an RGB uint8 image."""
    norm = _normalise(density, size, percentile, gamma)
    if norm is None:
        norm = np.zeros((size[1], size[0]), dtype=np.float32)
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), colormap)
    return cv2.cvtColor(color, cv2.COLOR_BGR2RGB)


def overlay_heatmap(image_rgb, density, alpha=0.55, min_fraction=0.05, colormap=cv2.COLORMAP_JET,
                    percentile=100.0, gamma=1.0):
    """Blend a JET heatmap onto the frame.

    The map is normalised by its own peak, so colours show where the crowd is
    densest in THIS frame (relative), not an absolute density. Areas below
    `min_fraction` of the peak stay untinted so the scene remains visible.
    """
    h, w = image_rgb.shape[:2]
    norm = _normalise(density, (w, h), percentile, gamma)
    if norm is None:
        return image_rgb.copy()
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), colormap)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB).astype(np.float32)
    weight = (np.clip((norm - min_fraction) / (1.0 - min_fraction), 0, 1) * alpha)[..., None]
    out = image_rgb.astype(np.float32) * (1.0 - weight) + color * weight
    return out.astype(np.uint8)
