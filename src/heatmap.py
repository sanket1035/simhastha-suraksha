"""Overlay a density map on a camera frame as a colour heatmap."""
import cv2
import numpy as np


def overlay_heatmap(image_rgb, density, alpha=0.55, min_fraction=0.05):
    """Blend a JET heatmap onto the frame.

    The map is normalised by its own peak, so colours show where the crowd is
    densest in THIS frame (relative), not an absolute density. Areas below
    `min_fraction` of the peak stay untinted so the scene remains visible.
    """
    h, w = image_rgb.shape[:2]
    d = cv2.resize(np.asarray(density, dtype=np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    d = np.clip(d, 0, None)
    peak = float(d.max())
    if peak <= 0:
        return image_rgb.copy()
    norm = d / peak
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB).astype(np.float32)
    weight = (np.clip((norm - min_fraction) / (1.0 - min_fraction), 0, 1) * alpha)[..., None]
    out = image_rgb.astype(np.float32) * (1.0 - weight) + color * weight
    return out.astype(np.uint8)
