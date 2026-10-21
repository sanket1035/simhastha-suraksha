"""Density map generation from head-point annotations (NumPy/SciPy only).

A density map has a small Gaussian blob at every annotated head. Each blob sums
to 1, so the sum of the whole map equals the number of people.
"""
import numpy as np
from scipy.spatial import cKDTree

MAX_SIGMA = 40.0


def _add_gaussian(density, x, y, sigma):
    """Add one Gaussian (sum 1 inside the image) centred on pixel (x, y)."""
    h, w = density.shape
    radius = max(1, int(round(4 * sigma)))
    x0, x1 = max(x - radius, 0), min(x + radius + 1, w)
    y0, y1 = max(y - radius, 0), min(y + radius + 1, h)
    xs = np.arange(x0, x1) - x
    ys = np.arange(y0, y1) - y
    kx = np.exp(-(xs ** 2) / (2.0 * sigma ** 2))
    ky = np.exp(-(ys ** 2) / (2.0 * sigma ** 2))
    kernel = np.outer(ky, kx)
    kernel /= kernel.sum()  # keeps the count exact even near image borders
    density[y0:y1, x0:x1] += kernel.astype(np.float32)


def make_density_map(shape_hw, points, adaptive=True, beta=0.3, k=3, fixed_sigma=15.0):
    """Build a density map of size `shape_hw` from (x, y) head points.

    adaptive=True: geometry-adaptive sigma = beta * mean distance to the k
    nearest neighbours (good for dense crowds, ShanghaiTech Part A).
    adaptive=False: the same fixed sigma for every head (sparse crowds, Part B).
    Points outside the image are dropped.
    """
    h, w = shape_hw
    density = np.zeros((h, w), dtype=np.float32)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    inside = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    pts = pts[inside]
    n = len(pts)
    if n == 0:
        return density
    if adaptive and n > 1:
        kk = min(k, n - 1)
        dist, _ = cKDTree(pts).query(pts, k=kk + 1)
        sigmas = beta * dist[:, 1:].mean(axis=1)
    else:
        sigmas = np.full(n, fixed_sigma)
    sigmas = np.clip(sigmas, 1.0, MAX_SIGMA)
    for (px, py), s in zip(pts, sigmas):
        x = min(max(int(round(px)), 0), w - 1)
        y = min(max(int(round(py)), 0), h - 1)
        _add_gaussian(density, x, y, float(s))
    return density


def downsample_density_sum(density, factor=8):
    """Shrink a density map by summing factor x factor blocks.

    Sum pooling keeps the total count exact, so no 'multiply by 64' fix is
    needed (that fix is only required when resizing with interpolation).
    Edges that do not fill a whole block are cropped.
    """
    h, w = density.shape
    hh, ww = h // factor * factor, w // factor * factor
    d = density[:hh, :ww]
    return d.reshape(hh // factor, factor, ww // factor, factor).sum(axis=(1, 3))
