"""Put a camera's crowd density on the map.

A camera angle covers a rectangle of ground: centre (lat, lon), `bearing` (compass direction the camera
looks, degrees clockwise from north), `width_m` across and `area / width_m` deep. The bottom of the kept
photo is the near edge, the top is the far edge. The density map is cut into a coarse grid and each cell
is placed inside the rectangle, then coloured by people per square metre.

This is a flat, top-down approximation: it ignores perspective (far rows really cover more ground), so
treat the picture as "where along the area", not as survey-grade positions.
"""
import math

import cv2
import numpy as np

from .risk import level_for_density

GRID_ROWS, GRID_COLS = 14, 20
METRES_PER_DEG_LAT = 111320.0


def coarse_grid(density_map, ignore_rows=0, rows=GRID_ROWS, cols=GRID_COLS):
    """People per cell on a rows x cols grid over the kept part (below `ignore_rows`). Sum is preserved."""
    kept = np.asarray(density_map, dtype=np.float32)[ignore_rows:]
    if kept.size == 0:
        return np.zeros((rows, cols), dtype=np.float32)
    h, w = kept.shape
    grid = cv2.resize(kept, (cols, rows), interpolation=cv2.INTER_AREA)
    return (grid * (h * w) / float(rows * cols)).astype(np.float32)


def _to_latlon(geo, forward_m, lateral_m):
    t = math.radians(geo.get("bearing", 0.0))
    east = forward_m * math.sin(t) + lateral_m * math.cos(t)
    north = forward_m * math.cos(t) - lateral_m * math.sin(t)
    lat = geo["lat"] + north / METRES_PER_DEG_LAT
    lon = geo["lon"] + east / (METRES_PER_DEG_LAT * math.cos(math.radians(geo["lat"])))
    return [round(lat, 6), round(lon, 6)]


def default_geo(lat, lon, area, bearing=0.0):
    """A square footprint with the given area, for a camera whose exact coverage is not known yet."""
    return {"lat": float(lat), "lon": float(lon), "bearing": float(bearing), "width_m": math.sqrt(max(area, 1.0))}


def footprint(geo, area):
    """Corners (lat, lon) of the rectangle the angle covers."""
    width = max(float(geo["width_m"]), 1.0)
    depth = area / width
    return [_to_latlon(geo, f, l) for f, l in ((-depth / 2, -width / 2), (-depth / 2, width / 2),
                                                (depth / 2, width / 2), (depth / 2, -width / 2))]


def map_cells(grid, geo, area, thresholds=None):
    """One dict per grid cell: polygon corners, people, people per m2 and risk level."""
    grid = np.asarray(grid, dtype=np.float32)
    rows, cols = grid.shape
    width = max(float(geo["width_m"]), 1.0)
    depth = area / width
    cell_area = area / float(rows * cols)
    cells = []
    for i in range(rows):
        f0 = ((rows - i - 1) / float(rows) - 0.5) * depth
        f1 = ((rows - i) / float(rows) - 0.5) * depth
        for j in range(cols):
            l0 = (j / float(cols) - 0.5) * width
            l1 = ((j + 1) / float(cols) - 0.5) * width
            count = float(grid[i, j])
            density = count / cell_area
            cells.append({"poly": [_to_latlon(geo, f0, l0), _to_latlon(geo, f0, l1),
                                   _to_latlon(geo, f1, l1), _to_latlon(geo, f1, l0)],
                          "count": round(count, 3), "density": round(density, 3),
                          "level": level_for_density(density, thresholds)})
    return cells
