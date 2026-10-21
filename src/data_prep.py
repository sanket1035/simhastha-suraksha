"""Turn ShanghaiTech crowd-counting annotations into density maps.

Usage:
  python -m src.data_prep --root /path/to/ShanghaiTech --part A --out data/processed

Expected layout (names vary slightly between mirrors; the script tries a few):
  <root>/part_A_final/train_data/images/IMG_1.jpg
  <root>/part_A_final/train_data/ground_truth/GT_IMG_1.mat

Writes, per split:
  <out>/part_A/density/<split>/IMG_1.npy   float32 density map (full resolution)
  <out>/part_A/<split>_index.csv           image_path, density_path, gt_count, ...
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from .density import make_density_map

INDEX_FIELDS = ["image_path", "density_path", "gt_count", "density_sum", "height", "width"]


def load_points(mat_path):
    """Read the (x, y) head points from a ShanghaiTech ground-truth .mat file."""
    from scipy.io import loadmat
    mat = loadmat(str(mat_path))
    try:
        pts = mat["image_info"][0, 0][0, 0][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Unexpected .mat layout in %s (is this the crowd-counting dataset?)" % mat_path) from exc
    return np.asarray(pts, dtype=np.float64).reshape(-1, 2)


def find_part_dir(root, part):
    root = Path(root)
    for name in ("part_%s_final" % part, "part_%s" % part, "Part_%s_final" % part, "Part_%s" % part, "part%s" % part):
        if (root / name / "train_data").exists():
            return root / name
    if (root / "train_data").exists():
        return root
    listing = sorted(p.name for p in root.iterdir()) if root.exists() else []
    raise FileNotFoundError("Could not find part %s under %s. Found: %s" % (part, root, listing))


def process_split(part_dir, split, out_dir, adaptive):
    img_dir = Path(part_dir) / ("%s_data" % split) / "images"
    gt_dir = Path(part_dir) / ("%s_data" % split) / "ground_truth"
    den_dir = Path(out_dir) / "density" / split
    den_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        gt_path = gt_dir / ("GT_%s.mat" % img_path.stem)
        with Image.open(img_path) as im:
            w, h = im.size
        pts = load_points(gt_path)
        dens = make_density_map((h, w), pts, adaptive=adaptive)
        out_path = den_dir / ("%s.npy" % img_path.stem)
        np.save(out_path, dens)
        kept = int(((pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)).sum())
        if abs(float(dens.sum()) - kept) > 0.5:
            print("WARNING: %s density sum %.2f != %d heads" % (img_path.name, dens.sum(), kept))
        rows.append({"image_path": str(img_path.resolve()), "density_path": str(out_path.resolve()),
                     "gt_count": len(pts), "density_sum": round(float(dens.sum()), 3), "height": h, "width": w})
    index_path = Path(out_dir) / ("%s_index.csv" % split)
    with open(index_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print("%s: %d images -> %s" % (split, len(rows), index_path))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="Folder containing the ShanghaiTech data")
    ap.add_argument("--part", choices=["A", "B"], default="A")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--adaptive", choices=["auto", "yes", "no"], default="auto",
                    help="Geometry-adaptive sigma. auto = yes for Part A (dense), no for Part B")
    args = ap.parse_args()
    adaptive = (args.part == "A") if args.adaptive == "auto" else (args.adaptive == "yes")
    part_dir = find_part_dir(args.root, args.part)
    out_dir = Path(args.out) / ("part_%s" % args.part)
    for split in ("train", "test"):
        process_split(part_dir, split, out_dir, adaptive)


if __name__ == "__main__":
    main()
