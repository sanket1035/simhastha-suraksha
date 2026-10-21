"""Evaluate a checkpoint on the test split: MAE, RMSE, predictions.csv, worst cases.

  python -m src.evaluate --checkpoint runs/csrnet/best.pth \
      --test-index data/processed/part_A/test_index.csv --out runs/csrnet/eval --save-worst 10
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import CrowdDataset, read_index
from .heatmap import overlay_heatmap
from .metrics import count_metrics
from .models import build_model


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    model = build_model(cfg["model"], activation=cfg.get("activation", "relu"), dropout=0.0,
                        input_size=cfg.get("count_size", 128), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test-index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-worst", type=int, default=0, help="save heatmap overlays of the N worst images (csrnet)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_checkpoint(args.checkpoint, device)
    is_csr = cfg["model"] == "csrnet"
    rows = read_index(args.test_index)
    ds = CrowdDataset(rows, "density" if is_csr else "count", train=False, count_size=cfg.get("count_size", 128))
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    results, maps = [], {}
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            out = model(x.to(device)).float().cpu()
            if is_csr:
                pred, gt = float(out.sum()), float(y.sum())
                maps[i] = out[0, 0].clamp(min=0).numpy()
            else:
                pred, gt = float(out[0]) * cfg.get("count_scale", 100.0), float(y[0])
            results.append({"image": Path(rows[i]["image_path"]).name, "gt_count": round(gt, 2),
                            "pred_count": round(pred, 2), "abs_error": round(abs(pred - gt), 2)})

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "predictions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "gt_count", "pred_count", "abs_error"])
        writer.writeheader()
        writer.writerows(results)

    m = count_metrics([r["pred_count"] for r in results], [r["gt_count"] for r in results])
    print("Test images: %d | MAE %.2f | RMSE %.2f" % (len(results), m["mae"], m["rmse"]))
    print("Predictions saved to %s" % (outdir / "predictions.csv"))

    if args.save_worst and is_csr:
        worst = sorted(range(len(results)), key=lambda k: -results[k]["abs_error"])[:args.save_worst]
        for k in worst:
            img = cv2.cvtColor(cv2.imread(rows[k]["image_path"]), cv2.COLOR_BGR2RGB)
            h8, w8 = img.shape[0] // 8 * 8, img.shape[1] // 8 * 8
            over = overlay_heatmap(img[:h8, :w8], maps[k])
            label = "gt %.0f pred %.0f" % (results[k]["gt_count"], results[k]["pred_count"])
            cv2.putText(over, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.imwrite(str(outdir / ("worst_%s.jpg" % Path(rows[k]["image_path"]).stem)), cv2.cvtColor(over, cv2.COLOR_RGB2BGR))
        print("Saved %d worst-case overlays to %s" % (len(worst), outdir))


if __name__ == "__main__":
    main()
