"""Train MLP / CNN / CSRNet on the processed ShanghaiTech data.

Examples:
  python -m src.train --model mlp    --train-index data/processed/part_A/train_index.csv --out runs/mlp
  python -m src.train --model cnn    --train-index ... --out runs/cnn --dropout 0.3 --weight-decay 1e-4
  python -m src.train --model csrnet --train-index ... --out runs/csrnet --optimizer adam --lr 1e-5 --amp
  python -m src.train --model csrnet --freeze-frontend ...    # freeze vs fine-tune experiment

Validation MAE (people count) picks best.pth. Log goes to <out>/log.csv (and WandB with --wandb).
"""
import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import CrowdDataset, read_index, split_train_val
from .models import build_model


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["mlp", "cnn", "csrnet"], required=True)
    ap.add_argument("--train-index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=1, help="CSRNet default 1 (full images); use --crop-size for more")
    ap.add_argument("--crop-size", type=int, default=0, help="CSRNet random crop (multiple of 8); 0 = full image")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    ap.add_argument("--momentum", type=float, default=0.95)
    ap.add_argument("--weight-decay", type=float, default=0.0, help="L2 regularization")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--activation", choices=["relu", "tanh", "sigmoid"], default="relu", help="mlp/cnn only")
    ap.add_argument("--count-size", type=int, default=128, help="mlp/cnn input size")
    ap.add_argument("--count-scale", type=float, default=100.0, help="mlp/cnn: target = count / scale")
    ap.add_argument("--freeze-frontend", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--max-train-images", type=int, default=0, help="quick smoke test with fewer images")
    ap.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="simhastha-suraksha")
    return ap.parse_args(argv)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def predicted_and_true_counts(model_name, outputs, targets, scale):
    """Per-image counts. CSRNet: sum of the density map. MLP/CNN: the scalar output."""
    if model_name == "csrnet":
        return outputs.sum(dim=(1, 2, 3)), targets.sum(dim=(1, 2, 3))
    return outputs * scale, targets


def run_epoch(model, loader, device, args, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    loss_fn = torch.nn.MSELoss(reduction="sum") if args.model == "csrnet" else torch.nn.L1Loss()
    total_loss, preds, gts = 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if args.model != "csrnet":
                y_in = y / args.count_scale
            else:
                y_in = y
            with torch.autocast(device_type=device.type, enabled=bool(args.amp and device.type == "cuda")):
                out = model(x)
                loss = loss_fn(out.float(), y_in)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += float(loss.item())
            p, g = predicted_and_true_counts(args.model, out.detach().float(), y, args.count_scale)
            preds += p.cpu().tolist()
            gts += g.cpu().tolist()
    preds, gts = np.array(preds), np.array(gts)
    return {"loss": total_loss / max(1, len(loader)), "mae": float(np.abs(preds - gts).mean()),
            "rmse": float(np.sqrt(((preds - gts) ** 2).mean()))}


def make_scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = read_index(args.train_index)
    if args.max_train_images:
        rows = rows[:args.max_train_images]
    train_rows, val_rows = split_train_val(rows, args.val_fraction, args.seed)
    mode = "density" if args.model == "csrnet" else "count"
    train_ds = CrowdDataset(train_rows, mode, train=True, crop_size=args.crop_size, count_size=args.count_size)
    val_ds = CrowdDataset(val_rows, mode, train=False, count_size=args.count_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)
    print("device=%s train=%d val=%d" % (device, len(train_ds), len(val_ds)))

    model = build_model(args.model, activation=args.activation, dropout=args.dropout, input_size=args.count_size,
                        pretrained=not args.no_pretrained, freeze_frontend=args.freeze_frontend).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scaler = make_scaler(bool(args.amp and device.type == "cuda"))

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, config=vars(args))

    config = vars(args)
    best = float("inf")
    log_path = out / "log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_mae", "val_mae", "val_rmse", "seconds"])
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            tr = run_epoch(model, train_loader, device, args, optimizer, scaler)
            va = run_epoch(model, val_loader, device, args)
            secs = time.time() - t0
            writer.writerow([epoch, tr["loss"], tr["mae"], va["mae"], va["rmse"], round(secs, 1)])
            f.flush()
            print("epoch %d | train mae %.2f | val mae %.2f rmse %.2f | %.0fs" % (epoch, tr["mae"], va["mae"], va["rmse"], secs))
            if run is not None:
                run.log({"epoch": epoch, "train_loss": tr["loss"], "train_mae": tr["mae"],
                         "val_mae": va["mae"], "val_rmse": va["rmse"]})
            ckpt = {"state_dict": model.state_dict(), "config": config, "epoch": epoch, "val_mae": va["mae"]}
            torch.save(ckpt, out / "last.pth")
            if va["mae"] < best:
                best = va["mae"]
                torch.save(ckpt, out / "best.pth")
    print("best val MAE %.2f -> %s" % (best, out / "best.pth"))
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
