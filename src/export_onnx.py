"""Export a trained CSRNet checkpoint to ONNX (dynamic height/width) and verify it.

  python -m src.export_onnx --checkpoint runs/csrnet/best.pth --out models/csrnet.onnx
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from .evaluate import load_checkpoint


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="models/csrnet.onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    model, cfg = load_checkpoint(args.checkpoint, torch.device("cpu"))
    if cfg["model"] != "csrnet":
        raise SystemExit("Only csrnet checkpoints can be exported (got %s)" % cfg["model"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, 512, 512)
    kwargs = dict(input_names=["image"], output_names=["density"], opset_version=args.opset,
                  dynamic_axes={"image": {0: "batch", 2: "height", 3: "width"},
                                "density": {0: "batch", 2: "height", 3: "width"}})
    try:
        torch.onnx.export(model, dummy, args.out, dynamo=False, **kwargs)  # legacy exporter keeps dynamic_axes
    except TypeError:  # older torch has no `dynamo` argument
        torch.onnx.export(model, dummy, args.out, **kwargs)
    print("Exported to %s (%.1f MB)" % (args.out, Path(args.out).stat().st_size / 1e6))

    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed: skipped verification")
        return
    session = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    test = torch.randn(1, 3, 320, 448)  # different size on purpose: checks dynamic axes
    with torch.no_grad():
        expected = model(test).numpy()
    got = session.run(None, {"image": test.numpy()})[0]
    diff = float(np.abs(expected - got).max())
    print("Max abs difference PyTorch vs ONNX: %.2e | count %.3f vs %.3f" % (diff, expected.sum(), got.sum()))
    if diff > 1e-3:
        raise SystemExit("ONNX output does not match PyTorch closely enough")
    print("ONNX export verified.")


if __name__ == "__main__":
    main()
