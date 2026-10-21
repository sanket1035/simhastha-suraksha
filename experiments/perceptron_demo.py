"""Experiment 1: perceptron from scratch, with decision-boundary plots.

  python experiments/perceptron_demo.py   -> docs/perceptron_boundary.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.perceptron import Perceptron  # noqa: E402


def plot_boundary(ax, p, X, y, title, xlabel, ylabel):
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=25)
    xs = np.linspace(X[:, 0].min() - 0.2, X[:, 0].max() + 0.2, 100)
    if abs(p.w[1]) > 1e-9:
        ax.plot(xs, -(p.w[0] * xs + p.b) / p.w[1], "k--", label="decision boundary")
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.legend()


def main():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    gates = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    for ax, (name, y) in zip(axes[:2], [("AND gate", [0, 0, 0, 1]), ("OR gate", [0, 1, 1, 1])]):
        p = Perceptron(2)
        p.fit(gates, y, epochs=50)
        plot_boundary(ax, p, gates, np.array(y), "%s (accuracy %.0f%%)" % (name, 100 * p.accuracy(gates, y)), "x1", "x2")

    rng = np.random.default_rng(3)
    area, count = rng.uniform(50, 400, 400), rng.uniform(0, 1800, 400)
    density = count / area
    keep = np.abs(density - 4.0) > 0.3
    X = np.column_stack([count[keep] / 100, area[keep] / 100])
    y = (density[keep] >= 4.0).astype(int)
    p = Perceptron(2, lr=0.1)
    history = p.fit(X, y, epochs=500)
    plot_boundary(axes[2], p, X, y, "Unsafe zone: density >= 4 (accuracy %.0f%%, %d epochs)" % (100 * p.accuracy(X, y), len(history)),
                  "people count / 100", "zone area / 100 m2")
    out = Path(__file__).resolve().parent.parent / "docs" / "perceptron_boundary.png"
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=120)
    print("Saved", out)


if __name__ == "__main__":
    main()
