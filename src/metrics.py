"""Counting metrics. MAE and RMSE are computed on per-image counts."""
import numpy as np


def mae(pred, gt):
    pred, gt = np.asarray(pred, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    return float(np.mean(np.abs(pred - gt)))


def rmse(pred, gt):
    pred, gt = np.asarray(pred, dtype=np.float64), np.asarray(gt, dtype=np.float64)
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def count_metrics(pred, gt):
    return {"mae": mae(pred, gt), "rmse": rmse(pred, gt)}
