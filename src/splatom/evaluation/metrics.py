"""Numerically safe metrics for joint-angle measurements."""
from __future__ import annotations

import numpy as np


def angle_error_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float | int]:
    """Compute MAE/RMSE in degrees, ignoring non-finite pairs."""
    pred = np.asarray(predicted, dtype=np.float64)
    true = np.asarray(target, dtype=np.float64)
    if pred.shape != true.shape:
        raise ValueError("predicted and target must have identical shapes")
    valid = np.isfinite(pred) & np.isfinite(true)
    count = int(valid.sum())
    if count == 0:
        return {"count": 0, "mae_deg": float("nan"), "rmse_deg": float("nan")}
    error = pred[valid] - true[valid]
    return {
        "count": count,
        "mae_deg": float(np.mean(np.abs(error))),
        "rmse_deg": float(np.sqrt(np.mean(np.square(error)))),
    }
