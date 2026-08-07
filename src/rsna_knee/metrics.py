from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def per_class_auc(y_true: np.ndarray, y_prob: np.ndarray) -> dict[int, float]:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.shape != y_prob.shape or y_true.ndim != 2:
        raise ValueError("y_true and y_prob must have the same 2D shape")
    out: dict[int, float] = {}
    for j in range(y_true.shape[1]):
        y = y_true[:, j]
        out[j] = float("nan") if np.unique(y).size < 2 else float(roc_auc_score(y, y_prob[:, j]))
    return out


def macro_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    vals = np.asarray(list(per_class_auc(y_true, y_prob).values()), dtype=float)
    return float("nan") if np.all(np.isnan(vals)) else float(np.nanmean(vals))
