from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import SUBMISSION_COLUMNS, TARGETS


def _load_prediction(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = set(SUBMISSION_COLUMNS)
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} missing prediction columns: {missing}")
    out = df[SUBMISSION_COLUMNS].copy()
    out["StudyInstanceUID"] = out["StudyInstanceUID"].astype(str)
    if out["StudyInstanceUID"].duplicated().any():
        raise ValueError(f"{path} contains duplicate StudyInstanceUID values")
    values = out[TARGETS].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains non-finite predictions")
    return out


def ensemble_predictions(paths: list[str | Path], method: str = "rank") -> pd.DataFrame:
    """Combine aligned OOF/submission files using rank or probability mean.

    Rank averaging is attractive for a macro-AUC competition because it focuses
    on ordering and reduces calibration-scale differences between heterogeneous
    2.5D, transformer and 3D models.
    """
    if len(paths) < 2:
        raise ValueError("ensemble requires at least two prediction files")
    frames = [_load_prediction(p) for p in paths]
    ids = frames[0]["StudyInstanceUID"].tolist()
    id_set = set(ids)
    aligned = []
    for path, frame in zip(paths, frames):
        if set(frame["StudyInstanceUID"]) != id_set:
            raise ValueError(f"prediction IDs do not match: {path}")
        order = {uid: i for i, uid in enumerate(ids)}
        aligned.append(frame.sort_values("StudyInstanceUID", key=lambda s: s.map(order)))

    method = str(method).lower()
    stacked = np.stack([f[TARGETS].to_numpy(float) for f in aligned], axis=0)
    if method == "mean":
        pred = stacked.mean(axis=0)
    elif method == "rank":
        ranked = []
        n = stacked.shape[1]
        for model_pred in stacked:
            rank_matrix = np.empty_like(model_pred, dtype=np.float64)
            for j in range(model_pred.shape[1]):
                rank_matrix[:, j] = pd.Series(model_pred[:, j]).rank(method="average").to_numpy()
            ranked.append((rank_matrix - 0.5) / max(n, 1))
        pred = np.mean(np.stack(ranked, axis=0), axis=0)
    else:
        raise ValueError("ensemble method must be 'rank' or 'mean'")

    out = pd.DataFrame(pred, columns=TARGETS)
    out.insert(0, "StudyInstanceUID", ids)
    return out[SUBMISSION_COLUMNS]
