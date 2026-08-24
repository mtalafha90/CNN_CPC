"""B46 prospective gold-anchored cross-fitted supervision contract.

B46 changes supervision only.  The image/model/training geometry is the frozen
B42 constant-area native-aspect sparse-MIL endpoint.  The 58 official expert
studies are partitioned once into five deterministic multilabel-balanced folds.
For fold f, the other four folds may enter gradients with hard official labels;
fold f is prediction-only and must never enter that fold model's gradients.

The gold cell weight is fixed prospectively at 4.0.  Target-balance multipliers
remain those computed from the historical weak/report supervision only, so
adding gold labels does not silently rescale the weak objective target by target.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .b42_constant_area_aspect_sparse_mil import require_b42_contract
from .constants import TARGETS
from .data import gold_mask, load_train_csv

B46_VERSION = "b46_gold_anchored_crossfit_b42_v1"
B46_EXPERIMENT = "B46_gold_anchored_crossfit_B42"
B46_NUMBERED_CONTAINER = "runs/079_Experiment_B46_gold_anchored_crossfit"
B46_RUN_ROOT = f"{B46_NUMBERED_CONTAINER}/b46_gold_anchored_crossfit"
B46_MANIFEST_NAME = "gold_folds.json"
B46_N_FOLDS = 5
B46_GOLD_STUDIES = 58
B46_GOLD_CELL_WEIGHT = 4.0
B46_FOLD_SALT = "CNN_CPC|B46|gold-crossfit|2026-08-25"
B46_GOLD_TARGETS = "hard_binary_0_1"
B46_TARGET_BALANCE_SOURCE = "weak_only_frozen"
B46_FIXED_EPOCHS = 2


def require_b46_contract(config: dict) -> dict:
    """Require B42 unchanged plus the prospective B46 supervision choices."""
    crop = require_b42_contract(config)
    expected = {
        "b46_n_folds": B46_N_FOLDS,
        "b46_gold_cell_weight": B46_GOLD_CELL_WEIGHT,
        "b46_fixed_epochs": B46_FIXED_EPOCHS,
    }
    if int(config.get("b46_n_folds", B46_N_FOLDS)) != B46_N_FOLDS:
        raise ValueError(f"B46 freezes b46_n_folds={B46_N_FOLDS}")
    if not np.isclose(
        float(config.get("b46_gold_cell_weight", B46_GOLD_CELL_WEIGHT)),
        B46_GOLD_CELL_WEIGHT,
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError(f"B46 freezes b46_gold_cell_weight={B46_GOLD_CELL_WEIGHT}")
    if int(config.get("b46_fixed_epochs", B46_FIXED_EPOCHS)) != B46_FIXED_EPOCHS:
        raise ValueError(f"B46 freezes b46_fixed_epochs={B46_FIXED_EPOCHS}")
    if str(config.get("b46_fold_salt", B46_FOLD_SALT)) != B46_FOLD_SALT:
        raise ValueError("B46 fold salt changed")
    if str(config.get("b46_gold_targets", B46_GOLD_TARGETS)) != B46_GOLD_TARGETS:
        raise ValueError("B46 requires hard official 0/1 gold targets")
    if str(
        config.get("b46_target_balance_source", B46_TARGET_BALANCE_SOURCE)
    ) != B46_TARGET_BALANCE_SOURCE:
        raise ValueError("B46 target balancing must remain weak-only/frozen")
    if bool(config.get("b46_use_gold_for_early_stopping", False)):
        raise ValueError("B46 forbids gold early stopping")
    if bool(config.get("b46_use_heldout_gold_in_gradients", False)):
        raise ValueError("B46 forbids held-out gold gradients")
    return crop


def _gold_frame(train: pd.DataFrame) -> pd.DataFrame:
    frame = train.loc[gold_mask(train), ["StudyInstanceUID", *TARGETS]].copy()
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    if len(frame) != B46_GOLD_STUDIES:
        raise ValueError(
            f"B46 requires exactly {B46_GOLD_STUDIES} official gold studies; got {len(frame)}"
        )
    if frame["StudyInstanceUID"].duplicated().any():
        raise ValueError("B46 gold StudyInstanceUID values are not unique")
    y = frame[TARGETS].to_numpy(np.float64)
    if not np.isfinite(y).all() or not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("B46 requires complete binary official gold labels")
    return frame.reset_index(drop=True)


def _uid_key(uid: str, *, salt: str = B46_FOLD_SALT) -> str:
    return hashlib.sha256(f"{salt}\0{uid}".encode("utf-8")).hexdigest()


def assign_gold_folds(
    uids: list[str],
    labels: np.ndarray,
    *,
    n_folds: int = B46_N_FOLDS,
    salt: str = B46_FOLD_SALT,
) -> np.ndarray:
    """Deterministic greedy multilabel stratification for the 58 gold studies.

    Fold capacities are fixed before assignment.  Studies carrying rarer class
    states are assigned first.  At each step, choose the non-full fold that
    minimizes squared deviation of positive counts from its capacity-scaled
    target prevalence, with deterministic SHA-256 tie breaking.  Since fold
    size is fixed, balancing positives also balances negatives.
    """
    uids = [str(uid) for uid in uids]
    y = np.asarray(labels, dtype=np.int64)
    if y.ndim != 2 or y.shape[0] != len(uids) or y.shape[1] != len(TARGETS):
        raise ValueError("B46 fold labels must have shape [N,12]")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("B46 fold labels must be binary")
    if len(set(uids)) != len(uids):
        raise ValueError("B46 fold UIDs must be unique")
    if int(n_folds) != B46_N_FOLDS:
        raise ValueError("B46 fold count is frozen at five")

    n = len(uids)
    base = n // n_folds
    remainder = n % n_folds
    capacities = np.asarray(
        [base + (1 if fold < remainder else 0) for fold in range(n_folds)],
        dtype=np.int64,
    )
    total_pos = y.sum(axis=0).astype(np.float64)
    total_neg = (n - y.sum(axis=0)).astype(np.float64)
    class_count = np.where(y == 1, total_pos[None, :], total_neg[None, :])
    rarity = (1.0 / np.maximum(class_count, 1.0)).sum(axis=1)
    order = sorted(
        range(n),
        key=lambda i: (-float(rarity[i]), _uid_key(uids[i], salt=salt)),
    )

    fold_size = np.zeros(n_folds, dtype=np.int64)
    fold_pos = np.zeros((n_folds, y.shape[1]), dtype=np.float64)
    assignment = np.full(n, -1, dtype=np.int64)
    desired_pos = capacities[:, None] * (total_pos[None, :] / float(n))
    denom = np.maximum(total_pos, 1.0)

    for index in order:
        candidates: list[tuple[float, str, int]] = []
        for fold in range(n_folds):
            if fold_size[fold] >= capacities[fold]:
                continue
            new_pos = fold_pos[fold] + y[index]
            pos_cost = float((((new_pos - desired_pos[fold]) ** 2) / denom).sum())
            fill_cost = float(
                ((fold_size[fold] + 1) / max(int(capacities[fold]), 1)) ** 2
            ) * 1e-6
            tie = hashlib.sha256(
                f"{salt}\0{uids[index]}\0fold={fold}".encode("utf-8")
            ).hexdigest()
            candidates.append((pos_cost + fill_cost, tie, fold))
        if not candidates:
            raise RuntimeError("B46 fold assignment exhausted all capacities")
        _, _, chosen = min(candidates)
        assignment[index] = int(chosen)
        fold_size[chosen] += 1
        fold_pos[chosen] += y[index]

    if (assignment < 0).any() or not np.array_equal(fold_size, capacities):
        raise RuntimeError("B46 deterministic fold assignment is incomplete")
    return assignment


def build_gold_fold_manifest(
    data_root: str | Path,
    *,
    out_path: str | Path,
) -> dict:
    """Create the one frozen B46 gold fold manifest from official train.csv."""
    root = Path(data_root).resolve()
    train_path = root / "train.csv"
    train = load_train_csv(train_path)
    if len(train) != 4407:
        raise ValueError("B46 requires the complete 4,407-study training release")
    gold = _gold_frame(train)
    uids = gold["StudyInstanceUID"].tolist()
    y = gold[TARGETS].to_numpy(np.int64)
    folds = assign_gold_folds(uids, y)

    rows = []
    fold_summary = []
    for uid, fold, target_row in zip(uids, folds, y):
        row = {"StudyInstanceUID": uid, "fold": int(fold)}
        row.update({target: int(value) for target, value in zip(TARGETS, target_row)})
        rows.append(row)
    for fold in range(B46_N_FOLDS):
        mask = folds == fold
        fold_y = y[mask]
        fold_summary.append(
            {
                "fold": fold,
                "studies": int(mask.sum()),
                "positive_counts": {
                    target: int(fold_y[:, j].sum()) for j, target in enumerate(TARGETS)
                },
                "negative_counts": {
                    target: int(mask.sum() - fold_y[:, j].sum())
                    for j, target in enumerate(TARGETS)
                },
            }
        )

    train_sha = hashlib.sha256(train_path.read_bytes()).hexdigest()
    payload = {
        "experiment": B46_EXPERIMENT,
        "version": B46_VERSION,
        "status": "frozen_before_B46_training_or_OOF_evaluation",
        "source_train_csv": str(train_path),
        "source_train_csv_sha256": train_sha,
        "n_gold_studies": B46_GOLD_STUDIES,
        "n_folds": B46_N_FOLDS,
        "fold_salt": B46_FOLD_SALT,
        "assignment_algorithm": (
            "deterministic greedy multilabel positive-count balancing with fixed "
            "12/12/12/11/11 capacities and SHA256 tie breaks"
        ),
        "fold_sizes": [int((folds == fold).sum()) for fold in range(B46_N_FOLDS)],
        "global_positive_counts": {
            target: int(y[:, j].sum()) for j, target in enumerate(TARGETS)
        },
        "global_negative_counts": {
            target: int(len(y) - y[:, j].sum()) for j, target in enumerate(TARGETS)
        },
        "fold_summary": fold_summary,
        "rows": rows,
        "governance": (
            "Fold membership is frozen before B46 training. A fold model may use gold "
            "labels only from the other four folds. The held-out fold is prediction-only."
        ),
    }
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(path)
    print("manifest_sha256=", hashlib.sha256(path.read_bytes()).hexdigest())
    return payload


def load_gold_fold_manifest(path: str | Path) -> dict:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("experiment") != B46_EXPERIMENT or payload.get("version") != B46_VERSION:
        raise ValueError("manifest is not the frozen B46 fold definition")
    if int(payload.get("n_gold_studies", -1)) != B46_GOLD_STUDIES:
        raise ValueError("B46 manifest gold-study count changed")
    if int(payload.get("n_folds", -1)) != B46_N_FOLDS:
        raise ValueError("B46 manifest fold count changed")
    if str(payload.get("fold_salt", "")) != B46_FOLD_SALT:
        raise ValueError("B46 manifest fold salt changed")
    rows = payload.get("rows", [])
    if len(rows) != B46_GOLD_STUDIES:
        raise ValueError("B46 manifest row count changed")
    uids = [str(row.get("StudyInstanceUID", "")) for row in rows]
    if len(set(uids)) != B46_GOLD_STUDIES or any(not uid for uid in uids):
        raise ValueError("B46 manifest UIDs are invalid")
    folds = np.asarray([int(row.get("fold", -1)) for row in rows], dtype=np.int64)
    if sorted(np.bincount(folds, minlength=B46_N_FOLDS).tolist()) != [11, 11, 12, 12, 12]:
        raise ValueError("B46 manifest fold sizes changed")
    for row in rows:
        values = np.asarray([row.get(target) for target in TARGETS], dtype=np.float64)
        if not np.isin(values, [0.0, 1.0]).all():
            raise ValueError("B46 manifest target labels are not complete binary values")
    return payload


def heldout_uids(manifest: dict, fold: int) -> list[str]:
    f = int(fold)
    if f < 0 or f >= B46_N_FOLDS:
        raise ValueError("B46 fold must be in [0,4]")
    return [
        str(row["StudyInstanceUID"])
        for row in manifest["rows"]
        if int(row["fold"]) == f
    ]


def training_gold_uids(manifest: dict, fold: int) -> list[str]:
    held = set(heldout_uids(manifest, fold))
    return [str(row["StudyInstanceUID"]) for row in manifest["rows"] if str(row["StudyInstanceUID"]) not in held]


__all__ = [
    "B46_VERSION",
    "B46_EXPERIMENT",
    "B46_RUN_ROOT",
    "B46_MANIFEST_NAME",
    "B46_N_FOLDS",
    "B46_GOLD_STUDIES",
    "B46_GOLD_CELL_WEIGHT",
    "B46_FOLD_SALT",
    "assign_gold_folds",
    "build_gold_fold_manifest",
    "heldout_uids",
    "load_gold_fold_manifest",
    "require_b46_contract",
    "training_gold_uids",
]
