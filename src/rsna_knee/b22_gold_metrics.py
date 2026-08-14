from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .b21_acceptance_protocol import require_b20_replay_sanity
from .b22_duration_protocol import (
    B22_E2_REPLAY_TOLERANCE,
    B22_GOLD_AUDIT_VARIANT,
    require_failed_b21_acceptance,
)
from .evaluation import bootstrap_macro_auc, compare_runs


def build_b22_gold_trajectory(
    truth,
    b20_prediction,
    epoch_predictions: dict[int, np.ndarray],
    *,
    b21_acceptance_path: str | Path,
    n_bootstrap: int,
    seed: int,
) -> dict:
    prior = require_failed_b21_acceptance(b21_acceptance_path)
    b20_eval = bootstrap_macro_auc(truth, b20_prediction, n_bootstrap=n_bootstrap, seed=seed + 501)
    b20_delta = require_b20_replay_sanity(b20_eval.macro_auc)

    epochs = {}
    for epoch in sorted(epoch_predictions):
        pred = epoch_predictions[epoch]
        score = bootstrap_macro_auc(truth, pred, n_bootstrap=n_bootstrap, seed=seed + 510 + epoch)
        paired = compare_runs(
            truth,
            b20_prediction,
            pred,
            n_bootstrap=n_bootstrap,
            seed=seed + 520 + epoch,
        )
        epochs[str(epoch)] = {
            **score.to_dict(),
            "paired_minus_b20": paired,
            "raw_minus_b20": float(score.macro_auc - b20_eval.macro_auc),
        }

    if "2" not in epochs:
        raise ValueError("B22 trajectory audit requires epoch 2")
    prior_e2 = float(prior["b21_candidate"]["macro_auc"])
    new_e2 = float(epochs["2"]["macro_auc"])
    e2_delta = new_e2 - prior_e2
    if abs(e2_delta) > B22_E2_REPLAY_TOLERANCE:
        raise RuntimeError(
            "B22 E2 does not reproduce B21 E2 closely enough; duration trajectory is not interpretable"
        )

    best_epoch = max(epochs, key=lambda key: epochs[key]["macro_auc"])
    return {
        "variant": B22_GOLD_AUDIT_VARIANT,
        "role": "exploratory post-hoc duration trajectory; not a promotion decision",
        "working_model_remains": "B20_crop_only_joint_focus",
        "b20_replay": {
            **b20_eval.to_dict(),
            "replay_minus_canonical": float(b20_delta),
        },
        "prior_b21_e2_macro_auc": prior_e2,
        "b22_e2_minus_prior_b21_e2": float(e2_delta),
        "e2_replay_tolerance": B22_E2_REPLAY_TOLERANCE,
        "epochs": epochs,
        "best_epoch_by_reused_gold_exploratory": int(best_epoch),
        "best_macro_auc_by_reused_gold_exploratory": float(epochs[best_epoch]["macro_auc"]),
        "governance": (
            "Gold was reused after B21 acceptance; target/epoch results are exploratory only. "
            "Do not promote or retune from this audit without independent evidence."
        ),
    }
