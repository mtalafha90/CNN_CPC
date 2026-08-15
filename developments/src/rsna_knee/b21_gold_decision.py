from __future__ import annotations

from pathlib import Path

from .b21_acceptance_protocol import (
    B20_CANONICAL_EPOCH,
    B20_CANONICAL_GOLD_MACRO_AUC,
    B20_REPLAY_SANITY_TOLERANCE,
    B21_GOLD_ACCEPTANCE_VARIANT,
    PROMOTION_RULE,
    SCIENTIFIC_SUPERIORITY_RULE,
    promotion_decision,
    require_b20_replay_sanity,
    scientific_superiority_decision,
)
from .evaluation import bootstrap_macro_auc, compare_runs


def gold_acceptance_metrics(truth, pred_b20, pred_b21, *, b20_checkpoint, b21_checkpoint, encoder_sha256, n_bootstrap, seed):
    b20_eval = bootstrap_macro_auc(truth, pred_b20, n_bootstrap=n_bootstrap, seed=seed + 221)
    b21_eval = bootstrap_macro_auc(truth, pred_b21, n_bootstrap=n_bootstrap, seed=seed + 222)
    replay_delta = require_b20_replay_sanity(b20_eval.macro_auc)
    paired = compare_runs(truth, pred_b20, pred_b21, n_bootstrap=n_bootstrap, seed=seed + 223)
    return {
        "variant": B21_GOLD_ACCEPTANCE_VARIANT,
        "one_gold_look_consumed": True,
        "working_model_before_comparison": "B20_crop_only_joint_focus",
        "automatic_working_model_update": False,
        "promotion_rule": PROMOTION_RULE,
        "scientific_superiority_rule": SCIENTIFIC_SUPERIORITY_RULE,
        "canonical_b20_macro_auc": B20_CANONICAL_GOLD_MACRO_AUC,
        "canonical_b20_epoch": B20_CANONICAL_EPOCH,
        "b20_replay_sanity_tolerance": B20_REPLAY_SANITY_TOLERANCE,
        "b20_replay": {**b20_eval.to_dict(), "checkpoint": str(Path(b20_checkpoint)), "replay_minus_canonical": replay_delta, "crop_stage": "post_resize_224"},
        "b21_candidate": {**b21_eval.to_dict(), "checkpoint": str(Path(b21_checkpoint)), "candidate_minus_canonical_b20": float(b21_eval.macro_auc - B20_CANONICAL_GOLD_MACRO_AUC), "crop_stage": "native_array_pre_resize"},
        "paired_b21_minus_b20_replay": {"raw_difference": float(b21_eval.macro_auc - b20_eval.macro_auc), **paired, "n_bootstrap": int(n_bootstrap)},
        "promotion_rule_passed": promotion_decision(b21_eval.macro_auc),
        "scientific_superiority_supported": scientific_superiority_decision(paired["ci_lower"]),
        "encoder_sha256": encoder_sha256,
        "target_level_role": "descriptive only; forbidden for target mixing, retuning, or promotion decisions",
        "interpretation": "Single predeclared acceptance look after B21 was frozen on weak-v2. The 58 studies were reused during historical B20 development, so this is not pristine independent validation.",
    }
