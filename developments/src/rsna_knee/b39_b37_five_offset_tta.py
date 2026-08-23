"""B39: five-offset inference refinement of the completed B37 endpoint.

B39 is intentionally an inference-only successor.  It does not train, alter, or
replace B37's fixed-E2 checkpoint.  It averages the exact B37 sparse-MIL model
over five symmetric deterministic through-plane centre offsets so local evidence
is less sensitive to a pathology falling between the original three views.

This is a new hidden-test candidate, not a post-hoc edit of B37.  No labels,
thresholds, blend weights, or Expert-58 results enter this module.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from .b7_weak_supervision import _read_config
from .b37_highres_sparse_submission import (
    B37_SUBMISSION_MAX_HOURS,
    _require_sparse_mil_submission_contract,
    generate_b37_sparse_mil_submission,
)

B39_VERSION = "b39_b37_five_offset_tta_v1"
B39_NUMBERED_CONTAINER = "runs/074_Experiment_B39_b37_five_offset_tta"
B39_RUN_ROOT = f"{B39_NUMBERED_CONTAINER}/b39_b37_five_offset_tta"
B39_TTA_OFFSETS = (-2, -1, 0, 1, 2)
B39_RUNTIME_BUDGET_HOURS = 8.25
B39_RUNTIME_RESERVE_MINUTES = 45.0
B39_SUBMISSION_EXPERIMENT = (
    "B39_b37_five_offset_tta_hidden_test_inference"
)


def require_b39_five_offset_contract(config: dict) -> dict:
    """Require the predeclared B39 inference-only recipe."""
    policy = _require_sparse_mil_submission_contract(
        config,
        expected_offsets=B39_TTA_OFFSETS,
        endpoint_name="B39",
    )
    budget = float(config.get("runtime_budget_hours", B39_RUNTIME_BUDGET_HOURS))
    reserve = float(
        config.get("runtime_reserve_minutes", B39_RUNTIME_RESERVE_MINUTES)
    )
    if not math.isclose(
        budget,
        B39_RUNTIME_BUDGET_HOURS,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"B39 freezes runtime_budget_hours={B39_RUNTIME_BUDGET_HOURS}"
        )
    if not math.isclose(
        reserve,
        B39_RUNTIME_RESERVE_MINUTES,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"B39 freezes runtime_reserve_minutes={B39_RUNTIME_RESERVE_MINUTES}"
        )
    return policy


def generate_b39_submission(
    config: dict,
    *,
    data_root: str | Path,
    checkpoint: str | Path,
    base_checkpoint: str | Path,
    out_path: str | Path = "submission.csv",
    preflight_only: bool = False,
) -> Path | None:
    """Write a five-offset probability submission from the frozen B37 model."""
    config = dict(config)
    require_b39_five_offset_contract(config)
    return generate_b37_sparse_mil_submission(
        config,
        data_root=data_root,
        checkpoint=checkpoint,
        base_checkpoint=base_checkpoint,
        out_path=out_path,
        tta_offsets=B39_TTA_OFFSETS,
        submission_experiment=B39_SUBMISSION_EXPERIMENT,
        submission_version=B39_VERSION,
        endpoint_name="B39",
        min_reserve_minutes=B39_RUNTIME_RESERVE_MINUTES,
        preflight_only=bool(preflight_only),
        governance=(
            "Prospective B39 inference-only successor: exact frozen B37 fixed-E2 "
            "combined sparse-MIL checkpoint averaged over predeclared five "
            "symmetric centre offsets [-2,-1,0,1,2]. No training, labels, "
            "thresholds, blending, or post-hoc target-specific tuning is used. "
            "Do not alter this recipe after observing hidden competition evidence."
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        "Generate B39 five-offset inference from the frozen B37 checkpoint"
    )
    parser.add_argument(
        "--config",
        default="config/b39_b37_five_offset_tta.yaml",
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="run the largest-series five-view memory probe and exit",
    )
    args = parser.parse_args()
    config = dict(_read_config(args.config))
    generate_b39_submission(
        config,
        data_root=args.data_root,
        checkpoint=args.checkpoint,
        base_checkpoint=args.base_checkpoint,
        out_path=args.out,
        preflight_only=bool(args.preflight_only),
    )


if __name__ == "__main__":
    main()
