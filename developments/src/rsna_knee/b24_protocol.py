"""B24 frozen protocol: does B23 supervision beat B6 supervision?

B24 is the first experiment that changes the labels and nothing else. The model
is B20 exactly -- same weak-v2-safe B16 encoder, frozen; same 90% post-resize
crop; same 224 input; same hierarchy; same optimiser; same fixed epoch-2
endpoint. Only the supervision source differs between the two arms.

## The design problem, and why there are two weak surfaces

A development surface built from a labeller favours models trained by that same
labeller, by construction. Scoring B24 only on the B23 holdout would be
circular; scoring it only on the B6 weak-v2 holdout would hand the advantage to
the control. Neither is neutral.

B24 therefore scores both arms on **both** weak surfaces and reads the
asymmetry:

```text
                         B23 holdout        weak-v2 (B6) holdout
B24 (B23 labels)         expected to win    the informative test
B6 control               ---                expected to win
```

Each arm winning on its own labeller's surface says nothing. **B24 winning on
B6's own surface is the strongest evidence available short of expert truth**,
because it means B23 supervision produced a model that reproduces the B6
teacher better than training on B6 labels did.

The only labeller-neutral surface is the 58 expert studies, and B22 measured
what that surface can resolve: a 0.0439 swing within a single run. It gets
exactly one predeclared look, and the promotion threshold is set so that a
difference smaller than the surface can resolve cannot promote anything.

## Matched studies, not matched cells

Both arms train on the **same studies** so the batch sequence, the optimiser
trajectory and the series exposure are identical. What differs is which cells
inside those studies carry supervision, and what state they carry -- which is
precisely the B23 hypothesis, since B6 discards 64% of the cells.

A `full` surface variant, where each arm uses its own active-study set, tests
the wider proposition that B23 also activates studies B6 misses entirely. It is
deferred: it changes the batch count between arms and so is not a single-
variable comparison.
"""
from __future__ import annotations

import json
from pathlib import Path

B24_EXPERIMENT = "B24_supervision_source"
B24_CONTROL_VARIANT = "b24_b6_supervision_matched_fixed_e2_v1"
B24_CANDIDATE_VARIANT = "b24_b23_supervision_matched_fixed_e2_v1"

MODE_CONTROL = "b6_control"
MODE_CANDIDATE = "b23_candidate"
MODES = (MODE_CONTROL, MODE_CANDIDATE)

# Frozen recipe, inherited from B20 and unchanged.
B24_FIXED_EPOCHS = 2
B24_SCHEDULER_HORIZON = 5
B24_CROP_FRACTION = 0.90
B24_IMAGE_SIZE = 224
B24_BATCH_SIZE = 2
B24_HEAD_LR = 1e-4
B24_ENCODER_LR = 0.0
B24_TTA_OFFSETS = (-1, 0, 1)

# Surface policy. `matched` is frozen for B24-v1.
SURFACE_MATCHED = "matched"
SURFACE_FULL = "full"

# Predeclared promotion thresholds on the single gold look. B22 measured a
# within-run swing of 0.0439 on this surface and the bootstrap implies a macro
# SE near 0.0250, so a bare point-estimate win is not evidence of anything.
GOLD_PROMOTION_MIN_PROBABILITY = 0.95
B20_REPLAY_TOLERANCE = 0.005


def mode_identity(mode: str) -> tuple[str, str]:
    if mode == MODE_CONTROL:
        return B24_CONTROL_VARIANT, "B6 v1.2.1 regex supervision"
    if mode == MODE_CANDIDATE:
        return B24_CANDIDATE_VARIANT, "B23 local-LLM supervision"
    raise ValueError(f"mode must be one of {MODES}")


def require_b24_contract(config: dict) -> None:
    """Refuse any deviation from the B20 recipe.

    B24's whole claim is that only the labels changed. A drifted learning rate
    or epoch count would make the comparison meaningless, so the contract is
    enforced rather than documented.
    """
    expected = {
        "b7_epochs": B24_FIXED_EPOCHS,
        "b7_image_size": B24_IMAGE_SIZE,
        "b7_batch_size": B24_BATCH_SIZE,
        "b7_n_slices": 16,
        "b7_transformer_layers": 2,
        "b7_transformer_heads": 8,
        "b7_pathology_layers": 1,
        "b12_1_series_pool_heads": 8,
        "seed": 2026,
    }
    for key, want in expected.items():
        if int(config.get(key, want)) != want:
            raise ValueError(f"B24 freezes {key}={want}")

    if float(config.get("b7_head_lr", B24_HEAD_LR)) != B24_HEAD_LR:
        raise ValueError(f"B24 freezes b7_head_lr={B24_HEAD_LR}")
    if float(config.get("b7_encoder_lr", B24_ENCODER_LR)) != B24_ENCODER_LR:
        raise ValueError("B24 keeps the encoder frozen")
    if not bool(config.get("b17_encoder_frozen", True)):
        raise ValueError("B24 keeps the encoder frozen")
    if bool(config.get("b18_expert_selection", False)):
        raise ValueError("B24 fixes epoch 2 in advance; no expert checkpoint selection")
    if str(config.get("b24_surface", SURFACE_MATCHED)) != SURFACE_MATCHED:
        raise ValueError("B24-v1 freezes the matched-study surface")


def require_passed_labeller_gate(path: str | Path) -> dict:
    """B24 may only run once B23 has been shown to beat B6 as a labeller.

    Training on labels that failed their own audit would waste a GPU run and
    produce a result nobody could interpret.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    gate = payload.get("labeller_gate") or payload.get("gate")
    if gate is None:
        raise ValueError(
            f"{path} carries no labeller gate; run rsna-knee-b23-audit and "
            "rsna-knee-b23-split first"
        )
    if not bool(gate.get("passed", False)):
        raise ValueError(
            "B23 labeller gate did not pass; B24 must not train on labels that "
            "are not established as better than B6.\n"
            + "\n".join(f"  {r}" for r in gate.get("reasons", []))
        )
    return gate


def cross_labeller_verdict(
    *,
    candidate_on_b23: float,
    control_on_b23: float,
    candidate_on_weak_v2: float,
    control_on_weak_v2: float,
) -> dict:
    """Read the two weak surfaces together.

    Each surface favours the arm trained by its own labeller, so only the
    cross-surface pattern is informative.
    """
    won_own = candidate_on_b23 > control_on_b23
    won_other = candidate_on_weak_v2 > control_on_weak_v2
    if won_own and won_other:
        strength = "strong"
        reading = (
            "B24 wins on B6's own surface as well as its own: B23 supervision "
            "reproduces the B6 teacher better than training on B6 labels did"
        )
    elif won_own and not won_other:
        strength = "uninformative"
        reading = (
            "each arm wins on its own labeller's surface, which is the expected "
            "result under no real difference"
        )
    elif not won_own:
        strength = "adverse"
        reading = (
            "B24 loses on the surface built from its own labels, which is "
            "evidence against the B23 supervision itself"
        )
    return {
        "candidate_won_own_surface": bool(won_own),
        "candidate_won_control_surface": bool(won_other),
        "strength": strength,
        "reading": reading,
        "note": (
            "weak surfaces measure teacher agreement, not expert truth; B15 and "
            "B21 both showed a weak-surface gain need not carry to gold"
        ),
    }


def gold_promotion_decision(
    *, paired_median: float, probability_candidate_better: float
) -> dict:
    """The single predeclared gold rule."""
    passed = bool(
        paired_median > 0.0
        and probability_candidate_better >= GOLD_PROMOTION_MIN_PROBABILITY
    )
    return {
        "promoted": passed,
        "paired_median": float(paired_median),
        "probability_candidate_better": float(probability_candidate_better),
        "required_probability": GOLD_PROMOTION_MIN_PROBABILITY,
        "rule": "paired median > 0 AND P(B24 > B20) >= 0.95",
        "interpretation": (
            "one predeclared look at a repeatedly reused 58-study surface; not "
            "independent validation. Hidden competition evaluation remains the "
            "only independent signal."
        ),
    }
