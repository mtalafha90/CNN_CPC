"""Submit the B54 endpoint through B42's proven dual-T4 inference path.

B54 is B52's run with a rebuilt teacher, the B6 v1.3.1 vocabulary, and one
extra parameter: a zero-initialised spacing term inside the study base. Its
checkpoint declares itself as B52 -- `experiment` is `B52_COMPETITION_FULL_
FINETUNE`, with B54's own details two levels down in `model_state` and
`spacing` -- so almost every one of `require_b52_endpoint`'s checks already
passes on it.

**Exactly one thing does not.** `load_b52_checkpoint` builds a
`B42ConstantAreaAspectSparseMILResidual` and then does

```python
model.base.load_state_dict(payload["base_state"], strict=True)
```

B54's `base_state` carries one key that model has never heard of,
`spacing_conditioning.projection.weight`, so the strict load raises
`Unexpected key(s) in state_dict`. That is the whole incompatibility. This
module installs the conditioning before the load and delegates the rest.

Strictness stays. Relaxing it to `strict=False` would make the same call
succeed by silently discarding the trained term, which is the failure this
project has repeatedly paid for: a switch that looks like it works and
measures nothing.

## The conditioning is switched off, but only when that is measured to be safe

The submitted forward pass runs with `enabled=False`, because B42's inference
loop does not carry a spacing: feeding one means a new per-series DICOM read
inside a hidden run whose exceptions are invisible, and B39, B41 and B51 each
passed a visible notebook and then threw on the hidden rerun. That path has
cost this project three submissions.

**Disabling a trained term is a change to the model, so it has to be earned.**
An earlier version of this module earned it with a sentence: the term is a
no-op, worth `+0.000152` on the 58 experts. That was true of B54 v1, whose
conditioning reached 0.07% of the sum it joined. It is false of v2, whose
scaled conditioning reaches **12.65%** — and the sentence would have shipped
v1's evidence attached to v2's weights, inside the one artefact nobody can
inspect during a hidden run.

So the claim is now checked against the checkpoint in hand, in two steps:

```text
conditioning_spread_ratio(payload)      measured from these weights, by the
                                        same code spacing_conditioning_probe
                                        uses -- not a recorded constant

below SPREAD_PRESENT (0.01)             a rounding error on its own sum;
                                        switching it off changes nothing and
                                        needs no further evidence

at or above it                          a real term. The disabled arm is
                                        refused unless an Expert-58 ablation
                                        *for this same checkpoint* shows the
                                        two arms agree within the surface's
                                        resolution.
```

The second case is the honest one to be strict about: it is exactly when a
convenient assumption would be most costly and least visible.
`assert_conditioning_disabled` then checks the loaded model rather than the
keyword that was passed, because `enabled=False` is one character from its
opposite and the failure would be silent.

## Read the population before you spend a submission

There are two B52 runs and they trained on different amounts of data:

```text
runs/086_...competition_full_finetune   0.802666   1,447 studies
runs/087_...b52_full_data               0.834998   3,801 studies   -> Kaggle 0.716
```

**B54 is 086's population, not 087's.** It trained on the same 1,447-study
B50 selection gate, and scored 0.804168 against 086's 0.802666. The standing
`0.716` was set by a model that saw 2,354 more studies.

So a B54 hidden score is not comparable with `0.716`, and a lower number would
be the expected result of the smaller population rather than evidence against
the teacher. The launcher prints both counts before it spends a GPU, and the
manifest records the comparison, because this is the kind of thing that is
obvious now and forgotten by the time a score appears.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from .b17_training import encoder_state_sha256
from .b35_training import sha256_file
from .b42_constant_area_aspect_sparse_mil import B42ConstantAreaAspectSparseMILResidual
from .b42_constant_area_aspect_sparse_submission_dualgpu_fast import (
    DEFAULT_FALLBACK_PROBABILITY,
    ON_UNREADABLE_FALLBACK,
    generate_b42_submission_dual_gpu_fast,
)
from .b52_competition_submission_dualgpu_fast import (
    b52_endpoint_manifest,
    require_b52_endpoint,
)
from .b54_spacing_run import B54_VERSION, install_spacing_conditioning
from .phase9_matched_supervision_training import load_phase9_checkpoint
from .spacing_conditioning import SpacingConditioning
from .spacing_conditioning_probe import SPREAD_PRESENT

B54_SUBMISSION_EXPERIMENT = "B54_teacher_rebuild_hidden_test_inference"

#: The population B54 trained on, and the one that set the standing 0.716.
#: Stated so the launcher can print the comparison rather than leaving it to
#: be remembered.
B52_LEADERBOARD_TRAINING_STUDIES = 3801
B52_LEADERBOARD_SCORE = 0.716

def conditioning_spread_ratio(payload: dict) -> float:
    """How much of the metadata sum's scale the trained term actually reaches.

    Delegates to `spacing_conditioning_probe`: same basis, same yardstick, same
    corpus endpoints. Reimplementing it here would let the gate and the probe
    disagree, and the gate would be the one nobody reads.

    The spread rather than the size, for the probe's own reason: a constant
    added to every series is a bias the study hierarchy absorbs, so only the
    movement across the corpus range can carry information.
    """
    from .spacing_conditioning_probe import (
        CONDITIONING_KEY,
        CORPUS_P05_MM,
        CORPUS_P95_MM,
        contributions,
        metadata_scale,
    )

    base_state = payload["base_state"]
    scale = float(payload.get("spacing", {}).get("conditioning_scale", 1.0))
    weight = base_state[CONDITIONING_KEY].detach().float() * scale

    thin, thick = contributions(weight, (CORPUS_P05_MM, CORPUS_P95_MM))
    typical = metadata_scale(base_state)["typical_metadata_norm"]
    if not typical:
        raise ValueError("the metadata embeddings are all zero; nothing to compare against")
    return float(float((thin - thick).norm()) / typical)


def load_ablation(path: str | Path, *, checkpoint: str | Path) -> dict:
    """Read an Expert-58 ablation and refuse one measured on another model.

    The whole point of the evidence is that it describes *these* weights. An
    ablation from a sibling run would look identical in every field that gets
    printed, which is precisely why the checkpoint identity is compared rather
    than assumed.
    """
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    measured = str(result.get("checkpoint", ""))
    wanted = str(Path(checkpoint).resolve())
    if measured != wanted:
        raise ValueError(
            f"{path} is an ablation of {measured or '<unrecorded>'}, not of "
            f"{wanted}; evidence from another run cannot license this one"
        )
    for key in ("spacing_delta", "resolution", "arms"):
        if key not in result:
            raise ValueError(f"{path} is not a b54_expert58_eval result; missing {key}")
    return result


def require_disabled_arm_is_safe(
    payload: dict, *, checkpoint: str | Path, ablation: str | Path | None
) -> dict:
    """Decide whether this checkpoint may be submitted with the spacing off.

    Returns the evidence for the manifest, or raises with the reason. A term
    too small to matter needs nothing; a real one needs a measurement on the
    same weights showing the two arms agree.
    """
    ratio = conditioning_spread_ratio(payload)
    evidence: dict = {
        "spread_over_metadata": ratio,
        "band_present": SPREAD_PRESENT,
        "term_is_negligible": bool(ratio < SPREAD_PRESENT),
    }

    if ratio < SPREAD_PRESENT:
        evidence["why_disabling_is_safe"] = (
            f"the trained term reaches {ratio:.4%} of the sum it joins, below "
            f"the {SPREAD_PRESENT:.0%} at which it could influence a ranking; "
            "switching it off is not a change to the model"
        )
        return evidence

    if ablation is None:
        raise ValueError(
            f"this checkpoint's spacing term reaches {ratio:.2%} of the sum it "
            f"joins, well above {SPREAD_PRESENT:.0%}, so switching it off is a "
            "real change to a trained model rather than a formality. B42's "
            "inference path supplies no spacing, so the disabled arm needs "
            "evidence: pass --expert58-ablation with the b54_expert58_eval "
            "result for this same checkpoint. If that ablation shows the two "
            "arms differ, the honest answer is that this model cannot be "
            "submitted through B42's loop as it stands."
        )

    measured = load_ablation(ablation, checkpoint=checkpoint)
    delta = float(measured["spacing_delta"])
    resolution = float(measured["resolution"])
    if abs(delta) >= resolution:
        raise ValueError(
            f"the Expert-58 ablation on this checkpoint moved the macro AUC by "
            f"{delta:+.6f}, at or beyond the {resolution} this surface "
            "resolves. The spacing term is doing measurable work, so the "
            "disabled arm is not the trained model and must not be submitted "
            "as one."
        )

    arms = measured["arms"]
    evidence.update(
        {
            "expert58_spacing_on": float(arms["spacing_on"]["macro_auc"]),
            "expert58_spacing_off": float(arms["spacing_off"]["macro_auc"]),
            "expert58_spacing_delta": delta,
            "expert58_resolution": resolution,
            "ablation_source": str(Path(ablation).resolve()),
            "why_disabling_is_safe": (
                f"the trained term reaches {ratio:.2%} of the sum it joins, but "
                f"the measured ablation on these same weights moved the "
                f"Expert-58 macro AUC by only {delta:+.6f} against a surface "
                f"that resolves to {resolution}"
            ),
        }
    )
    return evidence


def require_b54_endpoint(payload: dict) -> dict:
    """Every B52 check, plus the two that are only meaningful for B54.

    B54's checkpoint answers `experiment` with B52's name, so `require_b52_
    endpoint` applies to it in full and is called rather than reproduced. What
    it cannot know is whether the spacing conditioning is really there and
    really trained -- and a checkpoint that failed either would be a plain B52
    re-run being submitted under B54's name.
    """
    identity = require_b52_endpoint(payload)

    spacing = payload.get("spacing")
    if not isinstance(spacing, dict) or not spacing.get("enabled"):
        raise ValueError(
            "this checkpoint was not trained with the spacing conditioning, so "
            "it is a B52 run; submit it through b52_competition_submission_"
            "dualgpu_fast rather than under B54's name"
        )
    if not bool(spacing.get("conditioning_moved", False)):
        raise ValueError(
            "B54's conditioning never moved off zero during training, so this "
            "checkpoint is numerically a B52 run wearing B54's label"
        )
    if "spacing_conditioning.projection.weight" not in payload.get("base_state", {}):
        raise ValueError(
            "B54 base_state carries no spacing_conditioning weight; the audit "
            "and the weights disagree and one of them is wrong"
        )

    identity["spacing"] = {
        "conditioning_scale": float(spacing.get("conditioning_scale", 1.0)),
        "conditioning_moved": True,
        "conditioning_sites": spacing.get("conditioning_sites"),
        "optimiser": spacing.get("optimiser"),
    }
    return identity


def assert_conditioning_disabled(model: torch.nn.Module) -> int:
    """The submitted arm is the disabled one; check the model, not the argument.

    `install_spacing_conditioning(..., enabled=False)` is one keyword away from
    its opposite, and an enabled term with no spacing ever supplied would run
    silently: `condition_global_feature` returns its input unchanged when
    `series_spacing` is None. That is the same shape of silent no-op that made
    the first B54 ablation meaningless, so it is asserted rather than assumed.
    """
    sites = [m for m in model.modules() if isinstance(m, SpacingConditioning)]
    if not sites:
        raise RuntimeError(
            "no SpacingConditioning on the loaded model, so the trained weight "
            "was discarded rather than loaded"
        )
    live = [m for m in sites if m.enabled]
    if live:
        raise RuntimeError(
            f"{len(live)} of {len(sites)} conditioning sites are enabled, but "
            "B42's inference path supplies no spacing; the term would be a "
            "silent no-op presented as the trained configuration"
        )
    return len(sites)


def load_b54_checkpoint_for_submission(
    path: str | Path, *, base_checkpoint: str | Path, device
):
    """B52's reconstruction, with the conditioning installed before the load.

    Deliberately parallel to `load_b52_checkpoint` and deliberately not a call
    to it: the one line that differs is the one in the middle of it.

    The model class is B42's residual, exactly as B52's loader uses. B54 was
    trained as `B54SpacingConditionedMIL`, but that subclass adds no parameters
    of its own -- it overrides a forward to pass the spacing along -- and the
    spacing is not supplied here. With the conditioning disabled the two
    classes compute the same function, and B42's is the one B42's inference
    loop was measured against.
    """
    checkpoint = Path(path).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    identity = require_b54_endpoint(payload)

    base_path = Path(base_checkpoint).resolve()
    if sha256_file(base_path) != str(payload.get("base_checkpoint_sha256", "")):
        raise ValueError("B54 base checkpoint fingerprint mismatch")
    base, _ = load_phase9_checkpoint(base_path, expected_arm="llm_fill", device="cpu")

    sparse = identity["sparse_mil"]
    model = B42ConstantAreaAspectSparseMILResidual(
        base,
        grid_size=int(sparse["grid_size"]),
        top_k=int(sparse["top_k"]),
        temperature=float(sparse["temperature"]),
        encoder_trainable_stages=int(
            identity["encoder_finetune"]["encoder_trainable_stages"]
        ),
        encoder_chunk_size=int(identity["encoder_chunk_size"]),
    )
    # The one line B52's loader cannot have. Without it the strict load below
    # raises on `spacing_conditioning.projection.weight`; with `strict=False`
    # instead it would succeed and throw the trained term away.
    install_spacing_conditioning(
        model.base,
        enabled=False,
        scale=identity["spacing"]["conditioning_scale"],
    )
    model.base.load_state_dict(payload["base_state"], strict=True)
    model.head.load_state_dict(payload["head_state"], strict=True)
    model = model.to(device)
    model.eval()

    assert_conditioning_disabled(model)

    observed = encoder_state_sha256(model.base.encoder)
    expected = str(payload.get("encoder_sha256_final", ""))
    if observed != expected:
        raise RuntimeError("B54 reconstructed encoder fingerprint changed")
    return model, payload


def _load_b54_replica(checkpoint_path: Path, base_path: Path, device: torch.device):
    model, payload = load_b54_checkpoint_for_submission(
        checkpoint_path, base_checkpoint=base_path, device=device
    )
    model.eval()
    return model, payload


def b54_endpoint_manifest(payload: dict, *, spacing_evidence: dict | None = None) -> dict:
    """B52's manifest, corrected on the three things that are not true of B54.

    Reusing B52's is the point: everything about the training contract, the
    selection rule and the governance note applies unchanged. What must not
    carry over is the identity, the claim that the run's differences from B42
    are B52's, and any silence about which arm was submitted.
    """
    manifest = dict(b52_endpoint_manifest(payload))
    spacing = payload.get("spacing", {}) or {}
    trained = payload.get("training_studies")

    manifest.update(
        {
            "experiment": B54_SUBMISSION_EXPERIMENT,
            "version": B54_VERSION,
            "b52_manifest_version": manifest.get("version"),
            "spacing_conditioning": {
                "trained": True,
                "enabled_at_inference": False,
                "submitted_arm": "spacing_off",
                "why_disabled": (
                    "B42's inference path supplies no spacing. Whether that is "
                    "harmless is measured on these weights rather than assumed; "
                    "the evidence is in this block."
                ),
                "conditioning_scale": float(spacing.get("conditioning_scale", 1.0)),
                "conditioning_moved_during_training": bool(
                    spacing.get("conditioning_moved", False)
                ),
                # Measured from the checkpoint being submitted. Never a recorded
                # constant: an earlier version quoted B54 v1's numbers, which
                # would have described a term 180x smaller than v2's.
                "evidence": spacing_evidence,
            },
            "what_b54_changes_from_b52": [
                "teacher rebuilt on B52's rule: 34,842 cells, 3,513 more quoted",
                "B6 v1.3.1 report vocabulary",
                "a spacing term, trained and then disabled for this submission",
            ],
            "population_warning": {
                "b54_training_studies": trained,
                "leaderboard_reference_training_studies": (
                    B52_LEADERBOARD_TRAINING_STUDIES
                ),
                "leaderboard_reference_score": B52_LEADERBOARD_SCORE,
                "comparable": bool(trained == B52_LEADERBOARD_TRAINING_STUDIES),
                "reading": (
                    "The standing 0.716 was set by a B52 run trained on 3,801 "
                    "studies. A B54 trained on the 1,447-study selection gate is "
                    "a different population, so a lower hidden score is the "
                    "expected consequence of the smaller training set and is not "
                    "evidence about the teacher rebuild."
                ),
            },
        }
    )
    return manifest


def generate_b54_submission_dual_gpu_fast(
    config: dict,
    *,
    data_root: str | Path,
    checkpoint: str | Path,
    base_checkpoint: str | Path,
    expected_checkpoint_sha256: str,
    out_path: str | Path = "submission.csv",
    expert58_ablation: str | Path | None = None,
    on_unreadable: str = ON_UNREADABLE_FALLBACK,
    fallback_probability: float = DEFAULT_FALLBACK_PROBABILITY,
    stream_views: bool = True,
    abort_on_budget: bool = False,
) -> Path:
    """Run B42's dual-T4 path against the declared B54 checkpoint.

    Every execution default is B52's, for the same reasons B52 gives: the
    hidden-safe contract, telemetry-only budget projection, and a fallback row
    rather than an aborted run on one unreadable study out of 1,300.

    `expert58_ablation` is the `expert58.json` from `b54_expert58_eval` on this
    same checkpoint. It is needed only when the trained spacing term is large
    enough to matter, and it is checked before a GPU is spent rather than after.
    """
    path = Path(checkpoint).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"B54 submission checkpoint is missing: {path}")
    observed = sha256_file(path)
    if observed != expected_checkpoint_sha256:
        raise ValueError(
            "B54 hidden submission requires the declared checkpoint: "
            f"expected {expected_checkpoint_sha256}, got {observed}"
        )
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    identity = require_b54_endpoint(payload)

    # Before a GPU is spent, not after eight hours of it.
    evidence = require_disabled_arm_is_safe(
        payload, checkpoint=path, ablation=expert58_ablation
    )

    print(f"[B54 submit] {B54_SUBMISSION_EXPERIMENT}", flush=True)
    print(f"[B54 submit] checkpoint sha256 {observed}", flush=True)
    print(
        f"[B54 submit] epoch {identity['selected_epoch']} selected at "
        f"{identity['selection_value']}, trained on {identity['training_studies']} "
        f"studies, augmentation={identity['augmentation_enabled']}",
        flush=True,
    )
    print(f"[B54 submit] head geometry {identity['sparse_mil']}", flush=True)
    print(
        "[B54 submit] spacing conditioning trained then DISABLED for inference; "
        f"scale {identity['spacing']['conditioning_scale']:.6f}, term reaches "
        f"{evidence['spread_over_metadata']:.2%} of the sum it joins",
        flush=True,
    )
    print(f"[B54 submit] {evidence['why_disabling_is_safe']}", flush=True)

    trained = identity["training_studies"]
    if trained != B52_LEADERBOARD_TRAINING_STUDIES:
        print(
            f"[B54 submit] POPULATION: this model saw {trained} studies; the "
            f"standing {B52_LEADERBOARD_SCORE} was set by a B52 run that saw "
            f"{B52_LEADERBOARD_TRAINING_STUDIES}. The two scores are not "
            "comparable and a lower number here is expected.",
            flush=True,
        )

    print("[B54 submit] inference path is B42's, unchanged", flush=True)

    return generate_b42_submission_dual_gpu_fast(
        config,
        data_root=data_root,
        checkpoint=path,
        base_checkpoint=base_checkpoint,
        out_path=out_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        load_replica=_load_b54_replica,
        endpoint_manifest=lambda p: b54_endpoint_manifest(p, spacing_evidence=evidence),
        on_unreadable=on_unreadable,
        fallback_probability=fallback_probability,
        stream_views=stream_views,
        abort_on_budget=abort_on_budget,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit the B54 endpoint through B42's dual-T4 inference path"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--checkpoint", required=True, help="the B54 checkpoint, unmodified"
    )
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument(
        "--expected-checkpoint-sha256",
        required=True,
        help="sha256sum of the B54 checkpoint you intend to submit",
    )
    parser.add_argument("--out-path", default="submission.csv")
    parser.add_argument(
        "--expert58-ablation",
        default=None,
        help=(
            "expert58.json from b54_expert58_eval on this same checkpoint. "
            "Required when the trained spacing term is large enough to matter, "
            "because the submitted arm switches it off."
        ),
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    generate_b54_submission_dual_gpu_fast(
        dict(config),
        data_root=args.data_root,
        checkpoint=args.checkpoint,
        base_checkpoint=args.base_checkpoint,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        out_path=args.out_path,
        expert58_ablation=args.expert58_ablation,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "B54_SUBMISSION_EXPERIMENT",
    "assert_conditioning_disabled",
    "b54_endpoint_manifest",
    "conditioning_spread_ratio",
    "generate_b54_submission_dual_gpu_fast",
    "load_ablation",
    "load_b54_checkpoint_for_submission",
    "require_b54_endpoint",
    "require_disabled_arm_is_safe",
]
