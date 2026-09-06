"""Score B54 on the 58 expert studies, twice, from one checkpoint.

This is the measurement the run was built to produce. Everything else B54
touched — the teacher, the vocabulary — is confounded with everything else in a
single run. The spacing conditioning is not, because it is zero-initialised and
switchable: evaluate the same trained weights with it on and with it off, and
the difference is the spacing effect, with no second run and no matched control
to arrange.

## The loading order inverts

At training the conditioning is installed **after** the pretrained checkpoint is
loaded, because that checkpoint does not have the key and a strict load would
raise. Here it is the other way round: the B54 checkpoint *does* carry
`spacing_conditioning.projection.weight`, so the conditioning must exist
**before** the load or the same strict load raises on an unexpected key.

Both orders are correct and they are opposite. That is exactly the kind of thing
that is obvious for a week and then costs an afternoon, so it is stated here and
pinned by a test.

## What is reused, and what is not

The inference recipe is B50's, unchanged: the same 90% native crop, the
constant-area native-aspect resize, three centre offsets, and `_score_split`
doing the work. B54 adds one flag to that function, defaulting off, so B50's own
evaluation is untouched. Re-deriving the recipe here would risk measuring a
different thing and calling it a comparison.

## How to read the answer

The 58 studies resolve to roughly ±0.03. A spacing delta smaller than that is
not evidence of anything, in either direction — and it is measured on the same
weights, same studies, same crops, differing only in whether one learned vector
is added, so it is a far tighter comparison than any two-run delta on this
surface. `VETO_DELTA = -0.020` remains the threshold for the run as a whole
against B52's 0.678247.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from .b42_constant_area_aspect_sparse_mil import require_b42_contract
from .b50_adapted_hierarchy_eval import _score_split, expert58_surface
from .b52_competition_training import B52_EXPERIMENT
from .b54_spacing_conditioned_mil import B54SpacingConditionedMIL, conditioning_has_moved
from .b54_spacing_run import attach_spacing, install_spacing_conditioning, spacing_summary
from .b12_variable_series import build_variable_series_index
from .constants import TARGETS
from .data import backfill_series_metadata, load_series_csv
from .evaluation import macro_auc_from_arrays
from .phase9_matched_supervision_training import load_phase9_checkpoint
from .runtime import resolve_runtime
from .spacing_conditioning import SpacingConditioning

B54_EVAL_VERSION = "b54_expert58_eval_v1"
B54_EXPERT58_ROOT = "runs/085_B54/expert58"

#: B52's own Expert-58 macro, the reference this run is read against.
B52_EXPERT58_MACRO = 0.678247

#: The surface is 58 studies. Anything smaller than this is not a finding.
EXPERT58_RESOLUTION = 0.03

#: Declared before the run, as every mechanism decision in this project is.
VETO_DELTA = -0.020


def load_b54_checkpoint(path: str | Path, *, base_checkpoint: str | Path, device):
    """Rebuild the trained B54 model, conditioning included.

    The conditioning is installed **before** `load_state_dict`, the opposite of
    the training order, because the saved state dict contains its weight. Strict
    loading is kept deliberately: it is what proves the checkpoint really was
    trained with the conditioning rather than without it.
    """
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if payload.get("experiment") != B52_EXPERIMENT:
        raise ValueError(f"{path} is not a B52-lineage checkpoint")
    spacing_state = payload.get("spacing", {})
    if not spacing_state.get("enabled"):
        raise ValueError(
            f"{path} was trained without the spacing conditioning; there is no "
            "ablation to run on it"
        )

    base_model, _ = load_phase9_checkpoint(
        Path(base_checkpoint).resolve(), expected_arm="llm_fill", device="cpu"
    )
    model_state = payload["model_state"]
    model = B54SpacingConditionedMIL(
        base_model,
        grid_size=int(model_state["grid_size"]),
        top_k=int(model_state["top_k"]),
        temperature=float(model_state["temperature"]),
        encoder_trainable_stages=int(
            model_state.get("encoder_trainable_stages", 1)
        ),
        adapt_hierarchy=bool(payload.get("adapt_hierarchy", True)),
    )
    install_spacing_conditioning(model.base)
    model.base.load_state_dict(payload["base_state"], strict=True)
    model.head.load_state_dict(payload["head_state"], strict=True)
    model.eval().to(device)

    if not conditioning_has_moved(model):
        raise RuntimeError(
            "the loaded conditioning is still exactly zero; this checkpoint was "
            "never trained to use the spacing and the ablation would be empty"
        )
    return model, payload


def set_enabled(model: torch.nn.Module, enabled: bool) -> int:
    """Switch every conditioning site, and refuse a model that has none."""
    sites = [m for m in model.modules() if isinstance(m, SpacingConditioning)]
    if not sites:
        raise RuntimeError("no SpacingConditioning to switch")
    for site in sites:
        site.set_enabled(bool(enabled))
    return len(sites)


def _scores(surface: dict, key: str = "combined") -> dict:
    macro, per_target = macro_auc_from_arrays(surface["target"], surface[key])
    return {
        "macro_auc": float(macro),
        "per_target_auc": {
            target: float(value) for target, value in zip(TARGETS, per_target)
        },
    }


def evaluate_b54(
    config: dict,
    *,
    data_root: str | Path,
    checkpoint: str | Path,
    base_checkpoint: str | Path,
    spacing_geometry_csv: str | Path,
    out_root: str | Path = B54_EXPERT58_ROOT,
) -> dict:
    """Score the 58 experts with the conditioning on, then off."""
    settings = dict(config)
    settings["data_root"] = str(Path(data_root).resolve())
    crop_policy = require_b42_contract(settings)
    settings["b7_eval_batch_size"] = 1
    root = Path(settings["data_root"])
    runtime = resolve_runtime(settings)

    uids, targets, weights = expert58_surface(root, settings)

    series = load_series_csv(root / settings.get("train_series_csv", "train_series.csv"))
    series, _ = backfill_series_metadata(series, root, split="train")
    series_index = build_variable_series_index(series, uids)
    attached = attach_spacing(
        series_index,
        series_geometry_csv=spacing_geometry_csv,
        data_root=root,
        split="train",
    )
    print(f"[B54 eval] spacing {attached}", flush=True)
    if attached["unresolved"]:
        raise RuntimeError(
            f"{attached['unresolved']} expert-surface series have no spacing; "
            "the ablation would be partly silent"
        )

    model, payload = load_b54_checkpoint(
        checkpoint, base_checkpoint=base_checkpoint, device=runtime.device
    )

    arms: dict[str, dict] = {}
    for label, enabled in (("spacing_on", True), ("spacing_off", False)):
        sites = set_enabled(model, enabled)
        surface = _score_split(
            model=model,
            config=settings,
            root=root,
            runtime=runtime,
            uids=uids,
            targets=targets,
            weights=weights,
            series_index=series_index,
            crop_policy=crop_policy,
            label=f"B54 {label}",
            spacing=True,
        )
        arms[label] = {"conditioning_sites": sites, **_scores(surface)}
        print(f"[B54 eval] {label} macro {arms[label]['macro_auc']:.6f}", flush=True)

    delta = arms["spacing_on"]["macro_auc"] - arms["spacing_off"]["macro_auc"]
    against_b52 = arms["spacing_on"]["macro_auc"] - B52_EXPERT58_MACRO
    result = {
        "version": B54_EVAL_VERSION,
        "checkpoint": str(Path(checkpoint).resolve()),
        "expert_studies": len(uids),
        "spacing_summary": spacing_summary(series_index),
        "arms": arms,
        "spacing_delta": float(delta),
        "spacing_delta_per_target": {
            target: float(
                arms["spacing_on"]["per_target_auc"][target]
                - arms["spacing_off"]["per_target_auc"][target]
            )
            for target in TARGETS
        },
        "b52_expert58_macro": B52_EXPERT58_MACRO,
        "delta_against_b52": float(against_b52),
        "veto_delta": VETO_DELTA,
        "vetoed_against_b52": bool(against_b52 <= VETO_DELTA),
        "resolution": EXPERT58_RESOLUTION,
        "spacing_delta_is_resolvable": bool(abs(delta) >= EXPERT58_RESOLUTION),
        "training_spacing_state": payload.get("spacing", {}),
        "reading": (
            "The two arms share weights, studies and crops and differ only in "
            "whether one learned vector is added, so their delta is far tighter "
            "than any two-run comparison on 58 studies. The delta against B52 is "
            "not: it carries the teacher change and the population as well."
        ),
    }

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "expert58.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _report(result)
    return result


def _report(result: dict) -> None:
    on = result["arms"]["spacing_on"]
    off = result["arms"]["spacing_off"]
    print()
    print(f"  Expert-58, {result['expert_studies']} studies")
    print(f"    spacing on           {on['macro_auc']:.6f}")
    print(f"    spacing off          {off['macro_auc']:.6f}")
    print(f"    the spacing effect   {result['spacing_delta']:+.6f}")
    print(
        "      resolvable on this surface"
        if result["spacing_delta_is_resolvable"]
        else f"      below the {result['resolution']} this surface resolves"
    )
    print()
    print(f"    B52 reference        {result['b52_expert58_macro']:.6f}")
    print(f"    B54 against it       {result['delta_against_b52']:+.6f}")
    print(
        f"    VETOED at {result['veto_delta']}"
        if result["vetoed_against_b52"]
        else f"    within the {result['veto_delta']} veto threshold"
    )
    print()
    print("    per target, spacing on minus off")
    for target, value in sorted(
        result["spacing_delta_per_target"].items(), key=lambda kv: -abs(kv[1])
    ):
        print(f"      {target:<20}{value:+.4f}")
    print()
    print(
        "  Individual targets on 58 studies carry intervals far too wide to read\n"
        "  alone. Read the macro."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        "Score B54 on the 58 experts with the spacing conditioning on and off"
    )
    parser.add_argument("--config", default="config/b42_constant_area_aspect_sparse.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--spacing-geometry-csv", required=True)
    parser.add_argument("--out-root", default=B54_EXPERT58_ROOT)
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    evaluate_b54(
        config,
        data_root=args.data_root,
        checkpoint=args.checkpoint,
        base_checkpoint=args.base_checkpoint,
        spacing_geometry_csv=args.spacing_geometry_csv,
        out_root=args.out_root,
    )


if __name__ == "__main__":
    main()
