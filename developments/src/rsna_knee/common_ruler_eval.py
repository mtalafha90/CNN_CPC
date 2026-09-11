"""Score several checkpoints on one set of labels, so they can be compared.

## The problem this exists for

B55 regrades the teacher. That changes the *validation* labels as well as the
training ones, so B55's macro AUC and B52's are computed against different
answer keys. The difference between them would be mostly the ruler, and no
amount of care in reading it would recover the model comparison.

The fix is not to stop changing the teacher. It is to score every checkpoint
against **one** labels root:

```text
each model      its own geometry -- 448 fractional for B52 and B53,
                336 physical for B55. That is the model's input contract and
                changing it would measure a different model.

one ruler       the same labels, the same 548 unseen-scanner studies, the
                same weights. Choose which teacher is the ruler; what matters
                is that it is the same for everyone.
```

Same eyes, same answer key. The difference is then the model.

## Which labels root to use as the ruler

Either, and the choice should be stated rather than defaulted into:

* **The old teacher** puts every model on the scale B52 and B53 were selected
  on, so their published numbers are comparable with the new one.
* **The regraded teacher** uses the answer key closer to the competition's
  severity rule. No historical numbers exist on that scale.

**Neither isolates a change.** B55's weights were trained on regraded labels
*and* B55's geometry; scoring it against the old teacher scores a
differently-trained model on a familiar scale rather than measuring the
geometry. Attribution needs one change per run, which B55 deliberately gave up.

Both rulers are also report-derived, and the competition's labels are neither:
two MSK radiologists reading images, borderline findings graded negative. The
regraded teacher is the closer proxy. It is still a proxy.

Running both is cheap -- 548 studies, no training -- and the pair is more
informative than either. One ruler per invocation: run it twice with different
`--labels-root` values rather than mixing two answer keys into one table.

## What it does not do

It does not select anything. Every checkpoint here was already chosen by its own
run on its own surface; rescoring them on a common ruler is a *comparison*, not
a second selection, and treating the winner as promoted would be exactly the
post-hoc selection the archive forbids. The output says so.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .b7_weak_supervision import make_b7_dataset_config, target_balance_multipliers
from .b12_training import _load_series_policy
from .b12_variable_series import audit_variable_series_surface
from .b42_constant_area_aspect_sparse_mil import collate_b42, require_b42_contract
from .b48_global_conditioned_sparse_training import (
    _indices_for_split,
    _report_only_surface,
)
from .b50_adapted_hierarchy_training import load_b50_selection_gate
from .b52_competition_training import (
    B52_EXPERIMENT,
    B52_PRIMARY_SPLIT,
    _build_dataset,
    _read_config,
    evaluate_split,
)
from .b53_augmented_training import B53_EXPERIMENT
from .b55_physical_geometry_training import (
    B55_EXPERIMENT,
    b55_dataset_factory,
)
from .data import backfill_series_metadata, load_series_csv
from .loader_throughput import (
    add_worker_argument,
    apply_worker_override,
    loader_kwargs_with_sharing,
)
from .runtime import resolve_runtime

COMMON_RULER_VERSION = "common_ruler_eval_v1"

#: Which dataset each experiment was trained to see. A checkpoint whose
#: experiment is not here is refused rather than scored through the wrong
#: geometry, which would look like a bad model instead of a bad comparison.
GEOMETRY_BY_EXPERIMENT = {
    B52_EXPERIMENT: "b42",
    B53_EXPERIMENT: "b42",
    B55_EXPERIMENT: "b55",
}


def geometry_for(payload: dict) -> str:
    """Which input contract this checkpoint was trained under."""
    experiment = str(payload.get("experiment", ""))
    if experiment not in GEOMETRY_BY_EXPERIMENT:
        raise ValueError(
            f"{experiment!r} has no recorded geometry. Scoring it through "
            "another run's geometry would measure a different model and report "
            "the difference as a worse one. Add it to GEOMETRY_BY_EXPERIMENT "
            "once you know which dataset it was trained to see."
        )
    return GEOMETRY_BY_EXPERIMENT[experiment]


def per_target_table(scores: dict) -> dict:
    """The per-target AUCs, named and JSON-safe.

    `macro_auc` returns them **keyed by target name**, not as a list. Zipping
    that dict with TARGETS iterates its keys, so `float()` is handed `'ACL'`
    and the run dies with `could not convert string to float` -- after every
    study has already been through the model. Reading `.items()` is the fix,
    and this is a named function so a test can hold `macro_auc`'s real return
    against it rather than inspecting the source of `score_one`.
    """
    return {
        str(name): float(value)
        for name, value in dict(scores["per_target_auc"]).items()
    }


def b55_geometry_from(payload: dict) -> dict:
    """The crop and the resolution this B55 checkpoint was actually trained at.

    `train_b55` records them under `b55_geometry`, because that is the one
    place they exist: the model's own `model_state` carries the architecture,
    not the dataset, so a crop of 150 mm and a 224 reference side leave no
    trace in it at all. Reading `model_state` and falling back to the defaults
    -- which is what this did first -- silently scored such a run at 130 mm and
    336, through eyes it had never been trained to use, and reported the
    resulting loss as a worse model.

    A B55 checkpoint with no recorded geometry is refused rather than guessed
    at. `train_b55` has written the field since the run existed, so a payload
    missing it is not an old checkpoint; it is one this tool cannot describe.
    """
    recorded = payload.get("b55_geometry")
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError(
            "this B55 checkpoint records no `b55_geometry`, so the crop and "
            "the reference side it was trained at are unknown. Scoring it at "
            "the defaults would measure a different model and report the "
            "difference as a worse one. Re-run it with a `train_b55` that "
            "writes the field, or add the geometry to the payload by hand "
            "once you know what it was."
        )

    crop_mm = float(recorded["crop_mm"])
    side = int(recorded["reference_side"])
    area = int(recorded.get("reference_area", side * side))
    if area != side * side:
        raise ValueError(
            f"the checkpoint says reference_side={side} and "
            f"reference_area={area}, which do not agree ({side * side}). One "
            "of them is wrong and guessing which would change the geometry."
        )
    return {"crop_mm": crop_mm, "reference_side": side, "reference_area": area}


def dataset_factory_for(payload: dict):
    """The builder that reproduces this checkpoint's own input geometry."""
    if geometry_for(payload) == "b42":
        return _build_dataset

    geometry = b55_geometry_from(payload)
    return b55_dataset_factory(geometry["crop_mm"], geometry["reference_area"])


def load_model(payload_path: Path, base_checkpoint: Path, device):
    """Rebuild whichever B52-lineage model this checkpoint holds."""
    from .b50_adapted_hierarchy_mil import B50AdaptedHierarchySparseMILResidual
    from .phase9_matched_supervision_training import load_phase9_checkpoint

    payload = torch.load(str(payload_path), map_location="cpu", weights_only=False)
    model_state = payload["model_state"]
    base, _ = load_phase9_checkpoint(
        Path(base_checkpoint).resolve(), expected_arm="llm_fill", device="cpu"
    )
    model = B50AdaptedHierarchySparseMILResidual(
        base,
        grid_size=int(model_state["grid_size"]),
        top_k=int(model_state["top_k"]),
        temperature=float(model_state["temperature"]),
        encoder_trainable_stages=int(model_state.get("encoder_trainable_stages", 1)),
        encoder_chunk_size=int(model_state.get("encoder_chunk_size", 4)),
        adapt_hierarchy=bool(payload.get("adapt_hierarchy", True)),
    )
    model.base.load_state_dict(payload["base_state"], strict=True)
    model.head.load_state_dict(payload["head_state"], strict=True)
    return model.eval().to(device), payload


@torch.no_grad()
def predict_split(model, runtime, loader, multiplier_t, aux_weight: float):
    """`evaluate_split`, keeping the probabilities it computes and discards.

    Deliberately not a change to `evaluate_split`: that function is on the
    training path of every B52-lineage run and returning more from it would
    invite a caller to unpack it wrongly. This is the same loop with the
    predictions returned, and `macro_auc` called on the same arrays so the
    score here and the score there cannot disagree.

    **The decorator is part of that sameness.** Without it every forward pass
    keeps its activations alive for a backward pass that never comes. On this
    model that is the whole encoder over every slice of a study, and the
    16 GB card runs out of memory -- during scoring, not training, which is
    the last place anyone looks. `torch.no_grad` rather than
    `torch.inference_mode` for one reason only: it is what `evaluate_split`
    uses, and these two loops are supposed to be the same loop.
    """
    from .b42_constant_area_aspect_sparse_training import _losses, _move_study
    from .b37_highres_sparse_training import _trim_host_memory
    from .b52_competition_training import macro_auc

    was_training = model.training
    model.eval()
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []

    for items in loader:
        for item in items:
            tensors = _move_study(item, runtime.device)
            out, _total, _combined, _local = _losses(
                model, runtime, tensors, multiplier_t, aux_weight
            )
            probabilities.append(
                torch.sigmoid(out.logits.detach().float()).cpu().numpy().reshape(-1)
            )
            targets.append(item["target"].numpy().reshape(-1))
            weights.append(item["weight"].numpy().reshape(-1))
        _trim_host_memory()

    model.train(was_training)
    stacked = np.stack(probabilities)
    target_array, weight_array = np.stack(targets), np.stack(weights)
    scores = macro_auc(target_array, weight_array, stacked)
    scores["studies"] = len(probabilities)
    return stacked, target_array, weight_array, scores


def score_one(
    checkpoint: str | Path,
    *,
    config: dict,
    data_root: str | Path,
    labels_root: str | Path,
    series_policy_path: str | Path,
    base_checkpoint: str | Path,
    domain_split: str | Path,
    surface_cells: int | None = None,
    num_workers: int | None = None,
    center_offset: int = 0,
    with_predictions: bool = False,
) -> dict:
    """One checkpoint, one labels root, the 548 unseen-scanner studies.

    `center_offset` shifts every slice centre together, which is what a TTA
    view is. Averaging the probabilities from several offsets reproduces what
    the submission path does, without any change to the model or the dataset.

    `with_predictions` returns the per-study probabilities alongside the score,
    which is what an ensemble needs and what `evaluate_split` computes and then
    discards."""
    settings = dict(config)
    settings["data_root"] = str(Path(data_root).resolve())
    apply_worker_override(settings, num_workers)
    runtime = resolve_runtime(settings)
    root = Path(settings["data_root"])

    # Stage markers, because the alternative is what this tool shipped with:
    # one line printed after a whole checkpoint had been scored, so a hang in
    # the surface builder, the DICOM backfill and the loader all looked
    # identical from outside -- silence.
    def stage(message: str) -> None:
        print(f"[ruler]   {message}", flush=True)

    stage(f"labels {Path(labels_root).resolve()}")
    _, domain_rows, _ = load_b50_selection_gate(domain_split)

    # The surface builder needs the base checkpoint's own payload, so the model
    # is loaded first and its base reused rather than read twice.
    from .phase9_matched_supervision_training import load_phase9_checkpoint

    stage("reading the base checkpoint")
    _base_model, base_payload = load_phase9_checkpoint(
        Path(base_checkpoint).resolve(), expected_arm="llm_fill", device="cpu"
    )
    stage("building the supervision surface")
    (
        _train,
        uids,
        targets,
        weights,
        _lookup,
        _confidence,
        _fill_policy,
        _fill_audit,
        surface,
    ) = _report_only_surface(
        data_root=root,
        labels_root=labels_root,
        config=settings,
        domain_rows=domain_rows,
        base_payload=base_payload,
        # The guard still enforces: `None` means the frozen 34,010. That is
        # right for a ruler -- two label sets with different cell counts are
        # not the same ruler in the sense that matters, and the operator should
        # say so knowingly rather than have the comparison quietly widen.
        # Declare a regraded ruler's count with --expected-supervision-cells.
        expected_cells=int(surface_cells) if surface_cells else None,
    )
    indices = _indices_for_split(uids, domain_rows, B52_PRIMARY_SPLIT)
    valid_uids = [uids[i] for i in indices]
    stage(
        f"surface ok: {int(surface.get('usable_cells', -1)):,} usable cells, "
        f"{len(valid_uids)} studies in {B52_PRIMARY_SPLIT}"
    )

    series_policy = _load_series_policy(series_policy_path)
    if not series_policy:
        raise ValueError("a series policy is required")
    series = load_series_csv(root / settings.get("train_series_csv", "train_series.csv"))
    # Slow on a cold page cache: it opens DICOMs to repair missing metadata.
    stage("backfilling series metadata from DICOM")
    series, _ = backfill_series_metadata(series, root, split="train")
    _summary, valid_index = audit_variable_series_surface(series, valid_uids)

    crop_policy = require_b42_contract(settings)
    valid_config = make_b7_dataset_config(settings, root, train=False)
    valid_config.tta_center_offsets = ()

    stage("loading the model")
    model, payload = load_model(
        Path(checkpoint), Path(base_checkpoint), runtime.device
    )
    build = dataset_factory_for(payload)
    dataset = build(
        valid_uids,
        valid_index,
        valid_config,
        crop_policy,
        targets[indices],
        weights[indices],
    )
    if int(center_offset) != 0:
        dataset.center_offsets = (int(center_offset),)
    loader = DataLoader(
        dataset,
        batch_size=int(settings.get("b42_effective_batch", 2)),
        shuffle=False,
        drop_last=False,
        collate_fn=collate_b42,
        **loader_kwargs_with_sharing(runtime, seed=0),
    )
    multiplier = target_balance_multipliers(weights[indices])
    multiplier_t = torch.as_tensor(
        multiplier, dtype=torch.float32, device=runtime.device
    )
    aux_weight = float(settings.get("b37_local_aux_weight", 1.0))

    kwargs = loader_kwargs_with_sharing(runtime, seed=0)
    stage(
        f"scoring {len(valid_uids)} studies through geometry "
        f"{geometry_for(payload)} with {int(kwargs.get('num_workers', 0))} workers"
    )

    if with_predictions:
        probabilities, scored_targets, scored_weights, scores = predict_split(
            model, runtime, loader, multiplier_t, aux_weight
        )
    else:
        scores = evaluate_split(model, runtime, loader, multiplier_t, aux_weight)
        probabilities = scored_targets = scored_weights = None

    result = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "experiment": payload.get("experiment"),
        "version": payload.get("version"),
        "geometry": geometry_for(payload),
        # The numbers, not just the family name. A reader comparing two rows
        # can then see that they were scored through the eyes each was
        # trained with, rather than having to trust that they were.
        "geometry_used": (
            b55_geometry_from(payload) if geometry_for(payload) == "b55" else None
        ),
        "studies": len(valid_uids),
        "macro_auc": float(scores["macro_auc"]),
        "per_target_auc": per_target_table(scores),
        "supervision_cells": int(surface.get("usable_cells", -1)),
        "center_offset": int(center_offset),
    }
    if with_predictions:
        result["probabilities"] = probabilities
        result["targets"] = scored_targets
        result["weights"] = scored_weights

    release_model(model)
    return result


def release_model(model) -> None:
    """Give the card back before the next checkpoint asks for it.

    Each `score_one` builds a whole B50 hierarchy on the GPU. Python frees the
    object when it goes out of scope, but CUDA's caching allocator keeps the
    blocks, so scoring three checkpoints in one process holds three models'
    worth of memory on a 16 GB card that is also driving a desktop. Emptying
    the cache between them costs a few milliseconds and removes a failure that
    only appears on the third row of the table.
    """
    try:
        model.to("cpu")
    except Exception:  # pragma: no cover - a model already off the device
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def report(result: dict) -> None:
    """The comparison, and what it is not."""
    print()
    print(f"  Common ruler: {result['labels_root']}")
    print(f"  {result['studies']} studies from {B52_PRIMARY_SPLIT}")
    print()
    rows = sorted(result["scored"], key=lambda r: -r["macro_auc"])
    width = max((len(str(r["experiment"])) for r in rows), default=10)
    for row in rows:
        print(
            f"    {str(row['experiment']):<{width}}  {row['macro_auc']:.6f}  "
            f"geometry={row['geometry']}"
        )
    if len(rows) > 1:
        spread = rows[0]["macro_auc"] - rows[-1]["macro_auc"]
        print()
        print(f"    spread {spread:+.6f}")
    print()
    print(
        "  Same labels, same studies, each model's own geometry -- so the\n"
        "  difference is the model. This is a comparison, not a selection:\n"
        "  every checkpoint here was already chosen on its own surface, and\n"
        "  promoting the winner from this table would be the post-hoc\n"
        "  selection the archive forbids."
    )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score several checkpoints against one labels root"
    )
    parser.add_argument("--config", default="config/b42_constant_area_aspect_sparse.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--labels-root",
        required=True,
        help="the ruler. Run once per teacher you want to compare on.",
    )
    parser.add_argument("--series-policy", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--domain-split", required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="repeat for each checkpoint to compare",
    )
    parser.add_argument("--expected-supervision-cells", type=int, default=None)
    parser.add_argument("--out-json", default=None)
    add_worker_argument(parser)
    args = parser.parse_args()

    config = _read_config(args.config)
    scored = []

    def collected() -> dict:
        return {
            "version": COMMON_RULER_VERSION,
            "labels_root": str(Path(args.labels_root).resolve()),
            "split": B52_PRIMARY_SPLIT,
            "studies": scored[0]["studies"] if scored else 0,
            "scored": scored,
            "checkpoints_requested": len(args.checkpoint),
            "reading": (
                "One labels root, one study set, each model's own geometry. A "
                "comparison, not a selection."
            ),
        }

    def save() -> None:
        """Written after every checkpoint, not only at the end.

        A run that dies or hangs on the third model used to lose the first two
        as well, and each one is a scoring pass over 548 studies. The file is
        complete JSON at every point; `checkpoints_requested` beside the length
        of `scored` says whether it is the whole table.
        """
        if not args.out_json:
            return
        path = Path(args.out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(collected(), indent=2), encoding="utf-8")

    total = len(args.checkpoint)
    for position, path in enumerate(args.checkpoint, start=1):
        print(f"[ruler] {position}/{total} {path}", flush=True)
        row = score_one(
            path,
            config=config,
            data_root=args.data_root,
            labels_root=args.labels_root,
            series_policy_path=args.series_policy,
            base_checkpoint=args.base_checkpoint,
            domain_split=args.domain_split,
            surface_cells=args.expected_supervision_cells,
            num_workers=args.num_workers,
        )
        print(f"[ruler] {row['experiment']} {row['macro_auc']:.6f}", flush=True)
        scored.append(row)
        save()

    report(collected())


if __name__ == "__main__":
    main()


__all__ = [
    "COMMON_RULER_VERSION",
    "GEOMETRY_BY_EXPERIMENT",
    "b55_geometry_from",
    "dataset_factory_for",
    "geometry_for",
    "load_model",
    "per_target_table",
    "predict_split",
    "release_model",
    "report",
    "score_one",
]
