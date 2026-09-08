"""B55: the competition endpoint that fixes what the field measured.

Three changes to B52, and nothing else:

```text
teacher      graded against the competition's severity rubric, so a trace
             effusion trains as negative -- which is what the graders wrote
crop         130 mm of knee, not 90% of whatever the scanner produced
laterality   one canonical orientation, correctly per plane
resolution   336 reference area, not 448
```

The teacher change arrives through `--labels-root`, so it is not wired here:
run `b55_rubric_teacher` first, read `rubric_changes.csv`, and point this at
the export it wrote. That keeps the two decisions separable even though the run
bundles them.

## This is an endpoint, not an experiment

Three changes at once cannot attribute a result to any one of them, which is
B52's limitation and is accepted here for B52's reason: this project scores
`0.716` against public notebooks at `0.899`, and that is not a gap a single
careful ablation closes. B55 is recorded as a competition endpoint. Nothing
here licenses tuning the crop, the resolution or the rubric against its score.

## What it inherits rather than restates

Everything else is B52's, called rather than copied: `train_b52` itself, with
`dataset_factory` and `identity` hooks added for exactly this purpose. The
encoder, the sparse head, the study hierarchy, the parameter groups, the
schedule, the split, the seed, the resume and the supervision guard are
untouched, so a difference between B52 and B55 is the three changes above and
not a divergence that crept into a copied loop.

## The guard that will fire, and why that is correct

`_report_only_surface` asserts an exact usable-cell count. Downgrading a
positive to a negative does not change how many cells are usable -- a negative
cell is still supervision -- so the count should be unchanged and the guard
should pass. If it fires, the rubric did something other than downgrade, and
the run should stop until that is understood. Declare the new count with
`--expected-supervision-cells` only after reading why it moved.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .b42_constant_area_aspect_sparse_mil import B42ConstantAreaAspectDataset
from .b53_augmented_training import AugmentationPolicy, B53AugmentedDataset
from .b52_competition_training import (
    B52_DEFAULT_ENCODER_LR_SCALE,
    B52_DEFAULT_ENCODER_STAGES,
    B52_DEFAULT_HIERARCHY_LR_SCALE,
    B52_FULL_TRAIN_SPLITS,
    B52_SEED,
    B52_TRAIN_SPLIT,
    _read_config,
    train_b52,
)
from .b55_physical_geometry import (
    B55_REFERENCE_AREA,
    B55_REFERENCE_SIDE,
    B55PhysicalGeometryDataset,
)
from .loader_throughput import add_worker_argument
from .physical_crop import CANONICAL_SIDE, CROP_MM

B55_EXPERIMENT = "B55_PHYSICAL_GEOMETRY"
B55_VERSION = "b55_physical_geometry_v1"
B55_RUN_ROOT = "runs/089_Experiment_B55_physical_geometry"

#: B52's, because B55 changes geometry and supervision, not the schedule.
B55_DEFAULT_EPOCHS = 8


class B55AugmentedDataset(B53AugmentedDataset, B55PhysicalGeometryDataset):
    """B55's geometry, with B53's augmentation applied on top of it.

    The composition works because the two subclass B42 at different points:
    B55 overrides `_load_b42`, which produces the tensor, and B53 augments in
    `__getitem__` *after* `super().__getitem__()` has produced it. The method
    resolution order runs B53's `__getitem__`, which calls down into B42's,
    which calls `self._load_b42` -- B55's. Neither had to know about the other.
    """


def b55_dataset_factory(
    crop_mm: float,
    reference_area: int,
    *,
    policy=None,
    seed: int = B52_SEED,
):
    """A `_build_dataset` with B55's geometry baked in.

    Returns a callable with B52's exact signature, so `train_b52` needs to know
    nothing about B55.

    `policy` adds B53's augmentation, and **only to the training surface**.
    Augmenting validation would change what the score measures rather than what
    the model learns, which is why the contract carries `train`.
    """

    def build(
        uids,
        index,
        dataset_config,
        crop_policy,
        targets,
        weights,
        spacing=False,
        train=False,
    ):
        if spacing:
            raise ValueError(
                "B55 does not carry the spacing conditioning: it was tested at "
                "12.65% of its own sum and measured -0.004908 on Expert-58, so "
                "it is closed. Run without --spacing-geometry-csv."
            )
        shared = dict(
            crop_focus_policy=crop_policy,
            center_offsets=(0,),
            targets=targets,
            weights=weights,
            crop_mm=float(crop_mm),
            reference_area=int(reference_area),
        )
        if train and policy is not None and not policy.is_disabled():
            return B55AugmentedDataset(
                uids, index, dataset_config, policy=policy, seed=int(seed), **shared
            )
        return B55PhysicalGeometryDataset(uids, index, dataset_config, **shared)

    return build


def train_b55(
    config: dict,
    *,
    data_root: str | Path,
    labels_root: str | Path,
    series_policy_path: str | Path,
    base_checkpoint: str | Path,
    domain_split: str | Path,
    epochs: int = B55_DEFAULT_EPOCHS,
    crop_mm: float = CROP_MM,
    reference_side: int = B55_REFERENCE_SIDE,
    all_data: bool = True,
    augment: bool = False,
    expected_supervision_cells: int | None = None,
    num_workers: int | None = None,
    seed: int = B52_SEED,
    out_root: str | Path = B55_RUN_ROOT,
    preflight_only: bool = False,
):
    """B52's run with B55's geometry. Everything else is B52's, called not copied."""
    reference_area = int(reference_side) * int(reference_side)
    policy = AugmentationPolicy.from_config(config) if augment else None
    if augment and policy.is_disabled():
        raise ValueError(
            "--augment was asked for but every configured value is zero. Check "
            "b7_rotation_deg and friends in the config."
        )
    print(f"[B55] augmentation {policy.active() if policy else 'off'}", flush=True)
    print(
        f"[B55] crop={crop_mm:g} mm  canonical_side={CANONICAL_SIDE}  "
        f"reference={reference_side}^2 (B42 used 448^2)",
        flush=True,
    )
    print(f"[B55] labels {Path(labels_root).resolve()}", flush=True)

    return train_b52(
        config,
        data_root=data_root,
        labels_root=labels_root,
        series_policy_path=series_policy_path,
        base_checkpoint=base_checkpoint,
        domain_split=domain_split,
        epochs=int(epochs),
        encoder_trainable_stages=B52_DEFAULT_ENCODER_STAGES,
        encoder_lr_scale=B52_DEFAULT_ENCODER_LR_SCALE,
        hierarchy_lr_scale=B52_DEFAULT_HIERARCHY_LR_SCALE,
        train_splits=B52_FULL_TRAIN_SPLITS if all_data else (B52_TRAIN_SPLIT,),
        expected_supervision_cells=expected_supervision_cells,
        num_workers=num_workers,
        dataset_factory=b55_dataset_factory(
            crop_mm, reference_area, policy=policy, seed=int(seed)
        ),
        # Passed, not defaulted. train_b52's `augment` defaults to True, so
        # omitting it wrote augmentation_enabled: true into a checkpoint that
        # had trained on undistorted pixels -- which is precisely the B52
        # failure B53 exists to correct, reappearing here.
        augment=bool(augment),
        identity={"experiment": B55_EXPERIMENT, "version": B55_VERSION},
        extra={
            "b55_geometry": {
                "crop_mm": float(crop_mm),
                "reference_side": int(reference_side),
                "reference_area": reference_area,
                "canonical_side": CANONICAL_SIDE,
            },
            # The policy itself, not a boolean: a flag that says True while the
            # pixels are untouched is the thing this project has already paid
            # 27 hours to learn about.
            "b55_augmentation": policy.active() if policy else None,
        },
        seed=int(seed),
        out_root=out_root,
        preflight_only=preflight_only,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="B55: physical crop, canonical laterality, 336 reference area"
    )
    parser.add_argument("--config", default="config/b42_constant_area_aspect_sparse.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--labels-root",
        required=True,
        help="the export b55_rubric_teacher wrote, not the ungraded one",
    )
    parser.add_argument("--series-policy", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--domain-split", required=True)
    parser.add_argument("--epochs", type=int, default=B55_DEFAULT_EPOCHS)
    parser.add_argument("--crop-mm", type=float, default=CROP_MM)
    parser.add_argument("--reference-side", type=int, default=B55_REFERENCE_SIDE)
    parser.add_argument(
        "--augment",
        action="store_true",
        help=(
            "apply B53's augmentation to the training surface only. Off by "
            "default until B53 reports whether it helps at these settings."
        ),
    )
    parser.add_argument(
        "--gate-split",
        action="store_true",
        help="train on the 1,447-study gate rows only, instead of all data",
    )
    parser.add_argument("--expected-supervision-cells", type=int, default=None)
    parser.add_argument("--seed", type=int, default=B52_SEED)
    add_worker_argument(parser)
    parser.add_argument("--out-root", default=B55_RUN_ROOT)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    train_b55(
        _read_config(args.config),
        data_root=args.data_root,
        labels_root=args.labels_root,
        series_policy_path=args.series_policy,
        base_checkpoint=args.base_checkpoint,
        domain_split=args.domain_split,
        epochs=args.epochs,
        crop_mm=args.crop_mm,
        reference_side=args.reference_side,
        all_data=not args.gate_split,
        augment=args.augment,
        expected_supervision_cells=args.expected_supervision_cells,
        num_workers=args.num_workers,
        seed=args.seed,
        out_root=args.out_root,
        preflight_only=args.preflight_only,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "B55_DEFAULT_EPOCHS",
    "B55_EXPERIMENT",
    "B55_RUN_ROOT",
    "B55_VERSION",
    "b55_dataset_factory",
    "train_b55",
]
