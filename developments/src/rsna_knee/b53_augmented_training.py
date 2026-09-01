"""B53: the augmentation B52 configured but never applied.

B52 changed three things and reported three. Only two of them happened.

```text
the encoder trains                    real
cosine completes, best epoch kept     real
augmentation on                       inert
```

`train_b52` calls `make_b7_dataset_config(settings, root, train=True)`, which
faithfully sets `noise_std=0.02`, `rotation_deg=5.0` and the rest on the
`DatasetConfig`. The dataset it then builds never reads those fields.

Two independent reasons, either of which alone is enough:

```text
b37_highres_sparse_mil.py:184   super().__init__(..., train=False)   hard-coded
B42ConstantAreaAspectDataset    writes its own _load_b42, which goes straight
                                from DICOM to cropped triplets; _augment_mri is
                                never on that path at all
```

Measured, not inferred. Building the B42 dataset twice from the same DICOM
series, once with every augmentation set and once with none:

```text
two draws with augmentation ON, identical to each other : True
augmentation ON identical to augmentation OFF           : True
maximum absolute difference                             : 0.0
```

So B52's `+0.0395` and `+0.0719` came from two changes, not three, and the
standard remedy for a model memorising a few thousand studies has never once
been applied in this project.

## What B53 changes

One thing. The pixels the model trains on are distorted, using the values the
config has always carried and the operations `dataset._augment_mri` has always
defined. Everything else -- geometry, head, labels, loss, split, schedule,
learning rates, seed -- is B52's, imported from it rather than restated, so the
two runs differ in augmentation and nothing else.

## What it deliberately leaves alone

Two of the nine configured settings change *which slices are chosen*, not what
the chosen pixels look like: `center_jitter` and `train_gap_choices`. Slice
choice is the frozen B35 contract (`b35_centers` hard-codes `jitter=0`), and
moving it would be a second change in the same run. `--slice-jitter` exists for
a later experiment and defaults to `0`, which is off.

## The check that would have caught B52

`b53_preflight` draws the same study twice and refuses to start unless the two
tensors differ. A flag that sets a field nobody reads is not a hypothesis anyone
can test, and B52 ran for 27 hours on one.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF

from .b7_weak_supervision import (
    make_b7_dataset_config,
    seed_everything,
    target_balance_multipliers,
)
from .b12_training import _load_series_policy
from .b12_variable_series import audit_variable_series_surface
from .b13_training import B13_SERIES_SIGNATURE
from .b17_training import encoder_state_sha256
from .b35_training import _require_base_checkpoint, sha256_file
from .b37_highres_sparse_training import (
    B37_GRAD_CLIP,
    B37_HEAD_LR,
    B37_WEIGHT_DECAY,
    _format_memory_state,
    _memory_state,
    _trim_host_memory,
)
from .b42_constant_area_aspect_sparse_mil import (
    B42ConstantAreaAspectDataset,
    collate_b42,
    require_b42_contract,
)
from .b42_constant_area_aspect_sparse_training import (
    _batch_scales,
    _losses,
    _move_study,
)
from .b48_global_conditioned_sparse_training import (
    _config_sha256,
    _report_only_surface,
    _uid_sha256,
    b48_fill_artifacts,
)
from .b50_adapted_hierarchy_mil import B50AdaptedHierarchySparseMILResidual
from .b50_adapted_hierarchy_training import load_b50_selection_gate
from .b52_competition_training import (
    B52_CONSTRUCTION_SEED_OFFSET,
    B52_DEFAULT_ENCODER_LR_SCALE,
    B52_DEFAULT_ENCODER_STAGES,
    B52_DEFAULT_HIERARCHY_LR_SCALE,
    B52_FULL_TRAIN_SPLITS,
    B52_INHERITED_ENCODER_STAGES,
    B52_INHERITED_EPOCHS,
    B52_LOADER_SEED_OFFSET,
    B52_PRIMARY_SPLIT,
    B52_SEED,
    B52_TRAIN_SPLIT,
    _read_config,
    b52_parameter_groups,
    evaluate_split,
    select_train_and_validation,
)
from .constants import TARGETS
from .data import backfill_series_metadata, load_series_csv
from .encoder_finetune import MAX_TRAINABLE_STAGES
from .phase9_matched_supervision_training import load_phase9_checkpoint
from .runtime import make_scaler, resolve_runtime

B53_EXPERIMENT = "B53_AUGMENTATION_APPLIED"
B53_VERSION = "b53_augmentation_applied_v1"
B53_RUN_ROOT = "runs/087_Experiment_B53_augmentation_applied"
B53_CHECKPOINT_NAME = "b53_best_model.pt"

# B53 changes one thing against B52 and inherits every other value from it.
B53_DEFAULT_EPOCHS = 6
B53_SLICE_JITTER_DEFAULT = 0

# Seed offsets are B53's own, so its augmentation draws and loader order do not
# accidentally coincide with anything B52 did.
B53_AUGMENT_SEED_OFFSET = 53_000_003

# The B52 numbers B53 is measured against, on the same 548 unseen-scanner
# studies. Selection statistics, not effect sizes.
B52_GATE_SPLIT_MACRO_AUC = 0.802666
B52_ALL_DATA_MACRO_AUC = 0.834998


class AugmentationPolicy:
    """How hard to distort a training study.

    The defaults are read from the config rather than invented here: these nine
    keys have sat in `b42_constant_area_aspect_sparse.yaml` since B7 and were
    only ever applied to a dataset class this pipeline stopped using.
    """

    __slots__ = (
        "rotation_deg",
        "translate_frac",
        "scale_jitter",
        "gamma_jitter",
        "bias_field_strength",
        "noise_std",
        "slice_dropout",
    )

    def __init__(
        self,
        *,
        rotation_deg: float = 5.0,
        translate_frac: float = 0.03,
        scale_jitter: float = 0.05,
        gamma_jitter: float = 0.12,
        bias_field_strength: float = 0.08,
        noise_std: float = 0.02,
        slice_dropout: float = 0.08,
    ) -> None:
        for name, value in (
            ("rotation_deg", rotation_deg),
            ("translate_frac", translate_frac),
            ("scale_jitter", scale_jitter),
            ("gamma_jitter", gamma_jitter),
            ("bias_field_strength", bias_field_strength),
            ("noise_std", noise_std),
            ("slice_dropout", slice_dropout),
        ):
            if float(value) < 0:
                raise ValueError(f"{name} cannot be negative")
            setattr(self, name, float(value))
        if not 0 <= self.slice_dropout < 1:
            raise ValueError("slice_dropout must be in [0, 1)")

    @classmethod
    def from_config(cls, config: dict) -> "AugmentationPolicy":
        """Take every value from the frozen config, so none is invented here."""
        return cls(
            rotation_deg=float(config.get("b7_rotation_deg", 5.0)),
            translate_frac=float(config.get("b7_translate_frac", 0.03)),
            scale_jitter=float(config.get("b7_scale_jitter", 0.05)),
            gamma_jitter=float(config.get("b7_gamma_jitter", 0.12)),
            bias_field_strength=float(config.get("b7_bias_field_strength", 0.08)),
            noise_std=float(config.get("b7_noise_std", 0.02)),
            slice_dropout=float(config.get("b7_slice_dropout", 0.08)),
        )

    @classmethod
    def disabled(cls) -> "AugmentationPolicy":
        """Everything zero: what B52 actually ran, whatever its flag said."""
        return cls(
            rotation_deg=0.0,
            translate_frac=0.0,
            scale_jitter=0.0,
            gamma_jitter=0.0,
            bias_field_strength=0.0,
            noise_std=0.0,
            slice_dropout=0.0,
        )

    def to_dict(self) -> dict:
        return {name: float(getattr(self, name)) for name in self.__slots__}

    def active(self) -> dict:
        return {name: value for name, value in self.to_dict().items() if value > 0}

    def is_disabled(self) -> bool:
        return not self.active()


def augment_b42_series(
    series: torch.Tensor, policy: AugmentationPolicy, generator: torch.Generator
) -> torch.Tensor:
    """Distort one prepared B42 series of shape [slices, 3, height, width].

    The operations and their order are `dataset._augment_mri`'s, unchanged. What
    differs is where they run -- on the tensor B42 actually produces -- and that
    every draw comes from the generator passed in rather than from the global
    random state, so a run is reproducible and a DataLoader worker cannot repeat
    another worker's numbers.

    B42 pixels are percentile-normalised into [0, 1] by `_normalise_volume`, so
    the clamps here are the same clamps the original used and mean the same
    thing.
    """
    if series.ndim != 4 or int(series.shape[1]) != 3:
        raise ValueError(f"expected [slices,3,H,W], got {tuple(series.shape)}")

    def uniform(low: float, high: float) -> float:
        if high <= low:
            return low
        drawn = torch.rand((), generator=generator, dtype=torch.float32)
        return float(low + (high - low) * drawn)

    volume = series.float()
    slices, _channels, height, width = volume.shape

    # --- rotation, translation and scale, as one warp ----------------------
    # One interpolation rather than three, so the image is blurred once.
    if policy.rotation_deg > 0 or policy.translate_frac > 0 or policy.scale_jitter > 0:
        angle = uniform(-policy.rotation_deg, policy.rotation_deg)
        # The original scaled the shift by a square image_size. B42 series are
        # rectangles of roughly constant area, so each axis uses its own side --
        # otherwise a tall series would shift far further sideways than up.
        max_x = int(round(policy.translate_frac * width))
        max_y = int(round(policy.translate_frac * height))
        translate = [
            int(round(uniform(-max_x, max_x))) if max_x else 0,
            int(round(uniform(-max_y, max_y))) if max_y else 0,
        ]
        scale = 1.0 + uniform(-policy.scale_jitter, policy.scale_jitter)
        volume = TVF.affine(
            volume,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
        )

    # --- gamma -------------------------------------------------------------
    if policy.gamma_jitter > 0:
        gamma = 1.0 + uniform(-policy.gamma_jitter, policy.gamma_jitter)
        volume = volume.clamp(0, 1).pow(gamma)

    # --- smooth bias field -------------------------------------------------
    # A gentle tilt across the image, which is what an imperfect receive coil
    # produces. Clamped to [0.8, 1.2] exactly as the original clamped it.
    if policy.bias_field_strength > 0:
        yy = torch.linspace(-1, 1, height, dtype=volume.dtype).view(1, 1, height, 1)
        xx = torch.linspace(-1, 1, width, dtype=volume.dtype).view(1, 1, 1, width)
        ax = uniform(-policy.bias_field_strength, policy.bias_field_strength)
        ay = uniform(-policy.bias_field_strength, policy.bias_field_strength)
        field = (1 + ax * xx + ay * yy).clamp(0.8, 1.2)
        volume = (volume * field).clamp(0, 1)

    # --- noise -------------------------------------------------------------
    if policy.noise_std > 0:
        noise = torch.randn(volume.shape, generator=generator, dtype=volume.dtype)
        volume = (volume + policy.noise_std * noise).clamp(0, 1)

    # --- slice dropout -----------------------------------------------------
    if policy.slice_dropout > 0 and slices > 1:
        draw = torch.rand(slices, generator=generator, dtype=torch.float32)
        drop = draw < policy.slice_dropout
        # The original had no such guard, and at p=0.08 over 32 slices it would
        # essentially never fire. It costs nothing and the case it prevents --
        # a blank study still carrying a real label -- teaches the model
        # something false rather than nothing.
        if bool(drop.all()):
            drop[int(torch.argmax(draw))] = False
        volume = volume * (~drop).to(volume.dtype).view(slices, 1, 1, 1)

    return volume


class B53AugmentedDataset(B42ConstantAreaAspectDataset):
    """The B42 dataset, with the configured augmentation actually applied.

    Subclassed rather than edited. B42's geometry contract -- the 90% native
    crop, the constant-area resize, the 32 slice centres -- runs first and
    untouched; augmentation happens afterwards, on the tensor it produced. The
    validation dataset is a plain `B42ConstantAreaAspectDataset`, so the two
    surfaces stay exactly as comparable as they were.
    """

    def __init__(
        self,
        *args,
        policy: AugmentationPolicy | None = None,
        seed: int = B52_SEED,
        slice_jitter: int = B53_SLICE_JITTER_DEFAULT,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.policy = policy or AugmentationPolicy.disabled()
        self.augment_seed = int(seed) + B53_AUGMENT_SEED_OFFSET
        self.slice_jitter = int(slice_jitter)
        if self.slice_jitter < 0:
            raise ValueError("slice_jitter cannot be negative")
        self.epoch = 0
        self._draw: torch.Generator | None = None

    def set_epoch(self, epoch: int) -> None:
        """Give the next pass a different draw. The training loop calls this."""
        self.epoch = int(epoch)

    def _generator(self, index: int) -> torch.Generator:
        """A generator fixed by run seed, epoch and study position.

        Not shared state: a DataLoader worker holds its own copy of the dataset,
        so one generator on the instance would give every worker the same
        numbers and silently reduce the augmentation to one repeated draw.
        """
        generator = torch.Generator()
        generator.manual_seed(
            (self.augment_seed * 1_000_003 + self.epoch * 9_176 + index) % (2**31 - 1)
        )
        return generator

    def _load_b42(self, uid: str, series_uid: str, plane: str):
        """Optionally shift the slice centres before the frozen loader runs.

        Off by default. When on, it shifts every centre in a series together,
        which is what `center_offset` already means here, rather than reaching
        into the frozen `b35_centers`.
        """
        if self.slice_jitter <= 0 or self._draw is None:
            return super()._load_b42(uid, series_uid, plane)

        span = 2 * self.slice_jitter + 1
        shift = int(torch.randint(0, span, (1,), generator=self._draw).item()) - self.slice_jitter
        original = self.center_offsets
        self.center_offsets = tuple(offset + shift for offset in original)
        try:
            return super()._load_b42(uid, series_uid, plane)
        finally:
            self.center_offsets = original

    def __getitem__(self, index: int) -> dict:
        self._draw = self._generator(index)
        try:
            item = super().__getitem__(index)
        finally:
            draw = self._draw
            self._draw = None

        if self.policy.is_disabled():
            return item

        present = item["present"]
        volumes = item["volumes"]
        augmented = []
        for position, volume in enumerate(volumes):
            # A masked series is a zero placeholder the model already ignores.
            # Adding noise to it would turn a placeholder into something that is
            # no longer zero.
            if float(present[position]) <= 0:
                augmented.append(volume)
            else:
                augmented.append(augment_b42_series(volume, self.policy, draw))
        item["volumes"] = augmented
        return item


def _build_train_dataset(
    uids, index, dataset_config, crop_policy, targets, weights, *, policy, seed, slice_jitter
):
    return B53AugmentedDataset(
        uids,
        index,
        dataset_config,
        crop_focus_policy=crop_policy,
        center_offsets=(0,),
        targets=targets,
        weights=weights,
        policy=policy,
        seed=seed,
        slice_jitter=slice_jitter,
    )


def _build_valid_dataset(uids, index, dataset_config, crop_policy, targets, weights):
    """Validation is never augmented, so epochs stay comparable with each other."""
    return B42ConstantAreaAspectDataset(
        uids,
        index,
        dataset_config,
        crop_focus_policy=crop_policy,
        center_offsets=(0,),
        targets=targets,
        weights=weights,
    )


def verify_augmentation_reaches_pixels(dataset: B53AugmentedDataset) -> dict:
    """Draw the same study twice and confirm the pixels actually differ.

    This is the check B52 did not have. Its augmentation flag set fields on a
    config object that the dataset never read, and nothing in a 27-hour run
    would have told anyone. Reading two draws is the only way to know.
    """
    if not len(dataset):
        raise RuntimeError("cannot verify augmentation on an empty dataset")
    if dataset.policy.is_disabled():
        raise RuntimeError(
            "verify_augmentation_reaches_pixels was called with augmentation off"
        )

    dataset.set_epoch(1)
    first = dataset[0]
    dataset.set_epoch(2)
    second = dataset[0]

    live = [
        position
        for position in range(len(first["present"]))
        if float(first["present"][position]) > 0
    ]
    if not live:
        raise RuntimeError("the first study has no readable series to compare")

    differences = [
        float((first["volumes"][position] - second["volumes"][position]).abs().max())
        for position in live
    ]
    report = {
        "series_compared": len(live),
        "series_that_changed": int(sum(1 for value in differences if value > 0)),
        "max_absolute_difference": max(differences),
        "policy": dataset.policy.active(),
    }
    if report["series_that_changed"] == 0:
        raise RuntimeError(
            "B53 augmentation did not reach the pixels: two draws of the same "
            "study are identical. This is exactly the B52 failure, and the run "
            "would be B52 under another name."
        )

    # Reset, so the verification cannot change what epoch 1 trains on.
    dataset.set_epoch(0)
    return report


def b53_preflight(
    model, runtime, train_dataset, multiplier_t, aux_weight: float, scaler
) -> dict:
    """One forward and backward pass, plus the augmentation check."""
    augmentation = verify_augmentation_reaches_pixels(train_dataset)
    print(
        f"[B53 preflight] augmentation reaches the pixels: "
        f"{augmentation['series_that_changed']}/{augmentation['series_compared']} series "
        f"changed, max |diff| {augmentation['max_absolute_difference']:.6f}",
        flush=True,
    )

    items = [train_dataset[index] for index in range(min(2, len(train_dataset)))]
    for item, scale in zip(items, _batch_scales(items, multiplier_t.detach().cpu())):
        tensors = _move_study(item, runtime.device)
        _out, total, _combined, _local = _losses(
            model, runtime, tensors, multiplier_t, aux_weight
        )
        scaler.scale(total * float(scale)).backward()

    moved = sum(
        1
        for parameter in model.base.encoder.parameters()
        if parameter.requires_grad
        and parameter.grad is not None
        and torch.count_nonzero(parameter.grad) > 0
    )
    model.zero_grad(set_to_none=True)
    if moved == 0:
        raise RuntimeError("B53 preflight: no gradient reached the encoder")

    print(
        f"[B53 preflight] PASS encoder tensors with gradient={moved} "
        f"{_format_memory_state(_memory_state(runtime))}",
        flush=True,
    )
    return {"encoder_tensors_with_gradient": moved, "augmentation": augmentation}


def train_b53(
    config: dict,
    *,
    data_root: str | Path,
    labels_root: str | Path,
    series_policy_path: str | Path,
    base_checkpoint: str | Path,
    domain_split: str | Path,
    epochs: int = B53_DEFAULT_EPOCHS,
    encoder_trainable_stages: int = B52_DEFAULT_ENCODER_STAGES,
    encoder_lr_scale: float = B52_DEFAULT_ENCODER_LR_SCALE,
    hierarchy_lr_scale: float = B52_DEFAULT_HIERARCHY_LR_SCALE,
    augment: bool = True,
    slice_jitter: int = B53_SLICE_JITTER_DEFAULT,
    train_splits: tuple = (B52_TRAIN_SPLIT,),
    gradient_checkpointing: bool = True,
    seed: int = B52_SEED,
    out_root: str | Path = B53_RUN_ROOT,
    preflight_only: bool = False,
) -> Path | None:
    """B52's run with the augmentation actually applied, and nothing else moved."""
    settings = dict(config)
    settings["data_root"] = str(Path(data_root).resolve())
    settings["seed"] = int(seed)

    if not 1 <= int(encoder_trainable_stages) <= MAX_TRAINABLE_STAGES:
        raise ValueError(f"B53 trains the encoder; stages must be 1..{MAX_TRAINABLE_STAGES}")
    if int(epochs) < 1:
        raise ValueError("B53 needs at least one epoch")

    policy = AugmentationPolicy.from_config(settings) if augment else AugmentationPolicy.disabled()
    if augment and policy.is_disabled():
        raise ValueError(
            "B53 is the experiment in which augmentation is applied, but every "
            "configured value is zero. Check b7_rotation_deg and friends."
        )

    domain_payload, domain_rows, domain_meta = load_b50_selection_gate(domain_split)
    seed_everything(int(seed) + B52_CONSTRUCTION_SEED_OFFSET)
    runtime = resolve_runtime(settings)
    print(runtime.describe(), flush=True)
    print(
        f"[B53] epochs={epochs} encoder_stages={encoder_trainable_stages} "
        f"augment={augment} slice_jitter={slice_jitter}",
        flush=True,
    )
    print(f"[B53] augmentation: {policy.active() or 'none'}", flush=True)
    print(f"[B53] split sha={domain_meta['sha256']}", flush=True)

    base_path = Path(base_checkpoint).resolve()
    base_model, base_payload = load_phase9_checkpoint(
        base_path, expected_arm="llm_fill", device="cpu"
    )
    _require_base_checkpoint(base_payload)
    encoder_initial_sha = encoder_state_sha256(base_model.encoder)

    root = Path(settings["data_root"])
    expected_train_sha = str(domain_payload.get("source_train_csv_sha256", ""))
    if not expected_train_sha or sha256_file(
        root / settings.get("train_csv", "train.csv")
    ) != expected_train_sha:
        raise ValueError("B53 domain split source train.csv fingerprint mismatch")

    fill_artifacts = b48_fill_artifacts(labels_root)
    (
        _train,
        all_uids,
        all_targets,
        all_weights,
        _lookup,
        confidence,
        fill_policy,
        fill_audit,
        supervision,
    ) = _report_only_surface(
        data_root=root,
        labels_root=labels_root,
        config=settings,
        domain_rows=domain_rows,
        base_payload=base_payload,
    )

    train_indices, valid_indices, train_uids, valid_uids = select_train_and_validation(
        all_uids, domain_rows, tuple(train_splits)
    )
    print(
        f"[B53] training on {len(train_uids)} studies from {list(train_splits)}; "
        f"validating on {len(valid_uids)} from {B52_PRIMARY_SPLIT}",
        flush=True,
    )

    train_targets, train_weights = all_targets[train_indices], all_weights[train_indices]
    valid_targets, valid_weights = all_targets[valid_indices], all_weights[valid_indices]
    target_multiplier = target_balance_multipliers(train_weights)

    series_policy = _load_series_policy(series_policy_path)
    if (
        series_policy.get("series_summary", {}).get("series_signature_sha256")
        != B13_SERIES_SIGNATURE
    ):
        raise ValueError("B53 requires the frozen B12/B13 series policy")
    series = load_series_csv(root / settings.get("train_series_csv", "train_series.csv"))
    series, metadata_stats = backfill_series_metadata(series, root, split="train")
    _train_summary, train_index = audit_variable_series_surface(series, train_uids)
    _valid_summary, valid_index = audit_variable_series_surface(series, valid_uids)

    crop_policy = require_b42_contract(settings)

    # Both configs are built with train=False, and that is deliberate. The flag
    # sets fields this dataset does not read -- which is the whole finding --
    # so B53 leaves it alone rather than pretending it does something, and
    # applies the augmentation where it can actually be observed.
    train_config = make_b7_dataset_config(settings, root, train=False)
    train_config.tta_center_offsets = ()
    valid_config = make_b7_dataset_config(settings, root, train=False)
    valid_config.tta_center_offsets = ()

    train_dataset = _build_train_dataset(
        train_uids, train_index, train_config, crop_policy, train_targets, train_weights,
        policy=policy, seed=int(seed), slice_jitter=int(slice_jitter),
    )
    valid_dataset = _build_valid_dataset(
        valid_uids, valid_index, valid_config, crop_policy, valid_targets, valid_weights
    )

    batch_size = int(settings.get("b42_effective_batch", 2))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_b42,
        **runtime.loader_kwargs(seed=int(seed) + B52_LOADER_SEED_OFFSET),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_b42,
        **runtime.loader_kwargs(seed=int(seed) + B52_LOADER_SEED_OFFSET),
    )

    model = B50AdaptedHierarchySparseMILResidual(
        base_model,
        grid_size=int(settings["b37_grid_size"]),
        top_k=int(settings["b37_top_k"]),
        temperature=float(settings["b37_temperature"]),
        encoder_trainable_stages=int(encoder_trainable_stages),
        encoder_chunk_size=int(settings["b37_encoder_chunk_size"]),
        adapt_hierarchy=True,
    ).to(runtime.device)
    model.gradient_checkpointing = bool(gradient_checkpointing)
    model.train()

    trainable = model.trainable_parameter_summary()
    print(f"[B53] trainable={trainable}", flush=True)

    head_lr = float(settings.get("b37_head_lr", B37_HEAD_LR))
    groups = b52_parameter_groups(
        model,
        head_lr=head_lr,
        encoder_lr_scale=float(encoder_lr_scale),
        hierarchy_lr_scale=float(hierarchy_lr_scale),
    )
    for group in groups:
        print(
            f"[B53]   {group['name']:<16} lr={group['lr']:.3e} "
            f"params={sum(p.numel() for p in group['params']):,}",
            flush=True,
        )

    optimizer = torch.optim.AdamW(
        groups, weight_decay=float(settings.get("b37_weight_decay", B37_WEIGHT_DECAY))
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(epochs), eta_min=float(settings.get("b7_min_lr", 1e-6))
    )
    scaler = make_scaler(runtime)
    multiplier_cpu = torch.from_numpy(target_multiplier)
    multiplier_t = multiplier_cpu.to(runtime.device)
    aux_weight = float(settings["b37_local_aux_weight"])
    clip = float(settings.get("b37_grad_clip", B37_GRAD_CLIP))
    clipped = [p for group in groups for p in group["params"]]

    optimizer.zero_grad(set_to_none=True)
    preflight = b53_preflight(model, runtime, train_dataset, multiplier_t, aux_weight, scaler)
    optimizer.zero_grad(set_to_none=True)
    if preflight_only:
        return None

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out / B53_CHECKPOINT_NAME
    if checkpoint_path.exists():
        raise FileExistsError(f"B53 will not overwrite {checkpoint_path}")

    history: list[dict] = []
    best_macro = -float("inf")
    best_epoch = 0

    for epoch in range(1, int(epochs) + 1):
        started = time.monotonic()
        if runtime.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(runtime.device)
        # A different augmentation draw each epoch, reproducible from the seed.
        train_dataset.set_epoch(epoch)
        model.train()
        total_sum = 0.0
        batches = 0

        for items in train_loader:
            optimizer.zero_grad(set_to_none=True)
            scales = _batch_scales(items, multiplier_cpu)
            batch_total = 0.0
            for item, scale in zip(items, scales):
                tensors = _move_study(item, runtime.device)
                _out, total, _combined, _local = _losses(
                    model, runtime, tensors, multiplier_t, aux_weight
                )
                scaler.scale(total * float(scale)).backward()
                batch_total += float(total.detach().item()) * float(scale)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(clipped, clip)
            scaler.step(optimizer)
            scaler.update()
            total_sum += batch_total
            batches += 1
            _trim_host_memory()

        scheduler.step()
        scores = evaluate_split(model, runtime, valid_loader, multiplier_t, aux_weight)
        row = {
            "epoch": epoch,
            "train_loss": total_sum / max(batches, 1),
            "validation_loss": scores["loss"],
            "validation_macro_auc": scores["macro_auc"],
            "validation_per_target_auc": scores["per_target_auc"],
            "targets_defined": scores["targets_defined"],
            "learning_rates": [float(g["lr"]) for g in optimizer.param_groups],
            "epoch_minutes": round((time.monotonic() - started) / 60.0, 1),
        }
        history.append(row)
        print(
            f"[B53] E{epoch:>2} train={row['train_loss']:.6f} "
            f"val={row['validation_loss']:.6f} "
            f"macroAUC={row['validation_macro_auc']:.6f} "
            f"({row['epoch_minutes']} min)",
            flush=True,
        )

        if np.isfinite(row["validation_macro_auc"]) and row["validation_macro_auc"] > best_macro:
            best_macro = float(row["validation_macro_auc"])
            best_epoch = epoch
            payload = {
                "experiment": B53_EXPERIMENT,
                "version": B53_VERSION,
                "selected_epoch": epoch,
                "selection_metric": f"macro_auc on {B52_PRIMARY_SPLIT}",
                "selection_value": best_macro,
                "epochs_planned": int(epochs),
                "seed": int(seed),
                "encoder_trainable_stages": int(encoder_trainable_stages),
                "encoder_lr_scale": float(encoder_lr_scale),
                "hierarchy_lr_scale": float(hierarchy_lr_scale),
                "train_splits": list(train_splits),
                "gradient_checkpointing": bool(gradient_checkpointing),
                "head_lr": head_lr,
                # The one change, and the evidence that it happened. B52 wrote
                # augmentation_enabled: true while training on identical pixels
                # every epoch; a boolean nobody measured is what made that
                # possible, so B53 records the measurement instead.
                "augmentation_enabled": bool(augment),
                "augmentation_policy": policy.to_dict(),
                "augmentation_verified": preflight["augmentation"],
                "slice_jitter": int(slice_jitter),
                "changed_from_b52": {
                    "augmentation": [
                        "configured but never applied to the B42 dataset",
                        "applied to every present series, verified at preflight",
                    ],
                },
                "b52_reference": {
                    "gate_split_macro_auc": B52_GATE_SPLIT_MACRO_AUC,
                    "all_data_macro_auc": B52_ALL_DATA_MACRO_AUC,
                    "note": "same 548 unseen-scanner studies; selection statistics",
                },
                "base_checkpoint": str(base_path),
                "base_checkpoint_sha256": sha256_file(base_path),
                "base_state": model.base.state_dict(),
                "head_state": model.head.state_dict(),
                "model_state": model.state(),
                "encoder_sha256_initial": encoder_initial_sha,
                "encoder_sha256_final": encoder_state_sha256(model.base.encoder),
                "training_studies": len(train_uids),
                "validation_studies": len(valid_uids),
                "training_uids_sha256": _uid_sha256(train_uids),
                "gold_labels_used": False,
                "gold_studies_used_in_gradient": 0,
                "target_balance_multiplier": {
                    name: float(target_multiplier[index])
                    for index, name in enumerate(TARGETS)
                },
                "label_confidence": confidence,
                "fill_policy": fill_policy,
                "fill_audit": fill_audit,
                "fill_artifacts": fill_artifacts,
                "supervision": supervision,
                "series_policy_signature": B13_SERIES_SIGNATURE,
                "metadata_repair": metadata_stats,
                "domain_split_sha256": domain_meta["sha256"],
                "config_sha256": _config_sha256(settings),
                "source_sha256": {"training": sha256_file(Path(__file__))},
                "history": history,
                "governance": (
                    "B53 selects its checkpoint on a held-out report-labelled "
                    "split, which is competition practice and deliberately not "
                    "the frozen-endpoint policy the scientific line uses. Its "
                    "validation number is a selection statistic, not evidence of "
                    "an effect, and must not be quoted as one. It is comparable "
                    "with B52's number on the same split and with nothing else."
                ),
            }
            torch.save(payload, checkpoint_path)
            print(f"[B53]     new best at epoch {epoch}: {best_macro:.6f}", flush=True)

    if best_epoch == 0:
        raise RuntimeError("B53 finished without a usable validation score")

    (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"[B53] best epoch {best_epoch} macroAUC {best_macro:.6f}", flush=True)
    reference = (
        B52_ALL_DATA_MACRO_AUC
        if tuple(train_splits) == B52_FULL_TRAIN_SPLITS
        else B52_GATE_SPLIT_MACRO_AUC
    )
    print(
        f"[B53] against B52 on the same split: {best_macro - reference:+.6f} "
        f"({best_macro:.6f} vs {reference:.6f})",
        flush=True,
    )
    print(checkpoint_path, flush=True)
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser("Train B53: B52 with the augmentation applied")
    parser.add_argument("--config", default="config/b42_constant_area_aspect_sparse.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--labels-root", required=True)
    parser.add_argument("--series-policy", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--domain-split", required=True)
    parser.add_argument("--epochs", type=int, default=B53_DEFAULT_EPOCHS)
    parser.add_argument(
        "--encoder-stages", type=int, default=B52_DEFAULT_ENCODER_STAGES,
        help=f"1..{MAX_TRAINABLE_STAGES}; the frozen contract used {B52_INHERITED_ENCODER_STAGES}",
    )
    parser.add_argument("--encoder-lr-scale", type=float, default=B52_DEFAULT_ENCODER_LR_SCALE)
    parser.add_argument("--hierarchy-lr-scale", type=float, default=B52_DEFAULT_HIERARCHY_LR_SCALE)
    parser.add_argument(
        "--no-augment", action="store_true",
        help="turn the one change off, reproducing B52's actual behaviour",
    )
    parser.add_argument(
        "--slice-jitter", type=int, default=B53_SLICE_JITTER_DEFAULT,
        help=(
            "shift a series' slice centres by up to N either way. Off by "
            "default: it changes which slices are chosen, which is a second "
            "change and belongs in its own experiment"
        ),
    )
    parser.add_argument(
        "--all-data", action="store_true",
        help=(
            "train on every split except the unseen-scanner validation surface "
            f"({', '.join(B52_FULL_TRAIN_SPLITS)}) instead of the gate's train rows alone"
        ),
    )
    parser.add_argument(
        "--no-gradient-checkpointing", action="store_true",
        help="faster, uses more GPU memory; identical maths",
    )
    parser.add_argument("--seed", type=int, default=B52_SEED)
    parser.add_argument("--out-root", default=B53_RUN_ROOT)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    train_b53(
        _read_config(args.config),
        data_root=args.data_root,
        labels_root=args.labels_root,
        series_policy_path=args.series_policy,
        base_checkpoint=args.base_checkpoint,
        domain_split=args.domain_split,
        epochs=args.epochs,
        encoder_trainable_stages=args.encoder_stages,
        encoder_lr_scale=args.encoder_lr_scale,
        hierarchy_lr_scale=args.hierarchy_lr_scale,
        augment=not args.no_augment,
        slice_jitter=args.slice_jitter,
        train_splits=B52_FULL_TRAIN_SPLITS if args.all_data else (B52_TRAIN_SPLIT,),
        gradient_checkpointing=not args.no_gradient_checkpointing,
        seed=args.seed,
        out_root=args.out_root,
        preflight_only=args.preflight_only,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "B53_EXPERIMENT",
    "B53_VERSION",
    "B53_RUN_ROOT",
    "B53_CHECKPOINT_NAME",
    "B53_DEFAULT_EPOCHS",
    "AugmentationPolicy",
    "augment_b42_series",
    "B53AugmentedDataset",
    "verify_augmentation_reaches_pixels",
    "b53_preflight",
    "train_b53",
]
