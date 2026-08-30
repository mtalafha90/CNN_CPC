"""Present a B51 checkpoint to the B42 submission path, weights untouched.

B50's model subclasses B42's and changes only `requires_grad`, adding and
removing no parameters. So a B51 checkpoint's `base` and `head` state
dictionaries are key-for-key and shape-for-shape identical to B42's, and
`requires_grad` has no effect on a forward pass. **At inference a B51 checkpoint
is a B42 checkpoint.**

That is worth a small module rather than a fresh submission path. B41's first
hidden submission failed operationally rather than scientifically, and every new
line of inference code written for a submission is another chance to repeat it.
The B42 dual-GPU hidden-safe path has already completed a hidden run; reusing it
unchanged is the lowest-risk way to submit B51.

All this does is rewrite the metadata the B42 loader inspects. It copies the
weight tensors by reference and never reads, casts, or reconstructs one, and the
accompanying test asserts every tensor is bit-identical before and after. If
that test ever fails, the converter is wrong and the submission must not go out.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .b35_training import sha256_file
from .b42_constant_area_aspect_sparse_mil import B42_EXPERIMENT, B42_VERSION
from .b51_full_population_training import B51_EXPERIMENT

# The keys the B42 submission loader reads. Anything else is carried through
# untouched so the provenance of the original run survives conversion.
B42_REQUIRED_KEYS = ("experiment", "version", "base_state", "head_state", "model_state")

WEIGHT_KEYS = ("base_state", "head_state")

# `load_b42_checkpoint` reads the head geometry from a top-level `sparse_mil`
# block and the encoder policy from `encoder_finetune`, silently falling back to
# B42's own defaults when they are absent. B51 records the same numbers, but only
# inside `model_state`. Two of them -- `top_k` and `temperature` -- are not
# parameters, so a wrong value survives `load_state_dict(strict=True)` untouched
# and quietly changes every prediction. They are copied across explicitly rather
# than left to a default that merely happens to agree.
SPARSE_MIL_KEYS = ("grid_size", "top_k", "temperature")
ENCODER_FINETUNE_KEYS = ("encoder_trainable_stages",)


def _carry_geometry(payload: dict, model_state: dict) -> tuple[dict, dict]:
    """Lift the reconstruction settings out of model_state, refusing to guess."""
    blocks = {}
    for name, keys in (
        ("sparse_mil", SPARSE_MIL_KEYS),
        ("encoder_finetune", ENCODER_FINETUNE_KEYS),
    ):
        missing = [key for key in keys if key not in model_state]
        if missing:
            raise ValueError(
                f"B51 model_state is missing {missing}, so the B42 loader would "
                f"fall back to defaults for {name}; refusing to convert"
            )
        block = {key: model_state[key] for key in keys}

        # If the run already recorded the block, the two must agree. A conflict
        # means one of them is wrong and neither can be trusted.
        existing = payload.get(name)
        if isinstance(existing, dict):
            clashes = {
                key: (existing[key], block[key])
                for key in keys
                if key in existing and existing[key] != block[key]
            }
            if clashes:
                raise ValueError(f"{name} disagrees with model_state: {clashes}")
            block = {**existing, **block}
        blocks[name] = block
    return blocks["sparse_mil"], blocks["encoder_finetune"]


def convert(payload: dict) -> dict:
    """Return a B42-shaped payload sharing the original weight tensors."""
    if payload.get("experiment") != B51_EXPERIMENT:
        raise ValueError(
            f"expected a {B51_EXPERIMENT} checkpoint; got {payload.get('experiment')!r}"
        )
    for key in WEIGHT_KEYS:
        if key not in payload or not isinstance(payload[key], dict):
            raise ValueError(f"B51 checkpoint is missing its {key}")

    converted = dict(payload)
    # The weight dictionaries are carried by reference. Nothing reads a tensor.
    converted["experiment"] = B42_EXPERIMENT
    converted["version"] = B42_VERSION
    model_state = dict(payload.get("model_state") or {})
    model_state["version"] = B42_VERSION
    model_state["experiment"] = B42_EXPERIMENT
    converted["model_state"] = model_state

    sparse_mil, encoder_finetune = _carry_geometry(payload, model_state)
    converted["sparse_mil"] = sparse_mil
    converted["encoder_finetune"] = encoder_finetune

    # Provenance, so a converted file can never be mistaken for a real B42 run.
    converted["converted_from"] = {
        "experiment": B51_EXPERIMENT,
        "version": payload.get("version"),
        "adapt_hierarchy": bool(payload.get("adapt_hierarchy")),
        "hierarchy_lr_scale": payload.get("hierarchy_lr_scale"),
        "training_studies": payload.get("training_studies"),
        "seed": payload.get("seed"),
        "note": (
            "B50's class changes only requires_grad and adds no parameters, so "
            "this file's weights are exactly what B51 trained. Only metadata "
            "was rewritten, so the proven B42 inference path can load it."
        ),
    }
    missing = [key for key in B42_REQUIRED_KEYS if key not in converted]
    if missing:
        raise RuntimeError(f"converted payload is missing B42 keys: {missing}")
    return converted


def weights_are_identical(before: dict, after: dict) -> bool:
    """Every tensor bit-for-bit the same. The one property that matters."""
    for key in WEIGHT_KEYS:
        left, right = before[key], after[key]
        if set(left) != set(right):
            return False
        for name in left:
            a, b = left[name], right[name]
            if a.dtype != b.dtype or a.shape != b.shape:
                return False
            if not torch.equal(a, b):
                return False
    return True


def convert_file(source: str | Path, destination: str | Path) -> dict:
    source_path, destination_path = Path(source), Path(destination)
    if destination_path.exists():
        raise FileExistsError(f"refusing to overwrite {destination_path}")

    payload = torch.load(str(source_path), map_location="cpu", weights_only=False)
    converted = convert(payload)
    if not weights_are_identical(payload, converted):
        raise RuntimeError("conversion altered the weights; the submission must not go out")

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, destination_path)

    reloaded = torch.load(str(destination_path), map_location="cpu", weights_only=False)
    if not weights_are_identical(payload, reloaded):
        raise RuntimeError("the written file's weights differ from the source")

    record = {
        "source": str(source_path.resolve()),
        "source_sha256": sha256_file(source_path),
        "destination": str(destination_path.resolve()),
        "destination_sha256": sha256_file(destination_path),
        "experiment_in": B51_EXPERIMENT,
        "experiment_out": B42_EXPERIMENT,
        "weights_verified_identical": True,
        "tensors_checked": sum(len(payload[key]) for key in WEIGHT_KEYS),
        # Printed rather than implied: these are the settings the B42 loader
        # would otherwise have defaulted, and two of them are invisible to a
        # strict state-dict load. Seeing them is how the operator checks them.
        "sparse_mil": converted["sparse_mil"],
        "encoder_finetune": converted["encoder_finetune"],
    }
    destination_path.with_suffix(".conversion.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print(json.dumps(record, indent=2))
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite a B51 checkpoint's metadata so the B42 submission path loads it"
    )
    parser.add_argument("--source", required=True, help="the B51 checkpoint")
    parser.add_argument("--destination", required=True, help="where to write the B42-shaped copy")
    args = parser.parse_args()
    convert_file(args.source, args.destination)


if __name__ == "__main__":
    main()


__all__ = ["convert", "convert_file", "weights_are_identical"]
