"""Submit the B52 endpoint through B42's proven dual-T4 inference path.

B52 fine-tunes B50's model, which subclasses B42's and changes only
`requires_grad`. It adds and removes no parameters, and neither `requires_grad`
nor `encoder_trainable_stages` has any effect on a forward pass -- the latter
only selects which encoder stages receive gradients, and the gradient
checkpointing it gates is `self.training`-only. **At inference a B52 checkpoint
runs as a B42 checkpoint**, through the same geometry, the same ragged
per-series encoding, the same three centre offsets, the same sparse-MIL head and
the same probability averaging.

So this module does not write an inference path. It calls B42's, unchanged,
supplying only two things B42's own launcher cannot supply for another endpoint:
how to load the checkpoint, and what the manifest may truthfully claim about it.

## Why B52 cannot go through B51's converter

B51 rewrites its metadata to present as B42, and `load_b42_checkpoint` accepts
it because B51 genuinely satisfies every claim that loader checks: a fixed
two-epoch endpoint, trained on all 4,349 report-only studies, with the series
and supervision-cell counts B35 recorded.

B52 satisfies none of those four, by design:

```text
fixed_endpoint / completed_epochs == 2   B52 ran six epochs and selected the best
training_studies == 4349                 B52 held out a scanner-grouped validation split
training_series                          not recorded by B52
training_supervision_cells               not recorded by B52
```

A converter could set those fields anyway. It must not: they are assertions
about how a model was trained, and writing false ones into a checkpoint to get
past a check would corrupt the provenance of a real submission -- and would do
it silently, inside the one artefact nobody can inspect during a hidden run.

So B52 gets its own loader, which checks the properties that actually govern a
forward pass and a fair score, and states its own identity rather than borrowing
B42's. The file submitted is byte-for-byte the file that was trained; there is
no conversion step at all.

## What is still checked, and why each one earns its place

```text
declared SHA-256          a hidden run cannot use a file nobody named
B52 experiment + version  and not a B42 or B51 file, so this is not a back door
adapt_hierarchy           the model class the weights were trained under
base checkpoint SHA-256   the frozen base these weights were fine-tuned from
encoder fingerprint       the reconstructed encoder is the one that was trained
head geometry present     top_k and temperature are not weights, so a wrong
                          value survives a strict load and changes every score
encoder chunk size 4      the execution contract the timing budget assumes
no expert labels, no expert gradients    the 58 gold studies stay unseen
```
"""
from __future__ import annotations

import argparse
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
from .b50_adapted_hierarchy_mil import B50_EXPERIMENT
from .b52_competition_training import B52_EXPERIMENT, B52_VERSION
from .phase9_matched_supervision_training import load_phase9_checkpoint

B52_SUBMISSION_EXPERIMENT = "B52_competition_full_finetune_hidden_test_inference"

# The execution contract B42's runtime budget was measured under. Not a default:
# a different chunk changes how long a study takes, and the guard that decides
# whether a hidden run can finish is calibrated against this one.
B52_REQUIRED_ENCODER_CHUNK_SIZE = 4

# Read from model_state, never defaulted. `grid_size` would fail a strict load if
# it were wrong, but `top_k` and `temperature` are not parameters -- a wrong
# value loads cleanly and silently changes every prediction.
SPARSE_MIL_KEYS = ("grid_size", "top_k", "temperature")


def require_b52_endpoint(payload: dict) -> dict:
    """Refuse anything that is not a real B52 checkpoint, and say which part failed.

    Without this the launcher would accept any file whose hash the operator
    happened to declare, which would turn B42's pin into something to route
    around rather than something to use.
    """
    if payload.get("experiment") != B52_EXPERIMENT:
        raise ValueError(
            f"expected a {B52_EXPERIMENT} checkpoint; got {payload.get('experiment')!r}"
        )
    if payload.get("version") != B52_VERSION:
        raise ValueError(
            f"expected {B52_VERSION}; got {payload.get('version')!r}"
        )
    for key in ("base_state", "head_state", "model_state"):
        if not isinstance(payload.get(key), dict) or not payload[key]:
            raise ValueError(f"B52 checkpoint is missing its {key}")

    model_state = payload["model_state"]
    if model_state.get("experiment") != B50_EXPERIMENT:
        raise ValueError(
            "B52 checkpoint was not trained with B50's model class; "
            f"model_state names {model_state.get('experiment')!r}"
        )
    trainable = model_state.get("trainable")
    if not isinstance(trainable, dict) or not trainable.get("adapt_hierarchy"):
        raise ValueError("the source run did not adapt the study hierarchy, so it is not B52")

    missing = [key for key in SPARSE_MIL_KEYS if key not in model_state]
    if missing:
        raise ValueError(
            f"B52 model_state is missing {missing}, so the head geometry would be "
            "guessed; refusing to submit"
        )
    chunk = int(model_state.get("encoder_chunk_size", -1))
    if chunk != B52_REQUIRED_ENCODER_CHUNK_SIZE:
        raise ValueError(
            f"B52 hidden submission requires trained encoder chunk size "
            f"{B52_REQUIRED_ENCODER_CHUNK_SIZE}; got {chunk}"
        )

    # The 58 expert studies are the only gold labels in the competition data and
    # are held out of every run in this project. A checkpoint that touched them
    # must never be submitted, whatever else about it is correct.
    if bool(payload.get("gold_labels_used", True)):
        raise ValueError("B52 checkpoint unexpectedly used expert labels")
    if int(payload.get("gold_studies_used_in_gradient", -1)) != 0:
        raise ValueError("B52 checkpoint unexpectedly used expert gradients")

    return {
        "sparse_mil": {key: model_state[key] for key in SPARSE_MIL_KEYS},
        "encoder_finetune": {
            "encoder_trainable_stages": int(model_state.get("encoder_trainable_stages", -1))
        },
        "encoder_chunk_size": chunk,
        "selected_epoch": payload.get("selected_epoch"),
        "selection_value": payload.get("selection_value"),
        "training_studies": payload.get("training_studies"),
        "validation_studies": payload.get("validation_studies"),
        "augmentation_enabled": bool(payload.get("augmentation_enabled", False)),
        "seed": payload.get("seed"),
    }


def load_b52_checkpoint(path: str | Path, *, base_checkpoint: str | Path, device):
    """Reconstruct the B52 model for inference, refusing to guess anything.

    Deliberately parallel to `load_b42_checkpoint` and deliberately not a call to
    it: the reconstruction is identical, the identity assertions are B52's.
    """
    checkpoint = Path(path).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    identity = require_b52_endpoint(payload)

    base_path = Path(base_checkpoint).resolve()
    if sha256_file(base_path) != str(payload.get("base_checkpoint_sha256", "")):
        raise ValueError("B52 base checkpoint fingerprint mismatch")
    base, _ = load_phase9_checkpoint(base_path, expected_arm="llm_fill", device="cpu")

    sparse = identity["sparse_mil"]
    model = B42ConstantAreaAspectSparseMILResidual(
        base,
        grid_size=int(sparse["grid_size"]),
        top_k=int(sparse["top_k"]),
        temperature=float(sparse["temperature"]),
        encoder_trainable_stages=int(identity["encoder_finetune"]["encoder_trainable_stages"]),
        encoder_chunk_size=int(identity["encoder_chunk_size"]),
    )
    model.base.load_state_dict(payload["base_state"], strict=True)
    model.head.load_state_dict(payload["head_state"], strict=True)
    model = model.to(device)
    model.eval()

    observed = encoder_state_sha256(model.base.encoder)
    expected = str(payload.get("encoder_sha256_final", ""))
    if observed != expected:
        raise RuntimeError("B52 reconstructed encoder fingerprint changed")
    return model, payload


def _load_b52_replica(checkpoint_path: Path, base_path: Path, device: torch.device):
    model, payload = load_b52_checkpoint(
        checkpoint_path, base_checkpoint=base_path, device=device
    )
    model.eval()
    return model, payload


def b52_endpoint_manifest(payload: dict) -> dict:
    """What is true of B52, in the fields B42's manifest reserves for the endpoint.

    `fixed_endpoint` is `False` here and that is the point: B52 chose its epoch
    on a held-out split, so its validation number is a selection statistic. The
    manifest says so rather than inheriting B42's claim to a frozen endpoint.
    """
    history = payload.get("history")
    return {
        "experiment": B52_SUBMISSION_EXPERIMENT,
        "version": B52_VERSION,
        "fixed_endpoint": False,
        "selected_epoch": payload.get("selected_epoch"),
        "selection_metric": payload.get("selection_metric"),
        "selection_value": payload.get("selection_value"),
        "epochs_planned": payload.get("epochs_planned"),
        "completed_epochs": len(history) if isinstance(history, list) else None,
        # B42 records both; B52's trainer does not. Null is the honest answer --
        # a manifest that guessed a plausible number would be worse than one
        # that admits the run never counted them.
        "training_series": None,
        "training_supervision_cells": None,
        "counts_not_recorded_by_b52": ["training_series", "training_supervision_cells"],
        "training_studies": int(payload.get("training_studies", -1)),
        "validation_studies": int(payload.get("validation_studies", -1)),
        "train_splits": payload.get("train_splits"),
        "augmentation_enabled": bool(payload.get("augmentation_enabled", False)),
        "encoder_trainable_stages": payload.get("encoder_trainable_stages"),
        "changed_from_frozen_contract": payload.get("changed_from_frozen_contract"),
        "prediction": "B52 combined sparse-MIL logits; raw sigmoid probability",
        "governance": (
            "B52 selected its checkpoint on a held-out report-labelled split, "
            "which is competition practice and deliberately not the "
            "frozen-endpoint policy the scientific line uses. Its validation "
            "number is a selection statistic, not evidence of an effect, and "
            "must not be quoted as one. Inference execution is B42's unchanged."
        ),
    }


def generate_b52_submission_dual_gpu_fast(
    config: dict,
    *,
    data_root: str | Path,
    checkpoint: str | Path,
    base_checkpoint: str | Path,
    expected_checkpoint_sha256: str,
    out_path: str | Path = "submission.csv",
    on_unreadable: str = ON_UNREADABLE_FALLBACK,
    fallback_probability: float = DEFAULT_FALLBACK_PROBABILITY,
) -> Path:
    """Run B42's dual-T4 path against the declared B52 checkpoint.

    Defaults to `on_unreadable="fallback"`, unlike B42's own launcher. B42 keeps
    the strict default because its 0.714 hidden run was made under it and must
    stay reproducible. B52 has no hidden run to protect, and a competition
    submission that aborts on one unreadable study out of 1,300 produces no
    score at all.
    """
    path = Path(checkpoint).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"B52 submission checkpoint is missing: {path}")
    observed = sha256_file(path)
    if observed != expected_checkpoint_sha256:
        raise ValueError(
            "B52 hidden submission requires the declared checkpoint: "
            f"expected {expected_checkpoint_sha256}, got {observed}"
        )
    identity = require_b52_endpoint(
        torch.load(str(path), map_location="cpu", weights_only=False)
    )

    print(f"[B52 submit] {B52_SUBMISSION_EXPERIMENT}", flush=True)
    print(f"[B52 submit] checkpoint sha256 {observed}", flush=True)
    print(
        f"[B52 submit] epoch {identity['selected_epoch']} selected at "
        f"{identity['selection_value']}, trained on {identity['training_studies']} "
        f"studies, augmentation={identity['augmentation_enabled']}",
        flush=True,
    )
    print(f"[B52 submit] head geometry {identity['sparse_mil']}", flush=True)
    print("[B52 submit] inference path is B42's, unchanged", flush=True)

    return generate_b42_submission_dual_gpu_fast(
        config,
        data_root=data_root,
        checkpoint=path,
        base_checkpoint=base_checkpoint,
        out_path=out_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        load_replica=_load_b52_replica,
        endpoint_manifest=b52_endpoint_manifest,
        on_unreadable=on_unreadable,
        fallback_probability=fallback_probability,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit the B52 endpoint through B42's dual-T4 inference path"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True, help="the B52 checkpoint, unmodified")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument(
        "--expected-checkpoint-sha256",
        required=True,
        help="sha256sum of the B52 checkpoint you intend to submit",
    )
    parser.add_argument("--out-path", default="submission.csv")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    generate_b52_submission_dual_gpu_fast(
        dict(config),
        data_root=args.data_root,
        checkpoint=args.checkpoint,
        base_checkpoint=args.base_checkpoint,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        out_path=args.out_path,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "B52_SUBMISSION_EXPERIMENT",
    "b52_endpoint_manifest",
    "generate_b52_submission_dual_gpu_fast",
    "load_b52_checkpoint",
    "require_b52_endpoint",
]
