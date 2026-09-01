"""Submit the B51 endpoint through B42's proven dual-T4 inference path.

B50's model subclasses B42's and changes only `requires_grad`. It adds and
removes no parameters, and `requires_grad` has no effect on a forward pass, so
at inference a B51 checkpoint *is* a B42 checkpoint. Nothing about the geometry,
the ragged per-series encoding, the three centre offsets, the sparse-MIL head or
the probability averaging differs. Writing a second inference path would only
create a second chance to repeat B41's first hidden submission, which failed
operationally rather than scientifically.

So this module runs B42's launcher unchanged. It exists for one reason: that
launcher pins the SHA-256 of B42's own frozen checkpoint, and a B51 file cannot
match it -- different weights mean different bytes, by construction. The pin is
right and is deliberately kept. What changes is only *which* artefact is pinned.

The guarantee the pin provides is preserved exactly: a hidden run, where nobody
can look at what happened, cannot silently use a checkpoint that nobody named.
The operator must state the fingerprint they intend, and this module additionally
refuses any file that is not a converted B51 checkpoint, so it can never become a
back door for running an arbitrary file through the B42 path.

Produce the input with `b51_checkpoint_to_b42_format.py`, which rewrites metadata
only and verifies every weight tensor is bit-identical before and after.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from .b35_training import sha256_file
from .b42_constant_area_aspect_sparse_mil import B42_EXPERIMENT, B42_VERSION
from .b42_constant_area_aspect_sparse_submission_dualgpu_fast import (
    DEFAULT_FALLBACK_PROBABILITY,
    ON_UNREADABLE_FALLBACK,
    generate_b42_submission_dual_gpu_fast,
)
from .b51_full_population_training import B51_EXPERIMENT

B51_SUBMISSION_EXPERIMENT = "B51_full_population_adapted_hierarchy_hidden_test_inference"


def require_converted_b51(checkpoint: str | Path) -> dict:
    """Refuse anything but a checkpoint produced by the B51 converter.

    Without this the module would accept any file at all, provided its declared
    fingerprint matched -- which would turn a safety check into a way around
    B42's pin.
    """
    path = Path(checkpoint).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"B51 submission checkpoint is missing: {path}")

    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    origin = payload.get("converted_from")
    if not isinstance(origin, dict) or origin.get("experiment") != B51_EXPERIMENT:
        raise ValueError(
            "this file was not produced by b51_checkpoint_to_b42_format; run the "
            "converter on the B51 checkpoint first"
        )
    if payload.get("experiment") != B42_EXPERIMENT or payload.get("version") != B42_VERSION:
        raise ValueError("the converted checkpoint does not present as B42 to the loader")
    if not origin.get("adapt_hierarchy"):
        raise ValueError(
            "the source run did not adapt the study hierarchy, so it is not B51"
        )

    # The geometry the B42 loader would otherwise default. See the converter.
    for key in ("sparse_mil", "encoder_finetune"):
        if not isinstance(payload.get(key), dict) or not payload[key]:
            raise ValueError(
                f"the converted checkpoint has no {key} block, so the B42 loader "
                "would fall back to defaults instead of B51's real settings; "
                "re-run the converter"
            )
    return {
        "checkpoint": str(path),
        "sha256": sha256_file(path),
        "training_studies": origin.get("training_studies"),
        "hierarchy_lr_scale": origin.get("hierarchy_lr_scale"),
        "seed": origin.get("seed"),
        "sparse_mil": payload["sparse_mil"],
        "encoder_finetune": payload["encoder_finetune"],
    }


def generate_b51_submission_dual_gpu_fast(
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
    """Run B42's dual-T4 path against the declared B51 checkpoint.

    Defaults to `on_unreadable="fallback"`, unlike B42's own launcher, whose
    0.714 hidden run was made under the strict default and must stay
    reproducible. B51's first hidden attempt threw an exception it could not
    report, on data three clean example studies could not reveal; aborting 1,300
    studies for one unreadable one is not a property a submission wants.
    """
    identity = require_converted_b51(checkpoint)
    if identity["sha256"] != expected_checkpoint_sha256:
        raise ValueError(
            "B51 hidden submission requires the declared checkpoint: "
            f"expected {expected_checkpoint_sha256}, got {identity['sha256']}"
        )

    print(f"[B51 submit] {B51_SUBMISSION_EXPERIMENT}", flush=True)
    print(f"[B51 submit] checkpoint sha256 {identity['sha256']}", flush=True)
    print(
        f"[B51 submit] trained on {identity['training_studies']} studies, "
        f"hierarchy lr scale {identity['hierarchy_lr_scale']}",
        flush=True,
    )
    print(f"[B51 submit] head geometry {identity['sparse_mil']}", flush=True)
    print("[B51 submit] inference path is B42's, unchanged", flush=True)

    return generate_b42_submission_dual_gpu_fast(
        config,
        data_root=data_root,
        checkpoint=checkpoint,
        base_checkpoint=base_checkpoint,
        out_path=out_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        on_unreadable=on_unreadable,
        fallback_probability=fallback_probability,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit the B51 endpoint through B42's dual-T4 inference path"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True, help="the converted B51 checkpoint")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument(
        "--expected-checkpoint-sha256",
        required=True,
        help="the SHA-256 the converter reported for this file",
    )
    parser.add_argument("--out-path", default="submission.csv")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    generate_b51_submission_dual_gpu_fast(
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
    "B51_SUBMISSION_EXPERIMENT",
    "generate_b51_submission_dual_gpu_fast",
    "require_converted_b51",
]
