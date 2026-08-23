"""Train B41: B37 sparse-MIL with native-aspect-preserving 90% crops.

B41 is an isolated preprocessing ablation.  It uses B37's audited training
harness and all of its fixed model, supervision, memory, and duration controls.
Only the B41 dataset changes the final in-plane operation from direct square
stretching to antialiased resize-to-fit plus symmetric zero padding.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .b7_weak_supervision import _read_config
from .b37_highres_sparse_training import (
    B37_CONSTRUCTION_SEED_OFFSET,
    B37_EPOCHS,
    B37_LOADER_SEED_OFFSET,
    train_b37,
)
from .b41_highres_aspect_sparse_mil import (
    B41_EXPERIMENT,
    B41_RUN_ROOT,
    B41_VERSION,
    B41HighResAspectSparseDataset,
    b41_preprocessing_state,
    require_b41_aspect_contract,
)

# Keep B41's duration exactly equal to the completed B37 endpoint.
B41_EPOCHS = B37_EPOCHS
# Use a distinct, explicit initialization stream for this prospective ablation.
B41_CONSTRUCTION_SEED_OFFSET = B37_CONSTRUCTION_SEED_OFFSET + 4_000_000
# Use a distinct, explicit loader stream while keeping every loader control fixed.
B41_LOADER_SEED_OFFSET = B37_LOADER_SEED_OFFSET + 4_000_000


def train_b41(
    config: dict,
    *,
    data_root: str | Path,
    labels_root: str | Path,
    series_policy_path: str | Path,
    base_checkpoint: str | Path,
    out_root: str | Path = B41_RUN_ROOT,
    preflight_only: bool = False,
) -> Path | None:
    """Train the fixed B41 endpoint or run its no-update largest-batch preflight."""
    # Make a private copy so the caller's configuration object is never mutated.
    settings = dict(config)
    # Record a descriptive B41 preprocessing block in every recovery/final artifact.
    preprocessing = b41_preprocessing_state()
    # Run the B37 harness with B41's only changed component: the dataset transform.
    return train_b37(
        settings,
        data_root=data_root,
        labels_root=labels_root,
        series_policy_path=series_policy_path,
        base_checkpoint=base_checkpoint,
        out_root=out_root,
        preflight_only=bool(preflight_only),
        dataset_class=B41HighResAspectSparseDataset,
        contract_validator=require_b41_aspect_contract,
        experiment_name=B41_EXPERIMENT,
        version=B41_VERSION,
        run_tag="B41",
        checkpoint_filename="b41_model.pt",
        construction_seed_offset=B41_CONSTRUCTION_SEED_OFFSET,
        loader_seed_offset=B41_LOADER_SEED_OFFSET,
        joint_hypothesis=(
            "preserving the retained native in-plane aspect ratio after the exact "
            "B37 90% crop improves sparse-MIL evidence without changing B37's "
            "model, supervision, optimization, or duration"
        ),
        preprocessing=preprocessing,
        governance=(
            "Prospective B41 preprocessing-only endpoint. B37's 32 triplets, "
            "448 canvas, 6x6 grid, top-k=8, report-only supervision, encoder-tail "
            "adaptation, and fixed two epochs remain frozen. Expert58 is a reused "
            "diagnostic only; do not tune resize policy, crop fraction, grid size, "
            "top-k, target subset, or epoch after observing it. Hidden competition "
            "evidence is required for promotion."
        ),
    )


def main() -> None:
    """Parse the standalone local B41 command-line interface."""
    # Describe B41 rather than the shared B37 training harness in command help.
    parser = argparse.ArgumentParser(
        "Train B41 native-aspect-preserving high-resolution sparse MIL"
    )
    # Default to the immutable B41 configuration file committed with this module.
    parser.add_argument("--config", default="config/b41_highres_aspect_sparse_448.yaml")
    # Require the local competition train CSV and DICOM root.
    parser.add_argument("--data-root", required=True)
    # Require B67's frozen fill-merged supervision export root.
    parser.add_argument("--labels-root", required=True)
    # Require B12/B13's frozen all-series selection policy.
    parser.add_argument("--series-policy", required=True)
    # Require B67's frozen full-fill base checkpoint.
    parser.add_argument("--base-checkpoint", required=True)
    # Write only B41 artifacts below its permanent numbered run container by default.
    parser.add_argument("--out-root", default=B41_RUN_ROOT)
    # Allow a forward/backward memory and gradient test without an optimizer update.
    parser.add_argument("--preflight-only", action="store_true")
    # Parse user-supplied arguments once.
    args = parser.parse_args()
    # Read the YAML file through the repository's existing config reader.
    config = dict(_read_config(args.config))
    # Start the fixed endpoint or its no-update preflight.
    train_b41(
        config,
        data_root=args.data_root,
        labels_root=args.labels_root,
        series_policy_path=args.series_policy,
        base_checkpoint=args.base_checkpoint,
        out_root=args.out_root,
        preflight_only=bool(args.preflight_only),
    )


if __name__ == "__main__":
    main()
