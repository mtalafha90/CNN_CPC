"""Train the working model on the full report-only study population.

Labels come from the radiology reports, never from the expert-annotated
studies, which stay outside the gradient entirely.  The reports are
multilingual, and the two label surfaces are named for the reports each one
can actually read:

    latin-script   the frozen rule parser alone.  It matches Latin-script
                   vocabulary across several languages, so this is not an
                   English-only surface, but the Greek- and Cyrillic-script
                   reports yield almost nothing.
    all-script     the same labels plus the cells recovered by translating
                   the Greek- and Cyrillic-script reports before parsing.

Training stops at a fixed epoch.  No checkpoint is chosen by looking at a
labelled score, so the run is decided before any result is seen.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from model._implementation import (
    ENCODERS,
    read_config,
    run_directory,
    train_working_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the working knee MRI model")
    parser.add_argument(
        "--supervision",
        choices=("latin-script", "all-script"),
        required=True,
        help="which report-label surface enters the gradient",
    )
    parser.add_argument("--config", default="config/current_model.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--latin-script-labels",
        required=True,
        help="directory holding the frozen rule-parser label export",
    )
    parser.add_argument(
        "--all-script-labels",
        required=True,
        help="directory holding the merged translated-report label export",
    )
    parser.add_argument("--series-policy", required=True)
    parser.add_argument(
        "--encoder",
        choices=ENCODERS,
        default="report-aligned",
        help="which pretrained weights the frozen encoder starts from",
    )
    parser.add_argument(
        "--encoder-checkpoint",
        help="report-aligned encoder checkpoint; required for --encoder report-aligned",
    )
    parser.add_argument(
        "--encoder-trainable-stages",
        type=int,
        default=0,
        help="let the last N encoder blocks learn (0 keeps the encoder frozen)",
    )
    parser.add_argument(
        "--encoder-lr-scale",
        type=float,
        default=0.05,
        help="learning rate for those blocks, as a fraction of the head's",
    )
    parser.add_argument(
        "--dinov3-variant",
        choices=("tiny", "small"),
        default="tiny",
        help="DINOv3 ConvNeXt size; both are 768-d and drop in unchanged",
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="run name; everything it produces lands under runs/<experiment>/",
    )
    args = parser.parse_args()

    if args.encoder == "report-aligned" and not args.encoder_checkpoint:
        parser.error("--encoder report-aligned requires --encoder-checkpoint")

    config = read_config(args.config)
    config["data_root"] = str(Path(args.data_root).resolve())

    checkpoint = train_working_model(
        config,
        supervision=args.supervision,
        latin_script_labels_root=args.latin_script_labels,
        all_script_labels_root=args.all_script_labels,
        series_policy_path=args.series_policy,
        encoder_checkpoint=args.encoder_checkpoint,
        encoder=args.encoder,
        dinov3_variant=args.dinov3_variant,
        encoder_trainable_stages=args.encoder_trainable_stages,
        encoder_lr_scale=args.encoder_lr_scale,
        out_root=run_directory(args.experiment, "train"),
    )
    print(f"checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
