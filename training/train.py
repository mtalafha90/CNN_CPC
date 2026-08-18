"""Train the working model on the full report-only study population.

Labels come from the radiology reports, never from the expert-annotated
studies, which stay outside the gradient entirely.  Two label surfaces are
available:

    original   the frozen rule-parser labels
    merged     the same labels plus the cells recovered by translating the
               non-Latin-script reports before parsing

Training stops at a fixed epoch.  No checkpoint is chosen by looking at a
labelled score, so the run is decided before any result is seen.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from model._implementation import read_config, train_working_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the working knee MRI model")
    parser.add_argument(
        "--supervision",
        choices=("original", "merged"),
        required=True,
        help="which report-label surface enters the gradient",
    )
    parser.add_argument("--config", default="config/current_model.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--report-labels",
        required=True,
        help="directory holding the frozen rule-parser label export",
    )
    parser.add_argument(
        "--translated-labels",
        required=True,
        help="directory holding the merged translated-report label export",
    )
    parser.add_argument("--series-policy", required=True)
    parser.add_argument(
        "--encoder",
        required=True,
        help="frozen report-aligned encoder checkpoint",
    )
    parser.add_argument("--out-root", default="runs/working_model")
    args = parser.parse_args()

    config = read_config(args.config)
    config["data_root"] = str(Path(args.data_root).resolve())

    train_working_model(
        config,
        supervision=args.supervision,
        report_labels_root=args.report_labels,
        translated_labels_root=args.translated_labels,
        series_policy_path=args.series_policy,
        encoder_checkpoint=args.encoder,
        out_root=args.out_root,
    )


if __name__ == "__main__":
    main()
