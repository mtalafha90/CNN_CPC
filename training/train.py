"""Train the current B20 working model using the frozen recorded recipe."""

from __future__ import annotations

import argparse
from pathlib import Path

from model.bootstrap import ensure_developments_source


def main() -> None:
    ensure_developments_source()
    from rsna_knee.b7_weak_supervision import _read_config
    from rsna_knee.b20_crop_focus import train_b20

    parser = argparse.ArgumentParser(
        description="Train the active B20 CNN-based knee MRI model"
    )
    parser.add_argument("--config", default="config/current_model.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--b6-root", default="runs/b6_report_labels_v121")
    parser.add_argument(
        "--series-policy",
        default="runs/b12_variable_series/audit/series_policy.json",
    )
    parser.add_argument(
        "--report-ssl-checkpoint",
        default="runs/b16_full_report/report_ssl/b16_report_encoder.pt",
    )
    parser.add_argument("--out-root", default="runs/current_model")
    args = parser.parse_args()

    config = _read_config(args.config)
    config = dict(config)
    config["data_root"] = str(Path(args.data_root).resolve())

    checkpoint = train_b20(
        config,
        b6_root=args.b6_root,
        series_policy_path=args.series_policy,
        report_ssl_checkpoint=args.report_ssl_checkpoint,
        out_root=args.out_root,
    )
    print(f"current-model checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
