"""Run the active B20 model on the released competition test set."""

from __future__ import annotations

import argparse
from pathlib import Path

from model.bootstrap import ensure_developments_source


def main() -> None:
    ensure_developments_source()
    from rsna_knee.b7_weak_supervision import _read_config
    from rsna_knee.b20_submission import generate_b20_submission

    parser = argparse.ArgumentParser(
        description="Generate test-set predictions with the active B20 model"
    )
    parser.add_argument("--config", default="config/current_model.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--checkpoint", default="runs/b20_crop_focus/b20_model.pt"
    )
    parser.add_argument("--out", default="submission.csv")
    args = parser.parse_args()

    config = dict(_read_config(args.config))
    config["data_root"] = str(Path(args.data_root).resolve())
    generate_b20_submission(
        config,
        checkpoint=args.checkpoint,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
