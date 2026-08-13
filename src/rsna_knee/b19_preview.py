"""Preview the frozen B19 crop+cosine preprocessing on one expert study."""
from __future__ import annotations

import argparse
from pathlib import Path

from .b7_weak_supervision import _read_config
from .b19_joint_focus import require_b19_contract
from .focus_preview import run_focus_preview, select_preview_study
from .joint_focus import apply_joint_focus


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b19-preview")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--uid", default=None)
    parser.add_argument("--out", default="runs/b19_joint_focus/joint_focus_preview.png")
    args = parser.parse_args()

    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    policy = require_b19_contract(config)
    root = Path(config["data_root"])

    uid, series, repair = select_preview_study(
        config,
        root,
        requested_uid=args.uid,
        require_expert_uid=True,
    )
    run_focus_preview(
        root=root,
        uid=uid,
        series=series,
        repair=repair,
        transform=apply_joint_focus,
        policy=policy,
        title="B19 crop + cosine preview",
        subtitle=(
            f"crop={policy['crop_fraction']:.2f} | full={policy['full_weight_fraction']:.2f} | "
            f"zero={policy['outer_zero_fraction']:.2f}"
        ),
        transformed_label="B19 crop + cosine",
        output=Path(args.out),
    )


if __name__ == "__main__":
    main()
