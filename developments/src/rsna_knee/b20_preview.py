"""Preview the frozen B20 crop-only preprocessing on one training study."""
from __future__ import annotations

import argparse
from pathlib import Path

from .b7_weak_supervision import _read_config
from .b20_crop_focus import b20_crop_focus_policy
from .crop_focus import apply_crop_focus
from .focus_preview import run_focus_preview, select_preview_study


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b20-preview")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--uid", default=None)
    parser.add_argument("--out", default="runs/b20_crop_focus/crop_focus_preview.png")
    args = parser.parse_args()

    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    policy = b20_crop_focus_policy(config)
    root = Path(config["data_root"])

    uid, series, repair = select_preview_study(
        config,
        root,
        requested_uid=args.uid,
        require_expert_uid=False,
    )
    run_focus_preview(
        root=root,
        uid=uid,
        series=series,
        repair=repair,
        transform=apply_crop_focus,
        policy=policy,
        title="B20 crop-only preview",
        subtitle=f"crop={policy['crop_fraction']:.2f} | no vignette",
        transformed_label="B20 crop only",
        output=Path(args.out),
    )


if __name__ == "__main__":
    main()
