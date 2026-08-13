"""Preview the frozen B19 joint-focus transform before training."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .b7_weak_supervision import _read_config, make_b7_dataset_config
from .b12_variable_series import VariableSeriesKneeDataset, build_variable_series_index
from .b19_joint_focus import JointFocusedVariableSeriesKneeDataset, require_b19_contract
from .data import backfill_series_metadata, gold_mask, load_series_csv, load_train_csv


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b19-preview")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--uid", default=None)
    parser.add_argument(
        "--out", default="runs/b19_joint_focus/joint_focus_preview.png"
    )
    args = parser.parse_args()

    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    joint_policy = require_b19_contract(config)
    root = Path(config["data_root"])

    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    gold = train.loc[gold_mask(train), ["StudyInstanceUID"]].copy()
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)
    if len(gold) != 58:
        raise ValueError("B19 preview expects the complete 58-study expert surface")
    uid = str(args.uid) if args.uid is not None else str(gold.iloc[0]["StudyInstanceUID"])
    if uid not in set(gold["StudyInstanceUID"]):
        raise ValueError("--uid must identify one of the 58 expert-labelled studies")

    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, _ = backfill_series_metadata(series, root, split="train")
    index = build_variable_series_index(series, [uid])
    records = index[uid]
    if not records:
        raise ValueError("selected study has zero eligible series")

    dataset_config = make_b7_dataset_config(config, root, train=False, tta_offsets=())
    original_ds = VariableSeriesKneeDataset(
        [uid], index, dataset_config, train=False
    )
    focused_ds = JointFocusedVariableSeriesKneeDataset(
        [uid], index, dataset_config, train=False, joint_focus_policy=joint_policy
    )
    original = original_ds[0]["volumes"]
    focused = focused_ds[0]["volumes"]

    planes = ["Sagittal", "Coronal", "Axial"]
    chosen = []
    for plane in planes:
        match = next((i for i, record in enumerate(records) if record["plane"] == plane), None)
        chosen.append(match)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for col, (plane, series_idx) in enumerate(zip(planes, chosen)):
        if series_idx is None:
            axes[0, col].text(0.5, 0.5, f"No {plane} series", ha="center", va="center")
            axes[1, col].text(0.5, 0.5, f"No {plane} series", ha="center", va="center")
            axes[0, col].axis("off")
            axes[1, col].axis("off")
            continue
        slice_idx = int(original.shape[1] // 2)
        raw = original[series_idx, slice_idx, 1].numpy()
        foc = focused[series_idx, slice_idx, 1].numpy()
        record = records[series_idx]

        axes[0, col].imshow(raw, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"{plane} — original")
        axes[1, col].imshow(foc, cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title(f"{plane} — B19 focused")
        for row in (0, 1):
            axes[row, col].axis("off")
        print(
            {
                "plane": plane,
                "series_index": int(series_idx),
                "series_uid": str(record["series_uid"]),
                "sampled_slice_index": slice_idx,
                "original_mean": float(np.mean(raw)),
                "focused_mean": float(np.mean(foc)),
            }
        )

    fig.suptitle(
        f"B19 joint-focus preview | UID={uid} | policy={joint_policy}", fontsize=11
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
