"""Preview B20 crop-only preprocessing before training.

The preview deliberately filters train_series.csv to one selected study before
DICOM metadata backfill. This keeps a visual sanity check from accidentally
scanning the complete training collection.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .b7_weak_supervision import _read_config, make_b7_dataset_config
from .b12_variable_series import VariableSeriesKneeDataset, build_variable_series_index
from .b20_crop_focus import b20_crop_focus_policy
from .crop_focus import apply_crop_focus
from .data import backfill_series_metadata, load_series_csv, load_train_csv


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

    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    uids = train["StudyInstanceUID"].astype(str).tolist()
    uid = str(args.uid) if args.uid else uids[0]
    if uid not in set(uids):
        raise ValueError(f"unknown training StudyInstanceUID {uid}")

    # IMPORTANT: subset first. backfill_series_metadata may inspect DICOM files
    # for rows with incomplete metadata; a preview must never audit every series.
    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series = series.loc[series["StudyInstanceUID"].astype(str).eq(uid)].copy()
    if series.empty:
        raise ValueError("selected study has no rows in train_series.csv")
    print({"preview_uid": uid, "series_rows_before_backfill": int(len(series))})
    series, repair = backfill_series_metadata(series, root, split="train")
    print({"metadata_repair": repair})

    index = build_variable_series_index(series, [uid])
    records = index[uid]
    if not records:
        raise ValueError("selected study has no eligible series")

    ds = VariableSeriesKneeDataset(
        [uid],
        index,
        make_b7_dataset_config(config, root, train=False, tta_offsets=()),
        train=False,
    )
    item = ds[0]
    volumes = item["volumes"]
    focused = apply_crop_focus(volumes, policy)

    selected = []
    for plane in ("Sagittal", "Coronal", "Axial"):
        candidates = [i for i, record in enumerate(records) if record["plane"] == plane]
        if candidates:
            selected.append((plane, candidates[0]))
    if not selected:
        raise RuntimeError("no sagittal/coronal/axial series available for preview")

    fig, axes = plt.subplots(2, len(selected), figsize=(5 * len(selected), 8), squeeze=False)
    for col, (plane, k) in enumerate(selected):
        s = int(volumes.shape[1] // 2)
        original = volumes[k, s, 1].numpy()
        crop = focused[k, s, 1].numpy()
        axes[0, col].imshow(original, cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"{plane} — original")
        axes[1, col].imshow(crop, cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title(f"{plane} — B20 crop-only")
        for row in range(2):
            axes[row, col].axis("off")
    fig.suptitle(f"B20 crop-only preview | UID={uid} | policy={policy}", fontsize=11)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
