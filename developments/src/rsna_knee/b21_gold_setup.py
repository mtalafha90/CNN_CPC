from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .b7_weak_supervision import make_b7_dataset_config
from .b12_variable_series import build_variable_series_index, collate_variable_series
from .b12_1_gold_eval import predict_b12_1
from .b12_1_hierarchical import build_b12_1_model
from .b18_fisher_selection import B18_EXPECTED_GOLD_SERIES, B18_EXPECTED_GOLD_STUDIES
from .b20_crop_focus import CropFocusedVariableSeriesKneeDataset
from .b21_acceptance_protocol import (
    B21_CROP_FRACTION,
    B21_FIXED_EPOCHS,
    B21_FULL_EXPERIMENT,
    B21_FULL_TRAIN_SERIES,
    B21_FULL_TRAIN_STUDIES,
    B21_FULL_VARIANT,
)
from .b21_dataset import make_matched_crop_dataset
from .constants import TARGETS
from .crop_focus import CROP_FOCUS_VERSION
from .data import backfill_series_metadata, gold_mask, load_series_csv, load_train_csv


def load_b21_full(path: str | Path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("variant") != B21_FULL_VARIANT or payload.get("experiment") != B21_FULL_EXPERIMENT:
        raise ValueError("candidate checkpoint is not the frozen full-data B21 experiment")
    if payload.get("mode") != "full_preresize":
        raise ValueError("candidate checkpoint has the wrong B21 mode")
    if int(payload.get("completed_epochs", -1)) != B21_FIXED_EPOCHS:
        raise ValueError("B21 acceptance requires fixed E2")
    if int(payload.get("model_epoch", -1)) != B21_FIXED_EPOCHS:
        raise ValueError("B21 checkpoint model_epoch is not 2")
    if int(payload.get("training_studies", -1)) != B21_FULL_TRAIN_STUDIES:
        raise ValueError("B21 was not refit on all 3,120 B6 studies")
    if int(payload.get("training_series", -1)) != B21_FULL_TRAIN_SERIES:
        raise ValueError("B21 was not refit on all 17,475 B6 series")
    if payload.get("crop_stage") != "native_array_pre_resize":
        raise ValueError("B21 does not certify pre-resize cropping")
    if payload.get("weak_v2_gate_passed_before_full_refit") is not True:
        raise ValueError("B21 lacks the passed weak-v2 gate certificate")
    if payload.get("gold_labels_used_for_development") is not False:
        raise ValueError("B21 does not certify zero gold development use")
    if int(payload.get("gold_studies_used_in_gradient", -1)) != 0:
        raise ValueError("B21 does not certify zero gold gradient use")
    model = build_b12_1_model(payload["model_spec"], pretrained_weights=False)
    model.load_state_dict(payload["model_state"], strict=True)
    model = model.to(device)
    model.eval()
    return model, payload


def gold_surface(config: dict):
    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))
    gold = train.loc[gold_mask(train), ["StudyInstanceUID", *TARGETS]].copy()
    gold["StudyInstanceUID"] = gold["StudyInstanceUID"].astype(str)
    if len(gold) != B18_EXPECTED_GOLD_STUDIES or gold[TARGETS].isna().any().any():
        raise ValueError("B21 acceptance requires complete 58-study gold surface")
    uids = gold["StudyInstanceUID"].tolist()
    truth = gold[TARGETS].to_numpy(np.float64)
    series = load_series_csv(root / config.get("train_series_csv", "train_series.csv"))
    series, metadata = backfill_series_metadata(series, root, split="train")
    index = build_variable_series_index(series, uids)
    counts = [len(index.get(uid, [])) for uid in uids]
    if any(count == 0 for count in counts) or int(sum(counts)) != B18_EXPECTED_GOLD_SERIES:
        raise ValueError("B21 acceptance gold series surface changed")
    return root, uids, truth, index, counts, metadata


def make_gold_datasets(config: dict, root: Path, uids, index):
    offsets = tuple(int(x) for x in config.get("b7_eval_tta_offsets", [-1, 0, 1]))
    if offsets != (-1, 0, 1):
        raise ValueError("B21 acceptance freezes TTA [-1,0,1]")
    control = CropFocusedVariableSeriesKneeDataset(
        uids,
        index,
        make_b7_dataset_config(config, root, train=False, tta_offsets=offsets),
        train=False,
        crop_focus_policy={"version": CROP_FOCUS_VERSION, "crop_fraction": B21_CROP_FRACTION},
    )
    candidate = make_matched_crop_dataset(
        "preresize",
        uids,
        index,
        make_b7_dataset_config(config, root, train=False, tta_offsets=offsets),
        crop_fraction=B21_CROP_FRACTION,
        train=False,
    )
    return offsets, control, candidate


def predict_gold(model, dataset, uids, runtime, seed: int):
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_variable_series,
        **runtime.loader_kwargs(seed=seed),
    )
    pred_uids, prediction = predict_b12_1(model, loader, runtime)
    if pred_uids != list(uids):
        raise RuntimeError("gold prediction order changed")
    return prediction
