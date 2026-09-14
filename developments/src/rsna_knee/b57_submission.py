"""Turn a finished B57 arm into a competition submission.

Every other submission module here is B52-lineage: it rebuilds a hierarchy on
top of a base checkpoint and reads a payload shaped by `train_b52`. B57's
candidate is a different model with a different payload and no base checkpoint
at all, so none of them apply.

## The failure this module is shaped around

B41 once scored nothing on the hidden set for reasons that had nothing to do
with the model: the notebook path diverged from the training path, and the
first evidence of it was the leaderboard. A submission is the most expensive
place in this project to discover a bug -- it costs a slot and a day, and it
returns one number that cannot tell you whether the model or the plumbing was
at fault.

So `--verify` is not optional decoration. It re-runs this module's own
inference over the **validation** studies and checks the macro AUC against the
number the training run recorded in `final.pt`. Same weights, same studies, a
separately written loader: if the two disagree, the inference path is wrong and
the submission would have been wrong with it. Run it before spending the slot.

## What it does not do

It does not select a checkpoint. B57 declares `fixed_final_epoch` before it
runs, so the endpoint is the endpoint -- even when the curve peaked earlier,
which for the DINOv2 candidate it did. Submitting a better intermediate epoch
would be the post-hoc selection the protocol exists to prevent, and the weights
are gone in any case: the trainer keeps one rolling recovery file.

It also carries no test-time augmentation. The training run scored its
validation surface at a single centre offset, and a submission that quietly
averaged three views would not be the model that produced the recorded number.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .b12_variable_series import audit_variable_series_surface
from .b42_constant_area_aspect_sparse_mil import (
    B42ConstantAreaAspectDataset,
    require_b42_contract,
)
from .b57_models import ARMS, build_model
from .b57_protocol import VERSION
from .b57_training import make_loader, predict, runtime_for
from .b7_weak_supervision import make_b7_dataset_config
from .constants import SUBMISSION_COLUMNS, TARGETS
from .data import backfill_series_metadata, load_series_csv, load_test_csv

B57_SUBMISSION_VERSION = "b57_submission_v1"

#: How far the reproduced validation score may sit from the recorded one before
#: the submission is refused. Inference is deterministic here -- no augmentation,
#: one centre offset, `predict` under `no_grad` -- so the only expected drift is
#: floating-point ordering under autocast. Anything larger is a real divergence.
REPRODUCTION_TOLERANCE = 5e-4


def load_endpoint(path: str | Path) -> dict:
    """The finished arm, refused unless it really is finished."""
    payload = torch.load(str(Path(path)), map_location="cpu", weights_only=False)
    if payload.get("version") != VERSION:
        raise ValueError(f"not a {VERSION} checkpoint: {payload.get('version')!r}")
    if payload.get("arm") not in ARMS:
        raise ValueError(f"unknown B57 arm: {payload.get('arm')!r}")
    if payload.get("selection") != "fixed_final_epoch":
        raise ValueError(
            "B57 submits its declared endpoint. A payload with any other "
            "selection rule was chosen after seeing its own curve."
        )
    completed = int(payload.get("completed_epochs", -1))
    planned = int(payload["model_config"]["epochs"])
    if completed != planned:
        raise ValueError(
            f"this arm stopped at {completed} of {planned} epochs; `final.pt` is "
            "only written when the declared schedule finishes"
        )
    return payload


def rebuild_model(payload: dict, device):
    """The architecture from the payload's own record, then its own weights.

    `public_root=None` builds the layers without fetching the public encoder,
    which would be overwritten by `model_state` immediately afterwards and is
    not available inside an offline notebook anyway.
    """
    model = build_model(
        payload["arm"], payload["b42_config"], payload["model_config"], public_root=None
    )
    missing, unexpected = model.load_state_dict(payload["model_state"], strict=False)
    if missing or unexpected:
        raise ValueError(
            f"checkpoint does not fit the rebuilt model: missing={sorted(missing)[:5]} "
            f"unexpected={sorted(unexpected)[:5]}"
        )
    return model.eval().to(device)


def test_surface(data_root: str | Path, payload: dict, *, split: str = "test"):
    """The studies to predict, with placeholder labels.

    `predict` reads `target` and `weight` off each item because it is shared
    with the training path. The test set has neither, so zeros are passed and
    never read: nothing downstream of the probabilities touches them.
    """
    root = Path(data_root).expanduser().resolve()
    frame = load_test_csv(root / f"{split}.csv")
    uids = [str(uid) for uid in frame["StudyInstanceUID"]]

    series = load_series_csv(root / f"{split}_series.csv")
    series, repair = backfill_series_metadata(series, root, split=split)
    _summary, index = audit_variable_series_surface(series, uids)

    settings = dict(payload["b42_config"])
    settings["strict_dicom"] = True
    config = make_b7_dataset_config(settings, root, train=False)
    config.strict_dicom = True
    config.tta_center_offsets = ()

    placeholder = np.zeros((len(uids), len(TARGETS)), dtype=np.float32)
    dataset = B42ConstantAreaAspectDataset(
        uids, index, config,
        crop_focus_policy=require_b42_contract(settings),
        center_offsets=(0,), targets=placeholder, weights=placeholder,
    )
    return dataset, uids, repair


def validation_surface(run_root: str | Path, payload: dict):
    """The studies the training run scored, rebuilt from the frozen protocol."""
    from .b57_protocol import load_protocol
    from .b57_training import make_dataset

    protocol_root = Path(run_root) / "protocol"
    return make_dataset(protocol_root, load_protocol(protocol_root), "validation")


def recorded_macro_auc(payload: dict) -> float:
    """What the training run itself reported for the endpoint epoch."""
    history = payload.get("history") or []
    if not history:
        raise ValueError("this payload carries no history to verify against")
    return float(history[-1]["validation"]["macro_auc"])


def verify(run_root: str | Path, payload: dict, model, runtime) -> dict:
    """Re-score the validation studies through this module's own path.

    The point is not to recompute a number we already have. It is to prove that
    the loader, the model rebuild and the probability extraction in *this* file
    agree with the ones that produced it -- before a slot is spent finding out
    they do not.
    """
    from .b52_competition_training import macro_auc

    dataset = validation_surface(run_root, payload)
    predicted = predict(model, make_loader(dataset, runtime, payload["model_config"]), runtime)
    scores = macro_auc(predicted["target"], predicted["weight"], predicted["prediction"])

    expected = recorded_macro_auc(payload)
    delta = float(scores["macro_auc"]) - expected
    result = {
        "recorded_macro_auc": expected,
        "reproduced_macro_auc": float(scores["macro_auc"]),
        "delta": delta,
        "tolerance": REPRODUCTION_TOLERANCE,
        "studies": len(predicted["uids"]),
        "agrees": abs(delta) <= REPRODUCTION_TOLERANCE,
    }
    if not result["agrees"]:
        raise ValueError(
            f"this module reproduces {scores['macro_auc']:.6f} where the training run "
            f"recorded {expected:.6f} (delta {delta:+.6f}). The inference path "
            "disagrees with the training path, so the submission would be wrong "
            "in a way the leaderboard could not explain. Do not submit."
        )
    return result


def submission_frame(uids, probabilities: np.ndarray) -> pd.DataFrame:
    """One row per study, the twelve columns in the competition's order."""
    if probabilities.shape != (len(uids), len(TARGETS)):
        raise ValueError(
            f"expected {(len(uids), len(TARGETS))} probabilities, got {probabilities.shape}"
        )
    if not np.isfinite(probabilities).all():
        raise ValueError("nonfinite probability in the submission")
    if (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError("submission values must be probabilities")
    frame = pd.DataFrame(probabilities, columns=TARGETS)
    frame.insert(0, "StudyInstanceUID", list(uids))
    return frame[SUBMISSION_COLUMNS]


def generate(
    *,
    run_root: str | Path,
    arm: str = ARMS[1],
    data_root: str | Path,
    out_path: str | Path = "submission.csv",
    device: str = "auto",
    workers: int = 0,
    split: str = "test",
    skip_verify: bool = False,
) -> Path:
    checkpoint = Path(run_root) / arm / "final.pt"
    payload = load_endpoint(checkpoint)
    runtime = runtime_for(device, workers)
    print(f"[B57 submit] {runtime.describe()}", flush=True)
    print(
        f"[B57 submit] {payload['arm']} endpoint, {payload['completed_epochs']} epochs, "
        f"recorded macro AUC {recorded_macro_auc(payload):.6f}",
        flush=True,
    )

    model = rebuild_model(payload, runtime.device)

    if skip_verify:
        print(
            "[B57 submit] verification skipped. The last time an inference path "
            "diverged unnoticed in this project, the leaderboard was where it "
            "surfaced.",
            flush=True,
        )
    else:
        print("[B57 submit] reproducing the recorded validation score", flush=True)
        agreement = verify(run_root, payload, model, runtime)
        print(
            f"[B57 submit] reproduced {agreement['reproduced_macro_auc']:.6f} "
            f"against {agreement['recorded_macro_auc']:.6f} "
            f"(delta {agreement['delta']:+.2e}) on {agreement['studies']} studies",
            flush=True,
        )

    dataset, uids, repair = test_surface(data_root, payload, split=split)
    print(f"[B57 submit] {len(uids)} {split} studies; metadata repair {repair}", flush=True)
    predicted = predict(model, make_loader(dataset, runtime, payload["model_config"]), runtime)

    order = {uid: i for i, uid in enumerate(predicted["uids"].tolist())}
    if set(order) != set(uids):
        raise ValueError("predictions do not cover exactly the studies requested")
    probabilities = predicted["prediction"][[order[uid] for uid in uids]]

    frame = submission_frame(uids, probabilities)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"[B57 submit] wrote {out} with {len(frame)} rows", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        "rsna-knee-b57-submission",
        description="Write a submission from a finished B57 arm.",
    )
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--arm", default=ARMS[1], choices=ARMS)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-path", default="submission.csv")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--split", default="test", choices=("test", "train"))
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help=(
            "do not reproduce the recorded validation score first. Only for a "
            "notebook that has no access to the training labels."
        ),
    )
    args = parser.parse_args()

    generate(
        run_root=args.run_root,
        arm=args.arm,
        data_root=args.data_root,
        out_path=args.out_path,
        device=args.device,
        workers=args.num_workers,
        split=args.split,
        skip_verify=args.skip_verify,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "B57_SUBMISSION_VERSION",
    "REPRODUCTION_TOLERANCE",
    "generate",
    "load_endpoint",
    "rebuild_model",
    "recorded_macro_auc",
    "submission_frame",
    "test_surface",
    "verify",
]
