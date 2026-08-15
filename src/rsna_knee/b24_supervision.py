"""Build the matched B24 training surface from either labeller.

The single-variable guarantee lives here. Both arms must see:

  * the same studies, in the same order;
  * the same series, and therefore the same batch sequence;
  * the same optimiser trajectory.

and differ only in which cells inside those studies carry supervision, and what
state each carries. Everything in this module exists to enforce that, or to
report the difference the labels actually make.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .b23_llm_labels import load_frozen_b23_export
from .b23_validation_split import load_frozen_b23_holdout
from .b7_weak_supervision import load_frozen_b6_export, prepare_b7_supervision
from .b24_protocol import MODE_CANDIDATE, MODE_CONTROL
from .constants import TARGETS
from .data import gold_mask, load_train_csv
from .b15_ssl import load_frozen_v2_manifest


def _holdout_uids(weak_holdout_root: str | Path | None, b23_holdout_root: str | Path | None):
    """Every study that must stay out of gradients so it can be scored later."""
    excluded: set[str] = set()
    weak_v2: set[str] = set()
    b23_holdout: set[str] = set()
    if weak_holdout_root is not None:
        _payload, manifest = load_frozen_v2_manifest(weak_holdout_root)
        weak_v2 = set(
            manifest.loc[manifest["split"] == "holdout", "StudyInstanceUID"].astype(str)
        )
        excluded |= weak_v2
    if b23_holdout_root is not None:
        _payload, manifest = load_frozen_b23_holdout(b23_holdout_root)
        b23_holdout = set(
            manifest.loc[manifest["split"] == "holdout", "StudyInstanceUID"].astype(str)
        )
        excluded |= b23_holdout
    return excluded, weak_v2, b23_holdout


def build_matched_surface(
    config: dict,
    *,
    b6_root: str | Path,
    b23_root: str | Path,
    weak_holdout_root: str | Path | None = None,
    b23_holdout_root: str | Path | None = None,
) -> dict:
    """Return the shared study list plus each arm's targets and weights.

    The study list is the intersection of the two labellers' active sets, minus
    every holdout and every gold study. Using the intersection is what makes the
    batch sequence identical; the labels' effect then shows up as different
    cells within the same studies.
    """
    root = Path(config["data_root"])
    train = load_train_csv(root / config.get("train_csv", "train.csv"))

    b6_frame, _b6_policy, _b6_audit = load_frozen_b6_export(b6_root)
    b23_frame, _b23_policy, b23_audit = load_frozen_b23_export(b23_root)

    b6_uids, b6_y, b6_w, b6_summary = prepare_b7_supervision(train, b6_frame)
    b23_uids, b23_y, b23_w, b23_summary = prepare_b7_supervision(train, b23_frame)

    b6_index = {str(uid): i for i, uid in enumerate(b6_uids)}
    b23_index = {str(uid): i for i, uid in enumerate(b23_uids)}

    excluded, weak_v2_uids, b23_holdout_uids = _holdout_uids(
        weak_holdout_root, b23_holdout_root
    )
    gold_uids = set(train.loc[gold_mask(train), "StudyInstanceUID"].astype(str))
    excluded |= gold_uids

    # Active under both labellers, never held out, never gold. Sorted so the
    # order is reproducible and identical across arms and machines.
    b6_active = {str(u) for u, i in b6_index.items() if b6_w[i].sum() > 0}
    b23_active = {str(u) for u, i in b23_index.items() if b23_w[i].sum() > 0}
    shared = sorted((b6_active & b23_active) - excluded)
    if len(shared) < 2:
        raise ValueError("matched B24 surface has fewer than two studies")

    y_control = np.stack([b6_y[b6_index[u]] for u in shared])
    w_control = np.stack([b6_w[b6_index[u]] for u in shared])
    y_candidate = np.stack([b23_y[b23_index[u]] for u in shared])
    w_candidate = np.stack([b23_w[b23_index[u]] for u in shared])

    for name, uids_set in (("gold", gold_uids), ("holdout", excluded - gold_uids)):
        if set(shared) & uids_set:
            raise RuntimeError(f"{name} study leaked into the B24 training surface")

    return {
        "study_uids": shared,
        "control": {"targets": y_control, "weights": w_control},
        "candidate": {"targets": y_candidate, "weights": w_candidate},
        "diagnostics": surface_diagnostics(
            shared, y_control, w_control, y_candidate, w_candidate
        ),
        "excluded": {
            "gold": len(gold_uids),
            "weak_v2_holdout": len(weak_v2_uids),
            "b23_holdout": len(b23_holdout_uids),
        },
        "b6_active_studies": len(b6_active),
        "b23_active_studies": len(b23_active),
        "b23_cell_coverage": float(b23_audit.get("cell_coverage", float("nan"))),
        "b6_supervision": b6_summary,
        "b23_supervision": b23_summary,
    }


def surface_diagnostics(
    study_uids,
    y_control: np.ndarray,
    w_control: np.ndarray,
    y_candidate: np.ndarray,
    w_candidate: np.ndarray,
) -> dict:
    """Quantify exactly what the label swap changes, before any training.

    Worth reading before committing GPU time: if the two label sets barely
    differ, the experiment cannot show anything, and that is far cheaper to
    learn here than after a training run.
    """
    used_c = w_control > 0
    used_k = w_candidate > 0
    both = used_c & used_k
    # Disagreement only means anything where both labellers committed to a call.
    disagree = both & ((y_control > 0.5) != (y_candidate > 0.5))

    per_target = {}
    for j, target in enumerate(TARGETS):
        per_target[target] = {
            "control_cells": int(used_c[:, j].sum()),
            "candidate_cells": int(used_k[:, j].sum()),
            "added_by_candidate": int((used_k[:, j] & ~used_c[:, j]).sum()),
            "dropped_by_candidate": int((used_c[:, j] & ~used_k[:, j]).sum()),
            "disagreements": int(disagree[:, j].sum()),
        }

    return {
        "studies": int(len(study_uids)),
        "possible_cells": int(w_control.size),
        "control_usable_cells": int(used_c.sum()),
        "candidate_usable_cells": int(used_k.sum()),
        "cells_added_by_candidate": int((used_k & ~used_c).sum()),
        "cells_dropped_by_candidate": int((used_c & ~used_k).sum()),
        "cells_in_both": int(both.sum()),
        "disagreements_where_both_committed": int(disagree.sum()),
        "disagreement_rate": (
            float(disagree.sum() / both.sum()) if both.sum() else float("nan")
        ),
        "control_positive_cells": int((used_c & (y_control > 0.5)).sum()),
        "candidate_positive_cells": int((used_k & (y_candidate > 0.5)).sum()),
        "per_target": per_target,
    }


def arm_supervision(surface: dict, mode: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Pick one arm's targets and weights off a built surface."""
    if mode == MODE_CONTROL:
        arm = surface["control"]
    elif mode == MODE_CANDIDATE:
        arm = surface["candidate"]
    else:
        raise ValueError(f"unknown B24 mode {mode!r}")
    return surface["study_uids"], arm["targets"], arm["weights"]


def format_surface(surface: dict) -> str:
    d = surface["diagnostics"]
    added, dropped = d["cells_added_by_candidate"], d["cells_dropped_by_candidate"]
    return "\n".join(
        [
            "B24 matched training surface",
            f"  shared studies              {d['studies']}",
            f"  possible cells              {d['possible_cells']}",
            "",
            f"  B6 usable cells             {d['control_usable_cells']}"
            f"  ({d['control_usable_cells'] / d['possible_cells']:.1%})",
            f"  B23 usable cells            {d['candidate_usable_cells']}"
            f"  ({d['candidate_usable_cells'] / d['possible_cells']:.1%})",
            f"  added by B23                {added}",
            f"  dropped by B23              {dropped}",
            "",
            f"  cells both committed on     {d['cells_in_both']}",
            f"  disagreements there         {d['disagreements_where_both_committed']}"
            f"  ({d['disagreement_rate']:.1%})",
            "",
            f"  excluded gold               {surface['excluded']['gold']}",
            f"  excluded weak-v2 holdout    {surface['excluded']['weak_v2_holdout']}",
            f"  excluded B23 holdout        {surface['excluded']['b23_holdout']}",
            "",
            "  Both arms train on identical studies and batches; only the cells differ.",
        ]
    )
