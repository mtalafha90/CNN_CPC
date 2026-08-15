"""The matched-surface guarantee: identical studies, different cells only."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from rsna_knee.b24_protocol import MODE_CANDIDATE, MODE_CONTROL
from rsna_knee.b24_supervision import (
    arm_supervision,
    build_matched_surface,
    format_surface,
    surface_diagnostics,
)
from rsna_knee.constants import TARGETS

N_STUDIES = 40


def _export(tmp_path, name, states_for, *, version, experiment, extra_audit=None):
    """Write a minimal B6- or B23-shaped export."""
    rows = []
    for i in range(N_STUDIES):
        row = {"StudyInstanceUID": f"uid-{i:03d}"}
        for j, target in enumerate(TARGETS):
            state = states_for(i, j)
            row[target] = {"positive": 0.97, "negated": 0.03}.get(state, 0.50)
            row[f"{target}__confidence"] = 0.90 if state in ("positive", "negated") else 0.0
            row[f"{target}__state"] = state
        rows.append(row)
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "training_targets.csv", index=False)
    audit = {
        "gold_rows_in_training_targets": 0,
        "min_confidence_for_usable_cell": 0.75,
        "cell_coverage": 0.5,
    }
    audit.update(extra_audit or {})
    policy = {"version": version, "unmentioned_is_negative": False}
    if version == "1.2.1":
        audit["b6_version"] = version
        policy["experiment"] = "B6"
    else:
        audit["b23_version"] = version
        audit["external_model_reproducible"] = True
        policy["experiment"] = experiment
    (root / "audit.json").write_text(json.dumps(audit), encoding="utf-8")
    (root / "policy.json").write_text(json.dumps(policy), encoding="utf-8")
    return root


def _holdout(tmp_path, name, surface, held, *, gate_passed=True):
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(
        {
            "StudyInstanceUID": [f"uid-{i:03d}" for i in range(N_STUDIES)],
            "split": ["holdout" if i in held else "train" for i in range(N_STUDIES)],
        }
    )
    manifest.to_csv(root / "manifest.csv", index=False)
    from rsna_knee.b23_validation_split import manifest_sha256

    payload = {
        "surface": surface,
        "manifest_sha256": manifest_sha256(manifest),
        "active_studies": N_STUDIES,
        "labeller_gate": {"passed": gate_passed, "reasons": []},
    }
    (root / "weak_holdout.json").write_text(json.dumps(payload), encoding="utf-8")
    # weak-v2 uses a differently named manifest file.
    manifest.to_csv(root / "weak_holdout_manifest.csv", index=False)
    return root


def _fixture(tmp_path):
    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(N_STUDIES):
        row = {"StudyInstanceUID": f"uid-{i:03d}", "Report": f"report {i}"}
        for target in TARGETS:
            row[target] = np.nan
        rows.append(row)
    gold = {"StudyInstanceUID": "uid-gold", "Report": "gold"}
    for j, target in enumerate(TARGETS):
        gold[target] = float(j % 2)
    rows.append(gold)
    pd.DataFrame(rows).to_csv(data / "train.csv", index=False)

    # B6: sparse. Only every third cell is committed.
    b6 = _export(
        tmp_path, "b6",
        lambda i, j: "positive" if (i + j) % 3 == 0 else ("negated" if (i + j) % 5 == 0 else "unmentioned"),
        version="1.2.1", experiment="B6",
    )
    # B23: denser, and it disagrees with B6 on some shared cells.
    b23 = _export(
        tmp_path, "b23",
        lambda i, j: "negated" if (i + j) % 3 == 0 else ("positive" if (i + j) % 2 == 0 else "unmentioned"),
        version="1.0.0", experiment="B23_llm_report_labels",
    )
    return {"data_root": str(data), "train_csv": "train.csv"}, b6, b23


def test_both_arms_receive_identical_studies(tmp_path):
    config, b6, b23 = _fixture(tmp_path)
    surface = build_matched_surface(config, b6_root=b6, b23_root=b23)

    control_uids, y_c, w_c = arm_supervision(surface, MODE_CONTROL)
    candidate_uids, y_k, w_k = arm_supervision(surface, MODE_CANDIDATE)

    # This is the single-variable guarantee: same studies, same order, same
    # shape, so the batch sequence and optimiser trajectory are identical.
    assert control_uids == candidate_uids
    assert y_c.shape == y_k.shape == w_c.shape == w_k.shape
    assert len(control_uids) == len(set(control_uids))


def test_the_arms_actually_differ_in_their_cells(tmp_path):
    config, b6, b23 = _fixture(tmp_path)
    surface = build_matched_surface(config, b6_root=b6, b23_root=b23)
    _uids, y_c, w_c = arm_supervision(surface, MODE_CONTROL)
    _uids2, y_k, w_k = arm_supervision(surface, MODE_CANDIDATE)
    # If the label sets were identical the experiment could not show anything.
    assert not np.array_equal(w_c, w_k) or not np.array_equal(y_c, y_k)


def test_gold_studies_never_enter_the_training_surface(tmp_path):
    config, b6, b23 = _fixture(tmp_path)
    surface = build_matched_surface(config, b6_root=b6, b23_root=b23)
    assert "uid-gold" not in surface["study_uids"]
    assert surface["excluded"]["gold"] == 1


def test_the_b23_holdout_is_excluded_from_gradients(tmp_path):
    config, b6, b23 = _fixture(tmp_path)
    b23_hold = _holdout(tmp_path, "b23hold", "b23_llm_holdout_v1", held={4, 5, 6})

    surface = build_matched_surface(
        config, b6_root=b6, b23_root=b23, b23_holdout_root=b23_hold
    )
    trained = set(surface["study_uids"])
    for i in (4, 5, 6):
        assert f"uid-{i:03d}" not in trained, "a scored study leaked into training"
    assert surface["excluded"]["b23_holdout"] == 3


def test_a_fabricated_weak_v2_manifest_is_refused(tmp_path):
    """weak-v2 is frozen, so B24 must not accept a substitute for it.

    The loader pins the manifest SHA-256 recorded when the surface was frozen
    before B15. Anything else would let a re-derived split silently replace the
    one every historical comparison was made against.
    """
    config, b6, b23 = _fixture(tmp_path)
    fake = _holdout(tmp_path, "weakv2", "weak_b6_holdout_v2", held={0, 1, 2, 3})
    with pytest.raises(ValueError, match="SHA mismatch"):
        build_matched_surface(
            config, b6_root=b6, b23_root=b23, weak_holdout_root=fake
        )


def test_surface_diagnostics_quantify_the_label_swap():
    y_c = np.full((5, len(TARGETS)), 0.5)
    w_c = np.zeros((5, len(TARGETS)))
    y_k = np.full((5, len(TARGETS)), 0.5)
    w_k = np.zeros((5, len(TARGETS)))

    # Shared cell where both commit and agree.
    y_c[0, 0], w_c[0, 0] = 0.85, 0.5
    y_k[0, 0], w_k[0, 0] = 0.85, 0.5
    # Shared cell where both commit and disagree.
    y_c[1, 1], w_c[1, 1] = 0.85, 0.5
    y_k[1, 1], w_k[1, 1] = 0.05, 1.0
    # Cell only the candidate recovers.
    y_k[2, 2], w_k[2, 2] = 0.85, 0.5
    # Cell only the control has.
    y_c[3, 3], w_c[3, 3] = 0.05, 1.0

    d = surface_diagnostics(["a", "b", "c", "d", "e"], y_c, w_c, y_k, w_k)
    assert d["control_usable_cells"] == 3
    assert d["candidate_usable_cells"] == 3
    assert d["cells_added_by_candidate"] == 1
    assert d["cells_dropped_by_candidate"] == 1
    assert d["cells_in_both"] == 2
    assert d["disagreements_where_both_committed"] == 1
    assert d["disagreement_rate"] == pytest.approx(0.5)


def test_format_surface_states_the_single_variable_guarantee(tmp_path):
    config, b6, b23 = _fixture(tmp_path)
    surface = build_matched_surface(config, b6_root=b6, b23_root=b23)
    text = format_surface(surface)
    assert "identical studies and batches" in text
    assert "disagreements" in text


def test_an_unknown_arm_is_rejected(tmp_path):
    config, b6, b23 = _fixture(tmp_path)
    surface = build_matched_surface(config, b6_root=b6, b23_root=b23)
    with pytest.raises(ValueError, match="unknown B24 mode"):
        arm_supervision(surface, "whatever")
