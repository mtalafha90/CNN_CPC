from __future__ import annotations

import json

import numpy as np
import pandas as pd

from rsna_knee.constants import TARGETS
from rsna_knee.report_teacher import COMPONENTS, _target_weights, run_report_teacher_benchmark


def test_target_weights_are_normalized_and_reward_skill():
    auc = {
        "rules": np.full(len(TARGETS), 0.90),
        "word": np.full(len(TARGETS), 0.70),
        "char": np.full(len(TARGETS), 0.50),
    }
    weights = _target_weights(auc, adaptive_strength=0.75)
    assert weights.shape == (3, len(TARGETS))
    np.testing.assert_allclose(weights.sum(axis=0), 1.0)
    assert np.all(weights[0] > weights[1])
    assert np.all(weights[1] > weights[2])


def test_target_weights_fall_back_to_equal_when_no_component_beats_chance():
    auc = {
        "rules": np.full(len(TARGETS), 0.40),
        "word": np.full(len(TARGETS), np.nan),
        "char": np.full(len(TARGETS), 0.50),
    }
    weights = _target_weights(auc)
    np.testing.assert_allclose(weights, np.full_like(weights, 1.0 / len(COMPONENTS)))


def test_report_teacher_benchmark_exports_complete_fold_safe_outputs(tmp_path):
    rows = []
    # Thirty gold reports and eighteen report-only studies.  All 12 targets use
    # the same synthetic label so every target has positives/negatives while the
    # text teacher has a simple but learnable signal.
    for i in range(48):
        positive = i % 2 == 0
        report = (
            f"synthetic study {i} marker {'abnormalpositive' if positive else 'normalnegative'} "
            f"knee magnetic resonance report"
        )
        row = {"StudyInstanceUID": f"study-{i:03d}", "Report": report}
        for target in TARGETS:
            row[target] = float(positive) if i < 30 else np.nan
        rows.append(row)

    train_csv = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(train_csv, index=False)
    out_dir = tmp_path / "teacher"

    payload = run_report_teacher_benchmark(
        train_csv,
        out_dir=out_dir,
        n_folds=3,
        seed=17,
        n_bootstrap=20,
    )

    assert payload["n_studies"] == 48
    assert payload["n_gold"] == 30
    assert payload["external_models"] is False
    assert payload["external_data"] is False
    assert set(payload["component_oof"]) == set(COMPONENTS)
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "oof.csv").exists()
    assert (out_dir / "fold_assignments.csv").exists()

    oof = pd.read_csv(out_dir / "oof.csv")
    assert len(oof) == 30
    assert oof["StudyInstanceUID"].is_unique
    assert np.isfinite(oof[TARGETS].to_numpy(dtype=float)).all()

    assignments = pd.read_csv(out_dir / "fold_assignments.csv")
    for fold in range(3):
        pseudo_path = out_dir / f"fold{fold}" / "pseudo_labels.csv"
        teacher_path = out_dir / f"fold{fold}" / "teacher.json"
        assert pseudo_path.exists()
        assert teacher_path.exists()
        pseudo = pd.read_csv(pseudo_path)
        assert len(pseudo) == 48
        assert pseudo["StudyInstanceUID"].is_unique
        assert np.isfinite(pseudo[TARGETS].to_numpy(dtype=float)).all()
        confidence_cols = [f"{target}__confidence" for target in TARGETS]
        confidence = pseudo[confidence_cols].to_numpy(dtype=float)
        assert np.isfinite(confidence).all()
        assert ((confidence >= 0.0) & (confidence <= 1.0)).all()

        outer_ids = set(assignments.loc[assignments["gold_fold"].eq(fold) & assignments["is_gold"], "StudyInstanceUID"])
        flagged = set(pseudo.loc[pseudo["is_outer_gold"], "StudyInstanceUID"])
        assert flagged == outer_ids

        metadata = json.loads(teacher_path.read_text())
        assert metadata["n_outer_gold"] == len(outer_ids)
        for target in TARGETS:
            weights = metadata["component_weights"][target]
            assert set(weights) == set(COMPONENTS)
            assert abs(sum(weights.values()) - 1.0) < 1e-6
