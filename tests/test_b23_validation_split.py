import json

import numpy as np
import pandas as pd
import pytest

from rsna_knee.b23_llm_labels import run_b23_export
from rsna_knee.b23_validation_split import (
    B23_HOLDOUT_SURFACE,
    expected_standard_error,
    format_split,
    freeze_b23_holdout,
    load_frozen_b23_holdout,
    manifest_sha256,
)
from rsna_knee.constants import TARGETS

N_REPORT_STUDIES = 160


def _response_for(index):
    """Alternate confident positive/negated states so every class is populated."""
    findings = {}
    for j, target in enumerate(TARGETS):
        state = "positive" if (index + j) % 3 == 0 else "negated"
        findings[target] = {"state": state, "confidence": 0.9, "evidence": "evidence"}
    return json.dumps({"findings": findings})


def _build_export(tmp_path):
    rows = []
    for i in range(N_REPORT_STUDIES):
        row = {"StudyInstanceUID": f"uid-{i:04d}", "Report": f"distinct knee report {i}"}
        for target in TARGETS:
            row[target] = np.nan
        rows.append(row)
    gold = {"StudyInstanceUID": "uid-gold", "Report": "gold report"}
    for j, target in enumerate(TARGETS):
        gold[target] = float(j % 2)
    rows.append(gold)

    train = pd.DataFrame(rows)
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    train.to_csv(data_root / "train.csv", index=False)

    order = {f"distinct knee report {i}": i for i in range(N_REPORT_STUDIES)}

    def _backend(system, user):
        for text, index in order.items():
            if text in user:
                return _response_for(index)
        return _response_for(0)

    run_b23_export(
        data_root / "train.csv", _backend, out_root=tmp_path / "b23", progress_every=0
    )
    return {"data_root": str(data_root), "train_csv": "train.csv"}


def test_expected_standard_error_shrinks_with_more_studies():
    assert expected_standard_error(58) == pytest.approx(0.0250)
    assert expected_standard_error(800) < expected_standard_error(58)
    # Roughly 1/sqrt(n): a ~14x larger surface should more than halve the SE.
    assert expected_standard_error(800) == pytest.approx(0.0250 * np.sqrt(58 / 800))
    with pytest.raises(ValueError):
        expected_standard_error(0)


def test_manifest_sha256_is_order_independent():
    manifest = pd.DataFrame(
        {"StudyInstanceUID": ["b", "a", "c"], "split": ["train", "holdout", "train"]}
    )
    shuffled = manifest.iloc[[2, 0, 1]].reset_index(drop=True)
    assert manifest_sha256(manifest) == manifest_sha256(shuffled)


def test_manifest_sha256_changes_when_a_split_changes():
    manifest = pd.DataFrame({"StudyInstanceUID": ["a", "b"], "split": ["train", "holdout"]})
    moved = manifest.copy()
    moved.loc[0, "split"] = "holdout"
    assert manifest_sha256(manifest) != manifest_sha256(moved)


def test_freeze_b23_holdout_is_report_group_safe_and_excludes_gold(tmp_path):
    config = _build_export(tmp_path)
    payload = freeze_b23_holdout(
        config,
        b23_root=tmp_path / "b23",
        out_root=tmp_path / "holdout",
        holdout_fraction=0.25,
        n_candidates=256,
        min_class_count=2,
    )

    assert payload["surface"] == B23_HOLDOUT_SURFACE
    assert payload["report_group_overlap"] == 0
    assert payload["gold_labels_used"] is False
    assert payload["train_studies"] + payload["holdout_studies"] == payload["active_studies"]
    assert payload["active_studies"] == N_REPORT_STUDIES

    manifest = pd.read_csv(tmp_path / "holdout" / "manifest.csv")
    assert "uid-gold" not in set(manifest["StudyInstanceUID"].astype(str))
    train_groups = set(manifest.loc[manifest["split"] == "train", "report_group"])
    holdout_groups = set(manifest.loc[manifest["split"] == "holdout", "report_group"])
    assert train_groups.isdisjoint(holdout_groups)


def test_freeze_b23_holdout_is_deterministic_for_a_fixed_seed(tmp_path):
    config = _build_export(tmp_path)
    kwargs = dict(
        b23_root=tmp_path / "b23", holdout_fraction=0.25, n_candidates=256, min_class_count=2
    )
    first = freeze_b23_holdout(config, out_root=tmp_path / "a", **kwargs)
    second = freeze_b23_holdout(config, out_root=tmp_path / "b", **kwargs)
    assert first["manifest_sha256"] == second["manifest_sha256"]


def test_load_frozen_b23_holdout_detects_a_tampered_manifest(tmp_path):
    config = _build_export(tmp_path)
    freeze_b23_holdout(
        config,
        b23_root=tmp_path / "b23",
        out_root=tmp_path / "holdout",
        holdout_fraction=0.25,
        n_candidates=256,
        min_class_count=2,
    )
    payload, manifest = load_frozen_b23_holdout(tmp_path / "holdout")
    assert len(manifest) == payload["active_studies"]

    manifest.loc[manifest.index[0], "split"] = (
        "holdout" if manifest.loc[manifest.index[0], "split"] == "train" else "train"
    )
    manifest.to_csv(tmp_path / "holdout" / "manifest.csv", index=False)
    with pytest.raises(ValueError, match="SHA-256 does not match"):
        load_frozen_b23_holdout(tmp_path / "holdout")


def test_format_split_reports_the_resolution_gain(tmp_path):
    config = _build_export(tmp_path)
    payload = freeze_b23_holdout(
        config,
        b23_root=tmp_path / "b23",
        out_root=tmp_path / "holdout",
        holdout_fraction=0.25,
        n_candidates=256,
        min_class_count=2,
    )
    text = format_split(payload)
    assert "expected macro-AUC SE" in text
    assert "not expert truth" in text
