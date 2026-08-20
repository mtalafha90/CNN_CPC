"""The LLM-filled arm must be loadable, and must refuse what it cannot vouch for.

The two frozen arms are pinned to exact cell counts, so a changed export cannot
be trained on while still being called Phase-9 control or candidate. This arm is
a new surface with no counts to pin, which removes that protection -- so the two
things it *can* check matter more: that the merge overrode no parser call, and
that no gold study reached the training targets.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from rsna_knee.constants import TARGETS
from rsna_knee.phase9_supervision import load_fill_merged_export


def _write_export(root, *, overridden=0, gold_rows=0, uids=("a",)):
    root.mkdir(parents=True, exist_ok=True)
    frame = {"StudyInstanceUID": list(uids)}
    for target in TARGETS:
        frame[target] = [0.97] * len(uids)
        frame[f"{target}__confidence"] = [0.9] * len(uids)
        frame[f"{target}__state"] = ["positive"] * len(uids)
    pd.DataFrame(frame).to_csv(root / "training_targets.csv", index=False)

    (root / "audit.json").write_text(
        json.dumps(
            {
                "base_cells_overridden": overridden,
                "gold_rows_in_training_targets": gold_rows,
                "excluded_targets": ["Synovitis"],
            }
        ),
        encoding="utf-8",
    )
    (root / "policy.json").write_text(
        json.dumps(
            {
                "version": "b6_preserved_plus_b23_fill_v1",
                "base_version": "1.2.1",
                "filler_version": "1.0.0",
                "filler_provenance": {"model_id": "qwen3:14b"},
            }
        ),
        encoding="utf-8",
    )
    return root


def test_a_fill_merged_export_loads(tmp_path):
    root = _write_export(tmp_path / "merged")
    frame, policy, audit = load_fill_merged_export(root)
    assert len(frame) == 1
    assert policy["filler_provenance"]["model_id"] == "qwen3:14b"
    assert audit["excluded_targets"] == ["Synovitis"]


def test_an_export_that_overrode_parser_cells_is_refused(tmp_path):
    """Fill-only is what keeps the frozen specificity intact."""
    root = _write_export(tmp_path / "bad", overridden=12)
    with pytest.raises(ValueError, match="overrode base parser cells"):
        load_fill_merged_export(root)


def test_an_export_with_gold_rows_is_refused(tmp_path):
    root = _write_export(tmp_path / "leaky", gold_rows=3)
    with pytest.raises(ValueError, match="zero gold rows"):
        load_fill_merged_export(root)


def test_a_missing_artifact_is_named(tmp_path):
    root = tmp_path / "partial"
    root.mkdir()
    (root / "training_targets.csv").write_text("StudyInstanceUID\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="policy.json"):
        load_fill_merged_export(root)


def test_the_arm_requires_its_own_root():
    from rsna_knee.phase9_supervision import load_phase9_arm_supervision

    with pytest.raises(ValueError, match="requires llm_fill_root"):
        load_phase9_arm_supervision(
            pd.DataFrame(), arm="llm_fill", b6_root="a", phase8_root="b"
        )


def test_an_unknown_arm_names_the_three_that_exist():
    from rsna_knee.phase9_supervision import load_phase9_arm_supervision

    with pytest.raises(ValueError, match="control.*candidate.*llm_fill"):
        load_phase9_arm_supervision(
            pd.DataFrame(), arm="something", b6_root="a", phase8_root="b"
        )


def test_the_trainer_accepts_the_new_arm():
    from rsna_knee.phase9_matched_supervision_training import PHASE9_ARMS

    assert "llm_fill" in PHASE9_ARMS


def test_the_public_surface_name_maps_to_the_arm():
    from model._implementation import SUPERVISION_SURFACES

    assert SUPERVISION_SURFACES["llm-filled"] == "llm_fill"
    assert len(set(SUPERVISION_SURFACES.values())) == 3


def test_training_refuses_the_surface_without_its_labels():
    from model._implementation import train_working_model

    with pytest.raises(ValueError, match="requires llm_filled_labels_root"):
        train_working_model(
            {},
            supervision="llm-filled",
            latin_script_labels_root="a",
            all_script_labels_root="b",
            series_policy_path="c",
            encoder_checkpoint="d",
            out_root="e",
        )


def test_the_flag_reaches_the_command_line():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "training.train", "--help"],
        capture_output=True,
        text=True,
    )
    assert "llm-filled" in result.stdout
    assert "--llm-filled-labels" in result.stdout
