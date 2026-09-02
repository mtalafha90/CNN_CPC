"""Fill-only merging must add cells without ever changing a parser call.

The whole reason this formulation is usable is that it preserves the frozen
parser's decisions exactly, so the specificity clause that closed B23 does not
apply. If a single base call could be overridden, that argument collapses and
the merge would need the gate it was built to avoid.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from rsna_knee.b23_fill_merge import (
    FILL_BOTH_STATES,
    FILL_NEGATED_ONLY,
    merge_fill_only,
    write_merged_export,
)
from rsna_knee.constants import TARGETS


def _export(uids, states, confidences=None):
    """Build an export frame: `states` maps target -> per-study state."""
    frame = {"StudyInstanceUID": list(uids)}
    for target in TARGETS:
        column = states.get(target, ["unmentioned"] * len(uids))
        conf = (confidences or {}).get(target)
        if conf is None:
            conf = [0.9 if s in ("positive", "negated") else 0.0 for s in column]
        frame[target] = [
            0.97 if s == "positive" else 0.03 if s == "negated" else 0.5 for s in column
        ]
        frame[f"{target}__confidence"] = conf
        frame[f"{target}__state"] = column
    return pd.DataFrame(frame)


def test_a_silent_cell_is_filled():
    base = _export(["a"], {"ACL": ["unmentioned"]})
    filler = _export(["a"], {"ACL": ["positive"]})
    merged, audit = merge_fill_only(base, filler)

    assert merged["ACL__state"].iloc[0] == "positive"
    assert audit["targets"]["ACL"]["filled"] == 1
    assert audit["targets"]["ACL"]["filled_positive"] == 1


def test_a_committed_parser_call_is_never_overridden():
    """The clause that closed B23 turns on exactly this."""
    base = _export(["a"], {"ACL": ["negated"]})
    filler = _export(["a"], {"ACL": ["positive"]})
    merged, audit = merge_fill_only(base, filler)

    assert merged["ACL__state"].iloc[0] == "negated"
    assert audit["base_cells_overridden"] == 0
    assert audit["targets"]["ACL"]["filled"] == 0


def test_an_uncertain_filler_cell_does_not_fill():
    """Only a definite answer is worth adding; uncertainty supervises nothing."""
    base = _export(["a"], {"ACL": ["unmentioned"]})
    filler = _export(["a"], {"ACL": ["uncertain"]})
    merged, audit = merge_fill_only(base, filler)

    assert merged["ACL__state"].iloc[0] == "unmentioned"
    assert audit["filled_cells"] == 0


def test_a_low_confidence_filler_cell_does_not_fill():
    base = _export(["a"], {"ACL": ["unmentioned"]})
    filler = _export(["a"], {"ACL": ["positive"]}, {"ACL": [0.4]})
    merged, audit = merge_fill_only(base, filler)

    assert merged["ACL__state"].iloc[0] == "unmentioned"
    assert audit["filled_cells"] == 0


def test_a_study_the_filler_never_saw_is_left_alone():
    """A partial filler export must not blank the studies it skipped."""
    base = _export(["a", "b"], {"ACL": ["positive", "unmentioned"]})
    filler = _export(["a"], {"ACL": ["negated"]})
    merged, audit = merge_fill_only(base, filler)

    assert list(merged["StudyInstanceUID"]) == ["a", "b"]
    assert merged["ACL__state"].tolist() == ["positive", "unmentioned"]
    assert audit["studies_seen_by_filler"] == 1


def test_the_base_study_order_is_preserved():
    """Training compares arms study by study; reordering would break the match."""
    base = _export(["c", "a", "b"], {"ACL": ["unmentioned"] * 3})
    filler = _export(["a", "b", "c"], {"ACL": ["positive"] * 3})
    merged, _ = merge_fill_only(base, filler)
    assert list(merged["StudyInstanceUID"]) == ["c", "a", "b"]


def test_the_caller_frames_are_not_modified():
    base = _export(["a"], {"ACL": ["unmentioned"]})
    filler = _export(["a"], {"ACL": ["positive"]})
    merge_fill_only(base, filler)
    assert base["ACL__state"].iloc[0] == "unmentioned"


def test_coverage_accounting_adds_up():
    base = _export(["a", "b"], {"ACL": ["positive", "unmentioned"]})
    filler = _export(["a", "b"], {"ACL": ["negated", "negated"]})
    _, audit = merge_fill_only(base, filler)

    row = audit["targets"]["ACL"]
    assert row["base_committed"] == 1
    assert row["filled"] == 1
    assert row["final_committed"] == 2
    assert audit["final_committed_cells"] == (
        audit["base_committed_cells"] + audit["filled_cells"]
    )


def test_the_written_export_has_the_three_files_training_reads(tmp_path):
    base = _export(["a"], {"ACL": ["unmentioned"]})
    filler = _export(["a"], {"ACL": ["positive"]})
    merged, audit = merge_fill_only(base, filler)

    out = write_merged_export(
        tmp_path / "merged", merged, audit,
        base_audit={"b6_version": "1.2.1"},
        filler_audit={"b23_version": "1.0.0", "provenance": {"model_id": "qwen3:14b"}},
    )
    for name in ("training_targets.csv", "audit.json", "policy.json"):
        assert (out / name).is_file()

    written = json.loads((out / "audit.json").read_text(encoding="utf-8"))
    assert written["gold_rows_in_training_targets"] == 0
    assert written["base_cells_overridden"] == 0

    policy = json.loads((out / "policy.json").read_text(encoding="utf-8"))
    assert policy["filler_provenance"]["model_id"] == "qwen3:14b"


def test_the_written_columns_match_what_the_loaders_expect(tmp_path):
    base = _export(["a"], {})
    filler = _export(["a"], {})
    merged, audit = merge_fill_only(base, filler)
    out = write_merged_export(
        tmp_path / "m", merged, audit, base_audit={}, filler_audit={}
    )
    written = pd.read_csv(out / "training_targets.csv")
    expected = ["StudyInstanceUID"]
    for target in TARGETS:
        expected.extend([target, f"{target}__confidence", f"{target}__state"])
    assert list(written.columns) == expected


def test_an_export_missing_its_columns_is_refused(tmp_path):
    from rsna_knee.b23_fill_merge import _read_export

    root = tmp_path / "broken"
    root.mkdir()
    pd.DataFrame({"StudyInstanceUID": ["a"]}).to_csv(
        root / "training_targets.csv", index=False
    )
    (root / "audit.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        _read_export(root)


def test_a_duplicated_study_is_refused(tmp_path):
    from rsna_knee.b23_fill_merge import _read_export

    root = tmp_path / "dupes"
    root.mkdir()
    _export(["a", "a"], {}).to_csv(root / "training_targets.csv", index=False)
    (root / "audit.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="more than once"):
        _read_export(root)


def test_an_excluded_target_is_left_entirely_to_the_base():
    """Synovitis is the case: B26 filled it carefully and expert AUC fell."""
    base = _export(["a"], {"ACL": ["unmentioned"], "Synovitis": ["unmentioned"]})
    filler = _export(["a"], {"ACL": ["positive"], "Synovitis": ["negated"]})
    merged, audit = merge_fill_only(base, filler, exclude_targets=("Synovitis",))

    assert merged["ACL__state"].iloc[0] == "positive"
    assert merged["Synovitis__state"].iloc[0] == "unmentioned"
    assert audit["targets"]["Synovitis"]["filled"] == 0
    assert audit["targets"]["Synovitis"]["excluded_from_fill"] is True
    assert audit["excluded_targets"] == ["Synovitis"]


def test_excluding_a_target_does_not_disturb_the_totals():
    base = _export(["a"], {"ACL": ["unmentioned"], "Synovitis": ["positive"]})
    filler = _export(["a"], {"ACL": ["positive"], "Synovitis": ["negated"]})
    _, audit = merge_fill_only(base, filler, exclude_targets=("Synovitis",))
    assert audit["final_committed_cells"] == (
        audit["base_committed_cells"] + audit["filled_cells"]
    )


def test_an_unknown_target_name_is_refused():
    base = _export(["a"], {})
    filler = _export(["a"], {})
    with pytest.raises(ValueError, match="unknown target"):
        merge_fill_only(base, filler, exclude_targets=("Synovytis",))


# --- filling only what the parser is reliable about ------------------------
#
# Measured on the 58 expert studies, the filler's two states are not alike:
#
#     negated cells      137    97.8% correct
#     positive cells     305    67.2% correct
#
# and a filled cell is by definition one the base declined to answer, so
# "refuse any positive the parser does not corroborate" and "fill negated only"
# are the same rule. These pin that rule, and that the default is unchanged.


def _model_confidence(frame, values):
    """Attach the labeller's own confidence, which training_targets.csv lacks."""
    frame = frame.copy()
    for target in TARGETS:
        frame[f"{target}__model_confidence"] = values.get(target, [1.0] * len(frame))
    return frame


def test_negated_only_adds_negatives_and_refuses_positives():
    base = _export(["a", "b"], {"ACL": ["unmentioned", "unmentioned"]})
    filler = _export(["a", "b"], {"ACL": ["positive", "negated"]})

    merged, audit = merge_fill_only(base, filler, fill_states=FILL_NEGATED_ONLY)

    assert merged["ACL__state"].tolist() == ["unmentioned", "negated"]
    assert audit["targets"]["ACL"]["filled_positive"] == 0
    assert audit["targets"]["ACL"]["filled_negative"] == 1


def test_negated_only_still_preserves_every_parser_call():
    """Narrowing what may be filled must not touch what the base already said."""
    base = _export(["a", "b"], {"ACL": ["positive", "negated"]})
    filler = _export(["a", "b"], {"ACL": ["negated", "positive"]})

    merged, audit = merge_fill_only(base, filler, fill_states=FILL_NEGATED_ONLY)

    assert merged["ACL__state"].tolist() == ["positive", "negated"]
    assert audit["targets"]["ACL"]["base_overridden"] == 0
    assert audit["base_cells_overridden"] == 0


def test_the_default_still_fills_both_states():
    """Every completed run used this. It must not move."""
    import inspect  # noqa: PLC0415

    default = inspect.signature(merge_fill_only).parameters["fill_states"].default
    assert default == FILL_BOTH_STATES

    base = _export(["a", "b"], {"ACL": ["unmentioned", "unmentioned"]})
    filler = _export(["a", "b"], {"ACL": ["positive", "negated"]})
    _merged, audit = merge_fill_only(base, filler)
    assert audit["targets"]["ACL"]["filled"] == 2


def test_the_policy_is_written_into_the_audit():
    """A run whose rule cannot be read afterwards cannot be reproduced."""
    base = _export(["a"], {"ACL": ["unmentioned"]})
    filler = _export(["a"], {"ACL": ["negated"]})
    _merged, audit = merge_fill_only(base, filler, fill_states=FILL_NEGATED_ONLY)
    assert audit["fill_states"] == ["negated"]
    assert audit["min_model_confidence"] is None


def test_an_empty_or_unknown_fill_state_is_refused():
    base = _export(["a"], {"ACL": ["unmentioned"]})
    filler = _export(["a"], {"ACL": ["negated"]})
    for bad in ((), ("uncertain",), ("positive", "nonsense")):
        with pytest.raises(ValueError, match="fill_states must be"):
            merge_fill_only(base, filler, fill_states=bad)


# --- the labeller's own confidence, recorded but never used ---------------


def test_the_model_confidence_floor_filters_filled_cells():
    base = _export(["a", "b"], {"ACL": ["unmentioned", "unmentioned"]})
    filler = _model_confidence(
        _export(["a", "b"], {"ACL": ["negated", "negated"]}),
        {"ACL": [0.80, 1.00]},
    )
    merged, audit = merge_fill_only(filler=filler, base=base, min_model_confidence=0.99)

    assert merged["ACL__state"].tolist() == ["unmentioned", "negated"]
    assert audit["targets"]["ACL"]["filled"] == 1
    assert audit["min_model_confidence"] == 0.99


def test_asking_for_a_confidence_the_export_does_not_carry_says_which_file_to_use():
    """training_targets.csv has no __model_confidence. Silence here would be a
    filter that quietly does nothing."""
    base = _export(["a"], {"ACL": ["unmentioned"]})
    filler = _export(["a"], {"ACL": ["negated"]})
    with pytest.raises(ValueError, match="structured_labels.csv"):
        merge_fill_only(base, filler, min_model_confidence=0.9)


def test_the_two_filters_compose():
    base = _export(["a", "b", "c"], {"ACL": ["unmentioned"] * 3})
    filler = _model_confidence(
        _export(["a", "b", "c"], {"ACL": ["positive", "negated", "negated"]}),
        {"ACL": [1.00, 0.50, 1.00]},
    )
    merged, _audit = merge_fill_only(
        base, filler, fill_states=FILL_NEGATED_ONLY, min_model_confidence=0.9
    )
    # a: positive, refused by state. b: negated but unconfident. c: both pass.
    assert merged["ACL__state"].tolist() == ["unmentioned", "unmentioned", "negated"]
