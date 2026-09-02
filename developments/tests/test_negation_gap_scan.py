"""Counting the one real parser defect the expert reading turned up.

The pattern has to be narrow. A loose one matches everywhere and would turn a
single observed bug into a fake systematic finding, which is worse than not
looking. These tests pin both directions: the layout it must catch, and the
prose it must leave alone.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from rsna_knee.constants import TARGETS
from rsna_knee.negation_gap_scan import matches, scan


def _export(uids, states, *, evidence=None, gold=None):
    frame = {"StudyInstanceUID": list(uids)}
    if gold is not None:
        frame["is_gold"] = list(gold)
    for target in TARGETS:
        column = states.get(target, ["unmentioned"] * len(uids))
        frame[f"{target}__state"] = column
        frame[f"{target}__evidence"] = (evidence or {}).get(target, [""] * len(uids))
    return pd.DataFrame(frame)


def _write(tmp_path, frame, name="structured_labels.csv"):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


# --- what the pattern must catch ---------------------------------------------


@pytest.mark.parametrize(
    "clause",
    [
        "baker cyst: none",
        "Baker cyst: None",
        "meniscus: no tear",
        "effusion: nil",
        "acl: intact",
        "mcl - normal",
        "joint effusion: absent",
        "bone contusion: not present",
        "menisci: unremarkable",
    ],
)
def test_a_list_style_negation_is_caught(clause):
    assert matches(clause)


@pytest.mark.parametrize(
    "clause",
    ["baker kisti: yok", "menisk: nema", "derrame: sin", "erguss: kein"],
)
def test_the_same_layout_in_another_language_is_caught(clause):
    """The vocabulary is the one B6 already carries; only the layout is new."""
    assert matches(clause)


# --- what it must leave alone ------------------------------------------------


@pytest.mark.parametrize(
    "clause",
    [
        "complete tear of the anterior cruciate ligament",
        "trace baker's cyst",
        "tiny bone bruise at the posterior lateral tibial plateau",
        "small joint effusion is present",
        "bone contusion with neglected fracture line at fibular head",
    ],
)
def test_a_genuine_assertion_is_not_caught(clause):
    assert not matches(clause)


def test_a_negation_word_without_the_layout_is_not_caught():
    """Prose negation is B6's job and it already handles it."""
    assert not matches("there is no meniscal tear")


def test_a_colon_without_a_negation_is_not_caught():
    assert not matches("findings: complete acl rupture")


def test_an_empty_clause_is_not_caught():
    assert not matches("")


# --- counting ----------------------------------------------------------------


def test_a_positive_call_on_a_negating_clause_is_counted(tmp_path):
    path = _write(
        tmp_path,
        _export(["a"], {"Baker's": ["positive"]}, evidence={"Baker's": ["baker cyst: none"]}),
    )
    result = scan(b6_export=path)

    assert result["list_negated_calls"] == 1
    assert result["by_target"]["Baker's"]["list_negated"] == 1
    assert result["studies_affected"] == 1


def test_a_negated_call_on_the_same_clause_is_not_counted(tmp_path):
    """The parser got it right there; only its positives are in question."""
    path = _write(
        tmp_path,
        _export(["a"], {"Baker's": ["negated"]}, evidence={"Baker's": ["baker cyst: none"]}),
    )
    assert scan(b6_export=path)["list_negated_calls"] == 0


def test_a_positive_call_on_an_asserting_clause_is_not_counted(tmp_path):
    path = _write(
        tmp_path,
        _export(["a"], {"ACL": ["positive"]}, evidence={"ACL": ["complete acl tear"]}),
    )
    assert scan(b6_export=path)["list_negated_calls"] == 0


def test_the_share_is_taken_against_positive_calls_only(tmp_path):
    path = _write(
        tmp_path,
        _export(
            ["a", "b", "c", "d"],
            {"ACL": ["positive", "positive", "negated", "unmentioned"]},
            evidence={"ACL": ["acl: none", "acl tear", "acl: none", ""]},
        ),
    )
    result = scan(b6_export=path)

    assert result["by_target"]["ACL"]["positive_calls"] == 2
    assert result["by_target"]["ACL"]["list_negated"] == 1
    assert result["by_target"]["ACL"]["share"] == 0.5


def test_gold_studies_are_counted_separately_not_excluded(tmp_path):
    """The one observed instance is in a gold study; it must stay visible."""
    path = _write(
        tmp_path,
        _export(
            ["a", "g"],
            {"Baker's": ["positive", "positive"]},
            evidence={"Baker's": ["baker cyst: none", "baker cyst: none"]},
            gold=[False, True],
        ),
    )
    result = scan(b6_export=path)

    assert result["list_negated_calls"] == 2
    assert result["in_gold_studies"] == 1


def test_an_export_with_no_gold_column_reports_none_in_gold(tmp_path):
    path = _write(
        tmp_path,
        _export(["a"], {"ACL": ["positive"]}, evidence={"ACL": ["acl: none"]}),
    )
    assert scan(b6_export=path)["in_gold_studies"] == 0


# --- output ------------------------------------------------------------------


def test_the_matches_are_written_for_reading(tmp_path):
    path = _write(
        tmp_path,
        _export(
            ["a"],
            {"Baker's": ["positive"], "ACL": ["positive"]},
            evidence={"Baker's": ["baker cyst: none"], "ACL": ["acl tear"]},
        ),
    )
    out = tmp_path / "scan"

    scan(b6_export=path, out_root=out)
    found = pd.read_csv(out / "list_negated_positives.csv")

    assert found["target"].tolist() == ["Baker's"]
    assert found["evidence"].iloc[0] == "baker cyst: none"
    assert json.loads((out / "summary.json").read_text())["list_negated_calls"] == 1


def test_a_merged_export_is_refused_because_it_drops_the_evidence(tmp_path):
    frame = _export(["a"], {"ACL": ["positive"]}).drop(
        columns=[f"{target}__evidence" for target in TARGETS]
    )
    path = _write(tmp_path, frame)

    with pytest.raises(ValueError, match="__evidence"):
        scan(b6_export=path)


def test_a_missing_export_names_the_file_it_wants(tmp_path):
    with pytest.raises(FileNotFoundError, match="__evidence"):
        scan(b6_export=tmp_path / "nowhere")
