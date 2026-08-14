import json

import numpy as np
import pandas as pd
import pytest

from rsna_knee.b23_llm_labels import (
    B23_IGNORED_STATE_CONFIDENCE,
    B23_VERSION,
    ExtractionCache,
    build_user_prompt,
    empty_report_extraction,
    extract_report,
    load_frozen_b23_export,
    parse_extraction_response,
    run_b23_export,
)
from rsna_knee.constants import TARGETS


def _response(overrides=None, default_state="unmentioned", default_confidence=0.0):
    findings = {
        target: {"state": default_state, "confidence": default_confidence, "evidence": ""}
        for target in TARGETS
    }
    for target, cell in (overrides or {}).items():
        findings[target] = cell
    return json.dumps({"findings": findings})


def _stub_backend(response_text):
    def _call(system, user):
        return response_text

    return _call


def test_user_prompt_lists_every_target_with_a_definition():
    prompt = build_user_prompt("MRI rodilla derecha: sin derrame articular.")
    for target in TARGETS:
        assert f"- {target}:" in prompt
    assert "sin derrame articular" in prompt


def test_parse_extraction_response_accepts_all_twelve_targets():
    parsed = parse_extraction_response(
        _response({"ACL": {"state": "positive", "confidence": 0.9, "evidence": "ACL torn"}})
    )
    assert set(parsed) == set(TARGETS)
    assert parsed["ACL"].state == "positive"
    assert parsed["ACL"].evidence == "ACL torn"
    assert parsed["Effusion"].state == "unmentioned"


def test_parse_extraction_response_tolerates_a_fenced_block():
    fenced = "```json\n" + _response() + "\n```"
    assert set(parse_extraction_response(fenced)) == set(TARGETS)


@pytest.mark.parametrize(
    "bad",
    [
        "not json at all",
        json.dumps({"no_findings": {}}),
        json.dumps({"findings": {"ACL": {"state": "positive", "confidence": 0.5}}}),
    ],
)
def test_parse_extraction_response_rejects_malformed_payloads(bad):
    with pytest.raises(ValueError):
        parse_extraction_response(bad)


def test_parse_extraction_response_rejects_unknown_state():
    bad = _response({"ACL": {"state": "probably", "confidence": 0.5, "evidence": ""}})
    with pytest.raises(ValueError, match="unknown state"):
        parse_extraction_response(bad)


@pytest.mark.parametrize("confidence", [-0.1, 1.5, "high", float("nan")])
def test_parse_extraction_response_rejects_bad_confidence(confidence):
    bad = _response({"ACL": {"state": "positive", "confidence": confidence, "evidence": ""}})
    with pytest.raises(ValueError):
        parse_extraction_response(bad)


def test_uncertain_and_unmentioned_can_never_clear_the_usable_threshold():
    parsed = parse_extraction_response(
        _response(
            {
                "ACL": {"state": "uncertain", "confidence": 0.99, "evidence": "posible"},
                "MCL": {"state": "unmentioned", "confidence": 0.99, "evidence": ""},
                "Effusion": {"state": "positive", "confidence": 0.83, "evidence": "derrame"},
            }
        )
    )
    # A confident hedge is still a hedge: it must not become supervision.
    assert parsed["ACL"].usable_confidence() == B23_IGNORED_STATE_CONFIDENCE
    assert parsed["MCL"].usable_confidence() == B23_IGNORED_STATE_CONFIDENCE
    assert parsed["Effusion"].usable_confidence() == pytest.approx(0.83)


def test_negated_and_unmentioned_receive_different_probabilities():
    parsed = parse_extraction_response(
        _response(
            {
                "ACL": {"state": "negated", "confidence": 0.9, "evidence": "intact"},
                "MCL": {"state": "unmentioned", "confidence": 0.0, "evidence": ""},
            }
        )
    )
    assert parsed["ACL"].probability() < 0.5
    assert parsed["MCL"].probability() == pytest.approx(0.5)


def test_empty_report_is_all_unmentioned_without_calling_the_backend():
    def _explode(system, user):
        raise AssertionError("backend must not be called for an empty report")

    extraction, meta = extract_report("   ", _explode)
    assert meta["empty_report"] is True
    assert all(item.state == "unmentioned" for item in extraction.values())
    assert extraction == empty_report_extraction()


def test_extract_report_retries_then_succeeds():
    attempts = {"n": 0}

    def _flaky(system, user):
        attempts["n"] += 1
        return "garbage" if attempts["n"] == 1 else _response()

    extraction, meta = extract_report("real report text", _flaky, sleep=lambda _s: None)
    assert meta["attempts"] == 2
    assert len(extraction) == len(TARGETS)


def test_extract_report_raises_after_exhausting_attempts():
    with pytest.raises(RuntimeError, match="failed after retries"):
        extract_report(
            "real report text",
            _stub_backend("still garbage"),
            max_attempts=2,
            sleep=lambda _s: None,
        )


def test_extraction_cache_round_trips_and_resumes(tmp_path):
    path = tmp_path / "cache.jsonl"
    cache = ExtractionCache(path)
    assert len(cache) == 0
    cache.put("abc", {"report_sha1": "abc", "findings": {}})
    reopened = ExtractionCache(path)
    assert len(reopened) == 1
    assert reopened.get("abc")["report_sha1"] == "abc"
    assert reopened.get("missing") is None


def _train_frame():
    rows = []
    for i in range(4):
        row = {
            "StudyInstanceUID": f"uid-{i}",
            "Report": f"knee MRI report number {i}",
        }
        for target in TARGETS:
            row[target] = np.nan
        rows.append(row)
    # One gold study with complete expert labels.
    gold = {"StudyInstanceUID": "uid-gold", "Report": "gold knee MRI report"}
    for j, target in enumerate(TARGETS):
        gold[target] = float(j % 2)
    rows.append(gold)
    return pd.DataFrame(rows)


def test_run_b23_export_excludes_gold_and_round_trips(tmp_path):
    train_csv = tmp_path / "train.csv"
    _train_frame().to_csv(train_csv, index=False)
    backend = _stub_backend(
        _response({"Effusion": {"state": "positive", "confidence": 0.91, "evidence": "effusion"}})
    )

    audit = run_b23_export(
        train_csv, backend, out_root=tmp_path / "export", progress_every=0
    )
    assert audit["b23_version"] == B23_VERSION
    assert audit["gold_rows_in_training_targets"] == 0
    assert audit["n_gold_audit_only"] == 1
    assert audit["n_report_only_training"] == 4

    frame, policy, loaded_audit = load_frozen_b23_export(tmp_path / "export")
    assert len(frame) == 4
    assert "uid-gold" not in set(frame["StudyInstanceUID"])
    assert policy["unmentioned_is_negative"] is False
    assert loaded_audit["usable_cells_total"] == 4  # one usable Effusion cell per report study


def test_run_b23_export_reuses_cache_for_identical_reports(tmp_path):
    frame = _train_frame()
    frame["Report"] = "identical report for every study"
    train_csv = tmp_path / "train.csv"
    frame.to_csv(train_csv, index=False)

    calls = {"n": 0}

    def _counting(system, user):
        calls["n"] += 1
        return _response()

    run_b23_export(train_csv, _counting, out_root=tmp_path / "export", progress_every=0)
    # Five studies share one report hash, so the backend is hit exactly once.
    assert calls["n"] == 1


def test_load_frozen_b23_export_rejects_unmentioned_as_negative(tmp_path):
    train_csv = tmp_path / "train.csv"
    _train_frame().to_csv(train_csv, index=False)
    out = tmp_path / "export"
    run_b23_export(train_csv, _stub_backend(_response()), out_root=out, progress_every=0)

    policy = json.loads((out / "policy.json").read_text(encoding="utf-8"))
    policy["unmentioned_is_negative"] = True
    (out / "policy.json").write_text(json.dumps(policy), encoding="utf-8")

    with pytest.raises(ValueError, match="must not map unmentioned"):
        load_frozen_b23_export(out)
