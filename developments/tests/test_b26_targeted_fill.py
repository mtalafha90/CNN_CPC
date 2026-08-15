"""B26: the fill must be additive, and its scope must come from the audit."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from rsna_knee.b23_local_llm import (
    BACKEND_OLLAMA,
    DECODING_GREEDY,
    EVIDENCE_MAX_CHARS,
    ModelProvenance,
)
from rsna_knee.b26_targeted_fill import (
    B26_VERSION,
    build_findings_schema,
    build_fill_supervision,
    build_system_prompt,
    parse_targeted_response,
    resolve_fill_targets,
    run_targeted_fill,
    state_to_supervision,
)
from rsna_knee.constants import TARGETS

FLAGGED = ["Synovitis"]


def _provenance(**overrides):
    fields = dict(
        backend=BACKEND_OLLAMA, model_id="qwen3:14b", revision="a" * 64,
        dtype="gguf", quantisation="Q4_K_M", decoding=DECODING_GREEDY,
        max_new_tokens=4096, seed=2026, prompt_sha256="b" * 64,
        openly_downloadable=True,
    )
    fields.update(overrides)
    return ModelProvenance(**fields)


def _response(targets, state="negated", confidence=0.9, evidence="normal synovium"):
    return json.dumps(
        {"findings": {t: {"state": state, "confidence": confidence, "evidence": evidence} for t in targets}}
    )


# --- scope comes from the audit, never from the module -----------------------


def test_the_scope_is_read_from_the_balance_audit(tmp_path):
    path = tmp_path / "balance.json"
    path.write_text(json.dumps({"targets_needing_fill": ["Synovitis"]}), encoding="utf-8")
    assert resolve_fill_targets(path) == ["Synovitis"]


def test_whichever_target_the_audit_flags_is_what_gets_filled(tmp_path):
    # Nothing in B26 may hard-code a target name.
    for flagged in (["ACL"], ["Effusion", "Fracture"]):
        path = tmp_path / "b.json"
        path.write_text(json.dumps({"targets_needing_fill": flagged}), encoding="utf-8")
        assert resolve_fill_targets(path) == flagged
        prompt = build_system_prompt(flagged)
        for target in flagged:
            assert target in prompt
        assert build_findings_schema(flagged)["properties"]["findings"]["required"] == flagged


def test_an_audit_flagging_nothing_means_there_is_no_experiment(tmp_path):
    path = tmp_path / "balance.json"
    path.write_text(json.dumps({"targets_needing_fill": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="nothing for B26 to fill"):
        resolve_fill_targets(path)


def test_a_non_audit_payload_is_refused(tmp_path):
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"something": "else"}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a balance-audit payload"):
        resolve_fill_targets(path)


# --- the targeted prompt -----------------------------------------------------


def test_the_prompt_covers_only_the_flagged_targets():
    prompt = build_system_prompt(FLAGGED)
    named = [t for t in TARGETS if t in prompt]
    assert named == FLAGGED


def test_the_targeted_prompt_is_much_smaller_than_the_twelve_target_one():
    from rsna_knee.b23_llm_labels import SYSTEM_PROMPT as FULL

    # This is what removes the output-budget truncation risk.
    assert len(build_system_prompt(FLAGGED)) < len(FULL) * 0.75


def test_the_prompt_keeps_the_frozen_semantics():
    prompt = build_system_prompt(FLAGGED).lower()
    assert "never infer absence from silence" in prompt
    assert "any abnormality" in prompt
    assert "the impression" in prompt
    # Rule 1: the clinical request is not a finding.
    assert "indication" in prompt and "never evidence of a finding" in prompt


def test_the_prompt_does_not_ask_the_labeller_to_find_negatives():
    # Synovitis needs negatives, but soliciting them would manufacture the
    # class balance rather than read the report.
    prompt = build_system_prompt(FLAGGED).lower()
    for phrase in ("find negative", "prefer negated", "look for absent", "bias toward"):
        assert phrase not in prompt


def test_an_unknown_target_is_refused():
    with pytest.raises(ValueError, match="unknown targets"):
        build_system_prompt(["Not A Target"])
    with pytest.raises(ValueError, match="at least one target"):
        build_system_prompt([])


# --- parsing -----------------------------------------------------------------


def test_a_valid_response_parses():
    parsed = parse_targeted_response(_response(FLAGGED), FLAGGED)
    assert set(parsed) == set(FLAGGED)
    assert parsed["Synovitis"]["state"] == "negated"


def test_a_reasoning_block_never_reaches_the_parser():
    raw = "<think>the report calls the synovium normal</think>" + _response(FLAGGED)
    assert parse_targeted_response(raw, FLAGGED)["Synovitis"]["state"] == "negated"


@pytest.mark.parametrize(
    "raw",
    ["not json", json.dumps({"findings": {}}), json.dumps({"nope": {}})],
)
def test_malformed_responses_are_rejected_rather_than_defaulted(raw):
    with pytest.raises(ValueError):
        parse_targeted_response(raw, FLAGGED)


def test_an_over_long_evidence_span_is_capped():
    raw = _response(FLAGGED, evidence="x" * 500)
    parsed = parse_targeted_response(raw, FLAGGED)
    assert len(parsed["Synovitis"]["evidence"]) <= EVIDENCE_MAX_CHARS + 3


def test_definite_states_take_the_b6_matched_fixed_confidence():
    # B26 changes which cells exist, not how supervision is thresholded.
    _p, conf = state_to_supervision("positive", 0.41)
    assert conf == pytest.approx(0.90)
    _p, conf = state_to_supervision("uncertain", 0.99)
    assert conf == 0.0


# --- the fill is additive ----------------------------------------------------


def _base(n=6):
    y = np.full((n, len(TARGETS)), 0.5)
    w = np.zeros((n, len(TARGETS)))
    j = TARGETS.index("Synovitis")
    y[0, j], w[0, j] = 0.85, 0.50   # an existing positive
    y[1, j], w[1, j] = 0.05, 1.00   # an existing negative
    return y, w


def _fill_frame(states, n=6):
    data = {}
    for target in TARGETS:
        data[f"{target}__state"] = ["unmentioned"] * n
        data[f"{target}__confidence"] = [0.0] * n
    data["Synovitis__state"] = states
    data["Synovitis__confidence"] = [0.9 if s in ("positive", "negated") else 0.0 for s in states]
    return pd.DataFrame(data)


def test_fill_adds_only_where_the_base_is_silent():
    y, w = _base()
    frame = _fill_frame(["negated", "positive", "negated", "positive", "unmentioned", "uncertain"])
    result = build_fill_supervision(y, w, frame, FLAGGED)

    # Rows 0 and 1 already had supervision; rows 2 and 3 are new.
    assert result["cells_added"]["Synovitis"] == 2
    assert result["cells_skipped_already_supervised"]["Synovitis"] == 2
    assert result["final_usable_cells"] == result["base_usable_cells"] + 2


def test_no_existing_cell_is_dropped_or_overridden():
    y, w = _base()
    # The fill disagrees with the base on both already-supervised rows.
    frame = _fill_frame(["negated", "positive", "unmentioned", "unmentioned", "unmentioned", "unmentioned"])
    result = build_fill_supervision(y, w, frame, FLAGGED)

    j = TARGETS.index("Synovitis")
    assert result["targets"][0, j] == 0.85 and result["weights"][0, j] == 0.50
    assert result["targets"][1, j] == 0.05 and result["weights"][1, j] == 1.00
    assert result["base_cells_dropped"] == 0
    assert result["base_cells_overridden"] == 0


def test_targets_the_audit_did_not_flag_are_left_completely_alone():
    y, w = _base()
    j = TARGETS.index("ACL")
    y[3, j], w[3, j] = 0.85, 0.50
    frame = _fill_frame(["unmentioned"] * 6)
    # Even though the frame carries ACL columns, ACL is not in the fill list.
    frame["ACL__state"] = ["negated"] * 6
    frame["ACL__confidence"] = [0.9] * 6

    result = build_fill_supervision(y, w, frame, FLAGGED)
    np.testing.assert_array_equal(result["weights"][:, j], w[:, j])
    np.testing.assert_array_equal(result["targets"][:, j], y[:, j])


def test_hedged_and_silent_fill_states_never_become_supervision():
    y, w = _base()
    frame = _fill_frame(["unmentioned"] * 2 + ["uncertain", "unmentioned", "uncertain", "unmentioned"])
    result = build_fill_supervision(y, w, frame, FLAGGED)
    assert result["cells_added"]["Synovitis"] == 0
    assert result["final_usable_cells"] == result["base_usable_cells"]


def test_misaligned_fill_states_are_refused():
    y, w = _base(6)
    with pytest.raises(ValueError, match="align row-for-row"):
        build_fill_supervision(y, w, _fill_frame(["unmentioned"] * 3, n=3), FLAGGED)


# --- the run -----------------------------------------------------------------


def _train_csv(tmp_path, n=5):
    rows = []
    for i in range(n):
        row = {"StudyInstanceUID": f"uid-{i}", "Report": f"knee report {i}"}
        for t in TARGETS:
            row[t] = np.nan
        rows.append(row)
    gold = {"StudyInstanceUID": "uid-gold", "Report": "gold report"}
    for j, t in enumerate(TARGETS):
        gold[t] = float(j % 2)
    rows.append(gold)
    path = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_a_run_records_provenance_and_marks_gold(tmp_path):
    def backend(system, user):
        return _response(FLAGGED)

    audit = run_targeted_fill(
        _train_csv(tmp_path), backend, FLAGGED, _provenance(),
        out_root=tmp_path / "b26", progress_every=0,
    )
    assert audit["b26_version"] == B26_VERSION
    assert audit["targets"] == FLAGGED
    assert audit["n_gold_excluded_from_fill"] == 1
    assert audit["n_report_only"] == 5
    assert audit["external_model_reproducible"] is True
    assert audit["per_target_states"]["Synovitis"]["negated"] == 5


def test_an_unpinnable_run_is_refused(tmp_path):
    with pytest.raises(ValueError, match="reproducible provenance"):
        run_targeted_fill(
            _train_csv(tmp_path), lambda s, u: _response(FLAGGED), FLAGGED,
            _provenance(revision="unknown"), out_root=tmp_path / "b26", progress_every=0,
        )


def test_identical_reports_are_only_sent_once(tmp_path):
    frame = pd.read_csv(_train_csv(tmp_path))
    frame["Report"] = "one identical report"
    path = tmp_path / "same.csv"
    frame.to_csv(path, index=False)

    calls = {"n": 0}

    def counting(system, user):
        calls["n"] += 1
        return _response(FLAGGED)

    run_targeted_fill(path, counting, FLAGGED, _provenance(),
                      out_root=tmp_path / "b26", progress_every=0)
    assert calls["n"] == 1
