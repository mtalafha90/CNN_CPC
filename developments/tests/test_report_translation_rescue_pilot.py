from __future__ import annotations

import json

from rsna_knee.constants import TARGETS
from rsna_knee.report_translation_rescue_pilot import (
    EXPECTED_SAMPLE_VERSION,
    INACTIVE_STRATA,
    _candidate_merge,
    _parse_translation,
    b6_snapshot,
    run_translation_rescue_pilot,
)


def _record(uid: str, stratum: str, script: str, report: str, *, gold: bool = False) -> dict:
    row = {
        "audit_version": EXPECTED_SAMPLE_VERSION,
        "sample_stratum": stratum,
        "StudyInstanceUID": uid,
        "report_script_bucket": script,
        "report_text": report,
        "repository_gold": gold,
    }
    if gold:
        row["official_labels"] = {target: 0 for target in TARGETS}
    return row


def _write_frozen_sample(path):
    rows = []
    scripts = {
        "latin_b6_inactive": ("Latin", "neprepoznat tekst"),
        "greek_b6_inactive": ("Greek", "μη αναγνωρίσιμο κείμενο"),
        "cyrillic_b6_inactive": ("Cyrillic", "неразпознат текст"),
    }
    n = 0
    for stratum in INACTIVE_STRATA:
        script, report = scripts[stratum]
        for _ in range(12):
            n += 1
            rows.append(_record(f"inactive-{n}", stratum, script, report))

    rows.append(
        _record(
            "active-latin",
            "latin_b6_active_control",
            "Latin",
            "ACL: complete tear. No fracture.",
        )
    )
    rows.append(
        _record(
            "active-greek",
            "greek_b6_active_control",
            "Greek",
            "bone bruise.",
        )
    )
    rows.append(
        _record(
            "active-cyrillic",
            "cyrillic_b6_active_all",
            "Cyrillic",
            "subchondral insufficiency fracture.",
        )
    )
    rows.append(
        _record(
            "gold-1",
            "gold_latin_control",
            "Latin",
            "unseen gold wording",
            gold=True,
        )
    )
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_parse_translation_requires_json_and_nonempty():
    assert _parse_translation('{"translation":"ACL tear."}') == "ACL tear."

    try:
        _parse_translation('{"translation":""}')
    except ValueError:
        pass
    else:
        raise AssertionError("empty translation should fail")


def test_candidate_merge_never_overrides_original_usable_cells():
    original = b6_snapshot("ACL: complete tear. No fracture.")
    translated = b6_snapshot("ACL is intact. Fracture is present.")

    untouched = _candidate_merge(original, translated, eligible=False)
    assert untouched == original

    merged = _candidate_merge(original, translated, eligible=True)
    assert merged["targets"]["ACL"] == original["targets"]["ACL"]
    assert merged["targets"]["Fracture"] == original["targets"]["Fracture"]


def test_translation_rescue_pilot_passes_when_all_inactive_strata_are_recovered(tmp_path):
    sample = tmp_path / "sample.jsonl"
    _write_frozen_sample(sample)

    def translate(_report: str) -> str:
        return "ACL: complete tear. No fracture."

    out = tmp_path / "out"
    summary = run_translation_rescue_pilot(
        sample_jsonl=sample,
        out_root=out,
        translate=translate,
        provenance={"backend": "unit-test", "reproducible": True},
    )

    assert summary["inactive_sample_studies"] == 36
    assert summary["inactive_rescued_to_active"] == 36
    assert summary["inactive_overall_rescue_rate"] == 1.0
    assert summary["active_control_original_b6_cells_preserved"] is True
    assert summary["feasibility_passed"] is True
    for stratum in INACTIVE_STRATA:
        item = summary["strata"][stratum]
        assert item["rescue_rate"] == 1.0
        assert item["added_positive_cells"] > 0
        assert item["added_negative_cells"] > 0
        assert item["both_positive_and_negative_cells_recovered"] is True

    assert (out / "pilot_summary.json").is_file()
    assert (out / "pilot_cell_audit.csv").is_file()
    assert (out / "translation_results.jsonl").is_file()


def test_translation_failure_fails_feasibility(tmp_path):
    sample = tmp_path / "sample.jsonl"
    _write_frozen_sample(sample)
    calls = {"n": 0}

    def translate(_report: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("synthetic translation failure")
        return "ACL: complete tear. No fracture."

    summary = run_translation_rescue_pilot(
        sample_jsonl=sample,
        out_root=tmp_path / "out",
        translate=translate,
        provenance={"backend": "unit-test", "reproducible": True},
    )
    assert summary["translation_failures"] == 1
    assert summary["feasibility_passed"] is False


def test_translation_rescue_rejects_wrong_sample_version(tmp_path):
    sample = tmp_path / "sample.jsonl"
    sample.write_text(
        json.dumps(
            {
                "audit_version": "wrong",
                "sample_stratum": "latin_b6_inactive",
                "StudyInstanceUID": "x",
                "report_script_bucket": "Latin",
                "report_text": "text",
                "repository_gold": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        run_translation_rescue_pilot(
            sample_jsonl=sample,
            out_root=tmp_path / "out",
            translate=lambda x: x,
        )
    except ValueError as exc:
        assert "unexpected Phase-5 sample version" in str(exc)
    else:
        raise AssertionError("wrong Phase-5 version should fail")
