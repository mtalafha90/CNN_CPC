import hashlib
import json

import pandas as pd

from rsna_knee.constants import TARGETS
from rsna_knee import translation_rescue_supervision_merge as merge


def _blank_row(uid: str):
    row = {"StudyInstanceUID": uid}
    for target in TARGETS:
        row[target] = 0.5
        row[f"{target}__confidence"] = 0.0
        row[f"{target}__state"] = "unmentioned"
    return row


def test_phase8_global_merge_preserves_active_rows_and_only_adds_frozen_cells(tmp_path, monkeypatch):
    active = _blank_row("active")
    active["ACL"] = 0.97
    active["ACL__confidence"] = 0.9
    active["ACL__state"] = "positive"
    rescued = _blank_row("rescued")
    silent = _blank_row("silent")
    b6 = pd.DataFrame([active, rescued, silent])

    monkeypatch.setattr(
        merge,
        "load_frozen_b6_export",
        lambda root: (b6.copy(), {"version": "1.2.1"}, {"b6_version": "1.2.1"}),
    )
    monkeypatch.setattr(merge, "EXPECTED_REPORT_ONLY", 3)
    monkeypatch.setattr(merge, "EXPECTED_ORIGINAL_ACTIVE", 1)
    monkeypatch.setattr(merge, "EXPECTED_ORIGINAL_INACTIVE", 2)
    monkeypatch.setattr(merge, "EXPECTED_ORIGINAL_USABLE", 1)
    monkeypatch.setattr(merge, "EXPECTED_RECOVERED_STUDIES", 1)
    monkeypatch.setattr(merge, "EXPECTED_RECOVERED_CELLS", 2)
    monkeypatch.setattr(merge, "EXPECTED_RECOVERED_POSITIVE", 1)
    monkeypatch.setattr(merge, "EXPECTED_RECOVERED_NEGATIVE", 1)

    phase7 = tmp_path / "phase7"
    phase7.mkdir()
    summary = {
        "version": merge.REQUIRED_PHASE7_VERSION,
        "translator_matches_phase6_exactly": True,
    }
    (phase7 / "full_population_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    recovered = pd.DataFrame(
        [
            {
                "StudyInstanceUID": "rescued",
                "report_script_bucket": "Greek",
                "target": "MCL",
                "state": "negated",
                "confidence": 0.9,
                "probability": 0.03,
            },
            {
                "StudyInstanceUID": "rescued",
                "report_script_bucket": "Greek",
                "target": "Effusion",
                "state": "positive",
                "confidence": 0.9,
                "probability": 0.97,
            },
        ]
    )
    recovered_path = phase7 / "recovered_cells.csv"
    recovered.to_csv(recovered_path, index=False)
    sha = hashlib.sha256(recovered_path.read_bytes()).hexdigest()
    monkeypatch.setattr(merge, "REQUIRED_RECOVERED_CELLS_SHA256", sha)

    out = tmp_path / "merged"
    audit = merge.build_merged_supervision(
        b6_root=tmp_path / "b6",
        phase7_root=phase7,
        out_root=out,
    )

    result = pd.read_csv(out / "training_targets.csv")
    result["StudyInstanceUID"] = result["StudyInstanceUID"].astype(str)
    active_after = result.set_index("StudyInstanceUID").loc["active"]
    assert active_after["ACL__state"] == "positive"
    assert float(active_after["ACL__confidence"]) == 0.9
    assert result.set_index("StudyInstanceUID").loc["rescued", "MCL__state"] == "negated"
    assert result.set_index("StudyInstanceUID").loc["rescued", "Effusion__state"] == "positive"
    assert result.set_index("StudyInstanceUID").loc["silent", "ACL__state"] == "unmentioned"

    assert audit["original"]["active_studies"] == 1
    assert audit["rescue"]["studies"] == 1
    assert audit["candidate"]["active_studies"] == 2
    assert audit["candidate"]["usable_cells"] == 3
    assert audit["guardrails"]["all_original_b6_active_rows_preserved_exactly"] is True
    assert audit["guardrails"]["gold_in_training"] is False
