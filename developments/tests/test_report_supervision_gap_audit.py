from __future__ import annotations

import pandas as pd

from rsna_knee.constants import TARGETS
from rsna_knee.report_supervision_gap_audit import (
    REPORT_GAP_SALT,
    _deterministic_take,
    build_gap_sample,
)


def _train_row(uid: str, report: str, gold: bool = False) -> dict:
    row = {"StudyInstanceUID": uid, "Report": report}
    for i, target in enumerate(TARGETS):
        row[target] = int(i % 2) if gold else None
    return row


def _b6_row(uid: str, active: bool) -> dict:
    row = {"StudyInstanceUID": uid}
    for i, target in enumerate(TARGETS):
        if active and i == 0:
            row[f"{target}__state"] = "negated"
            row[f"{target}__confidence"] = 0.95
        else:
            row[f"{target}__state"] = "unmentioned"
            row[f"{target}__confidence"] = 0.0
    return row


def test_deterministic_take_is_stable() -> None:
    frame = pd.DataFrame({"StudyInstanceUID": [f"u{i}" for i in range(20)]})
    a = _deterministic_take(frame, 7)["StudyInstanceUID"].tolist()
    b = _deterministic_take(frame.sample(frac=1, random_state=3), 7)["StudyInstanceUID"].tolist()
    assert a == b
    assert len(a) == len(set(a)) == 7
    assert REPORT_GAP_SALT


def test_gap_sample_keeps_nonlatin_gold_and_strata_without_overlap() -> None:
    train_rows = [
        _train_row("g_gr", "Παρατηρείται ρήξη μηνίσκου.", gold=True),
        _train_row("g_cy", "Признаки повреждения мениска.", gold=True),
        _train_row("g_la", "Meniscal tear is present.", gold=True),
        _train_row("la_i", "No acute osseous abnormality."),
        _train_row("la_a", "ACL tear. No fracture."),
        _train_row("gr_i", "Χωρίς κάταγμα."),
        _train_row("gr_a", "Ρήξη πρόσθιου χιαστού."),
        _train_row("cy_i", "Без перелома."),
        _train_row("cy_a", "Разрыв передней крестообразной связки."),
    ]
    train = pd.DataFrame(train_rows)
    b6 = pd.DataFrame([
        _b6_row("la_i", False),
        _b6_row("la_a", True),
        _b6_row("gr_i", False),
        _b6_row("gr_a", True),
        _b6_row("cy_i", False),
        _b6_row("cy_a", True),
    ])
    manifest, records, summary = build_gap_sample(train, b6, per_stratum=1)
    assert manifest["StudyInstanceUID"].is_unique
    assert set(manifest.loc[manifest["sample_stratum"] == "gold_nonlatin_all", "StudyInstanceUID"]) == {
        "g_gr", "g_cy"
    }
    assert summary["selected_studies"] == len(manifest) == len(records)
    assert summary["raw_text_export_is_local_only"] is True
    for record in records:
        if record["repository_gold"]:
            assert "official_labels" in record
            assert "b6" not in record
        else:
            assert "b6" in record
