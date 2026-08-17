import numpy as np
import pandas as pd

from rsna_knee.constants import TARGETS
from rsna_knee import phase9_supervision as p9


def _blank_supervision_row(uid: str):
    row = {"StudyInstanceUID": uid}
    for target in TARGETS:
        row[target] = 0.5
        row[f"{target}__confidence"] = 0.0
        row[f"{target}__state"] = "unmentioned"
    return row


def _tiny_train():
    rows = []
    for uid in ("active", "rescued", "silent"):
        row = {"StudyInstanceUID": uid}
        row.update({target: np.nan for target in TARGETS})
        rows.append(row)
    gold = {"StudyInstanceUID": "gold"}
    gold.update({target: 0.0 for target in TARGETS})
    rows.append(gold)
    return pd.DataFrame(rows)


def test_phase9_all_report_only_supervision_retains_zero_weight_studies(monkeypatch):
    monkeypatch.setattr(p9, "REPORT_ONLY_STUDIES", 3)
    active = _blank_supervision_row("active")
    active["ACL"] = 0.97
    active["ACL__confidence"] = 0.9
    active["ACL__state"] = "positive"
    rescued = _blank_supervision_row("rescued")
    silent = _blank_supervision_row("silent")

    uids, targets, weights, summary = p9.prepare_all_report_only_supervision(
        _tiny_train(), pd.DataFrame([active, rescued, silent])
    )

    assert uids == ["active", "rescued", "silent"]
    assert targets.shape == (3, len(TARGETS))
    assert weights.shape == (3, len(TARGETS))
    assert int((weights > 0).sum()) == 1
    assert int((weights.sum(axis=1) == 0).sum()) == 2
    assert summary["active_studies"] == 1
    assert summary["inactive_studies_zero_usable_cells"] == 2
    assert summary["zero_weight_studies_retained_in_mri_exposure"] is True


def test_phase9_candidate_changes_supervision_not_uid_exposure(monkeypatch):
    monkeypatch.setattr(p9, "REPORT_ONLY_STUDIES", 3)
    monkeypatch.setattr(p9, "CONTROL_ACTIVE_STUDIES", 1)
    monkeypatch.setattr(p9, "CONTROL_USABLE_CELLS", 1)
    monkeypatch.setattr(p9, "CONTROL_POSITIVE_CELLS", 1)
    monkeypatch.setattr(p9, "CONTROL_NEGATIVE_CELLS", 0)
    monkeypatch.setattr(p9, "CANDIDATE_ACTIVE_STUDIES", 2)
    monkeypatch.setattr(p9, "CANDIDATE_USABLE_CELLS", 2)
    monkeypatch.setattr(p9, "CANDIDATE_POSITIVE_CELLS", 1)
    monkeypatch.setattr(p9, "CANDIDATE_NEGATIVE_CELLS", 1)

    active = _blank_supervision_row("active")
    active["ACL"] = 0.97
    active["ACL__confidence"] = 0.9
    active["ACL__state"] = "positive"
    rescued_control = _blank_supervision_row("rescued")
    silent_control = _blank_supervision_row("silent")
    control = pd.DataFrame([active, rescued_control, silent_control])

    rescued_candidate = _blank_supervision_row("rescued")
    rescued_candidate["MCL"] = 0.03
    rescued_candidate["MCL__confidence"] = 0.9
    rescued_candidate["MCL__state"] = "negated"
    candidate = pd.DataFrame([active.copy(), rescued_candidate, _blank_supervision_row("silent")])

    monkeypatch.setattr(
        p9,
        "load_frozen_b6_export",
        lambda root: (control.copy(), {"version": "1.2.1"}, {"b6_version": "1.2.1"}),
    )
    monkeypatch.setattr(
        p9,
        "load_frozen_phase8_export",
        lambda root: (candidate.copy(), {"version": p9.PHASE8_VERSION}, {"version": p9.PHASE8_VERSION}),
    )

    train = _tiny_train()
    cuids, _, cw, csummary, _ = p9.load_phase9_arm_supervision(
        train, arm="control", b6_root="b6", phase8_root="phase8"
    )
    auids, _, aw, asummary, _ = p9.load_phase9_arm_supervision(
        train, arm="candidate", b6_root="b6", phase8_root="phase8"
    )

    assert cuids == auids == ["active", "rescued", "silent"]
    assert csummary["usable_cells"] == 1
    assert asummary["usable_cells"] == 2
    assert int((cw.sum(axis=1) == 0).sum()) == 2
    assert int((aw.sum(axis=1) == 0).sum()) == 1
