import numpy as np
import pandas as pd

from rsna_knee.b6_b15_gold_diagnostic import (
    STATE_SOFT_SCORES,
    _alignment_cells,
    _macro_selective_auc,
    _summarize_alignment,
)
from rsna_knee.constants import TARGETS
from rsna_knee.report_labels import (
    STATE_NEGATED,
    STATE_POSITIVE,
    STATE_UNCERTAIN,
    STATE_UNMENTIONED,
)


def test_state_soft_scores_match_frozen_diagnostic_convention():
    assert STATE_SOFT_SCORES == {
        STATE_POSITIVE: 0.85,
        STATE_NEGATED: 0.05,
        STATE_UNCERTAIN: 0.50,
        STATE_UNMENTIONED: 0.50,
    }


def test_selective_auc_uses_only_eligible_cells():
    truth = np.array(
        [
            [0, 0],
            [1, 1],
            [1, 0],
            [0, 1],
        ],
        dtype=float,
    )
    score = np.array(
        [
            [0.05, 0.10],
            [0.85, 0.90],
            [0.01, 0.20],
            [0.99, 0.80],
        ],
        dtype=float,
    )
    eligible = np.array(
        [
            [True, True],
            [True, True],
            [False, True],
            [False, True],
        ]
    )

    macro, per_target = _macro_selective_auc(truth, score, eligible)
    assert np.allclose(per_target, [1.0, 1.0])
    assert macro == 1.0


def _alignment_fixture():
    rows = []
    for uid, acl_truth, acl_state in [
        ("A", 0, STATE_POSITIVE),
        ("B", 1, STATE_NEGATED),
    ]:
        row = {"StudyInstanceUID": uid}
        for target in TARGETS:
            row[target] = 0
            row[f"{target}__state"] = STATE_UNCERTAIN
            row[f"{target}__confidence"] = 0.10
        row["ACL"] = acl_truth
        row["ACL__state"] = acl_state
        row["ACL__confidence"] = 0.90
        rows.append(row)
    merged = pd.DataFrame(rows)

    b13 = pd.DataFrame({"StudyInstanceUID": ["A", "B"]})
    b15 = pd.DataFrame({"StudyInstanceUID": ["A", "B"]})
    for target in TARGETS:
        b13[target] = 0.5
        b15[target] = 0.5
    b13.loc[b13["StudyInstanceUID"].eq("A"), "ACL"] = 0.40
    b15.loc[b15["StudyInstanceUID"].eq("A"), "ACL"] = 0.80
    b13.loc[b13["StudyInstanceUID"].eq("B"), "ACL"] = 0.60
    b15.loc[b15["StudyInstanceUID"].eq("B"), "ACL"] = 0.20
    return merged, b13, b15


def test_teacher_wrong_cells_can_move_toward_b6_and_away_from_truth():
    merged, b13, b15 = _alignment_fixture()
    cells = _alignment_cells(merged, b13, b15, min_confidence=0.75)

    assert len(cells) == 2
    assert not cells["teacher_correct"].any()
    assert np.allclose(cells["movement_toward_teacher"], [0.4, 0.4])
    assert np.allclose(cells["movement_toward_truth"], [-0.4, -0.4])
    assert np.allclose(cells["change_abs_distance_to_truth"], [0.4, 0.4])

    summary = _summarize_alignment(cells)
    assert np.isclose(summary["mean_movement_toward_teacher"], 0.4)
    assert np.isclose(summary["mean_movement_toward_truth"], -0.4)
    assert np.isclose(summary["fraction_move_toward_teacher"], 1.0)
    assert np.isclose(summary["mean_change_abs_distance_to_truth"], 0.4)
