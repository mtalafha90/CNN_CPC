from __future__ import annotations

import json

import numpy as np
import pandas as pd

from rsna_knee.chatgpt_hybrid_supervision import (
    build_hybrid_export,
    load_hybrid_export,
)
from rsna_knee.constants import TARGETS
from rsna_knee.data import report_hash
from rsna_knee.b7_weak_supervision import prepare_b7_supervision


def _findings(default_state="unmentioned", default_confidence=0.99):
    return {
        target: {
            "state": default_state,
            "confidence": default_confidence,
            "evidence": "",
        }
        for target in TARGETS
    }


def test_hybrid_export_never_uses_raw_confidence_to_promote_silence(tmp_path):
    train = pd.DataFrame(
        {
            "StudyInstanceUID": ["non_gold_hit", "non_gold_miss", "gold_hit"],
            "Report": ["ACL tear. Possible effusion.", "Nothing cached", "Gold report"],
            **{target: [np.nan, np.nan, 0.0] for target in TARGETS},
        }
    )
    train_csv = tmp_path / "train.csv"
    train.to_csv(train_csv, index=False)

    first = _findings()
    first["ACL"] = {
        "state": "positive",
        "confidence": 0.40,
        "evidence": "ACL tear",
    }
    first["Effusion"] = {
        "state": "uncertain",
        "confidence": 0.80,
        "evidence": "Possible effusion",
    }
    gold = _findings(default_state="positive", default_confidence=1.0)

    cache_path = tmp_path / "hybrid.jsonl"
    cache_rows = [
        {
            "cache_key": "k1",
            "report_sha1": report_hash("ACL tear. Possible effusion."),
            "findings": first,
        },
        {
            "cache_key": "k2",
            "report_sha1": report_hash("Gold report"),
            "findings": gold,
        },
    ]
    cache_path.write_text(
        "\n".join(json.dumps(row) for row in cache_rows) + "\n",
        encoding="utf-8",
    )

    out = tmp_path / "export"
    audit = build_hybrid_export(train_csv, cache_path, out_root=out)

    structured = pd.read_csv(out / "structured_labels.csv")
    targets = pd.read_csv(out / "training_targets.csv")

    assert len(structured) == 3
    assert len(targets) == 2
    assert "gold_hit" not in set(targets["StudyInstanceUID"].astype(str))
    assert audit["gold_rows_in_training_targets"] == 0

    hit = structured.loc[structured["StudyInstanceUID"] == "non_gold_hit"].iloc[0]
    assert hit["ACL__state"] == "positive"
    assert np.isclose(hit["ACL__model_confidence"], 0.40)
    assert np.isclose(hit["ACL__confidence"], 0.90)

    # High raw confidence on silence must stay unusable.
    assert hit["MCL__state"] == "unmentioned"
    assert np.isclose(hit["MCL__model_confidence"], 0.99)
    assert np.isclose(hit["MCL__confidence"], 0.0)

    # Hedging must also stay unusable.
    assert hit["Effusion__state"] == "uncertain"
    assert np.isclose(hit["Effusion__model_confidence"], 0.80)
    assert np.isclose(hit["Effusion__confidence"], 0.0)

    miss = structured.loc[structured["StudyInstanceUID"] == "non_gold_miss"].iloc[0]
    assert miss["hybrid_cache_match"] in (False, np.bool_(False))
    for target in TARGETS:
        assert miss[f"{target}__state"] == "unmentioned"
        assert np.isclose(miss[f"{target}__confidence"], 0.0)

    frame, policy, loaded_audit = load_hybrid_export(out)
    assert policy["formal_b23_compatible"] is False
    assert loaded_audit["formal_b24_eligible"] is False

    active_uids, y, w, summary = prepare_b7_supervision(train, frame)
    assert active_uids == ["non_gold_hit"]
    assert y.shape == (1, len(TARGETS))
    assert w.shape == (1, len(TARGETS))
    assert summary["active_studies"] == 1
    assert summary["usable_cells"] == 1
