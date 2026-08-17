from __future__ import annotations

import json

import pandas as pd
import pytest

from rsna_knee.b23_local_llm import ModelProvenance
from rsna_knee.constants import TARGETS
from rsna_knee.report_translation_rescue_full import (
    PHASE6_MAX_NEW_TOKENS,
    PHASE6_MODEL_ID,
    PHASE6_OLLAMA_DIGEST,
    PHASE6_PROMPT_SHA256,
    PHASE6_QUANTISATION,
    PHASE6_SEED,
    _load_cache,
    _original_b6_summary,
    _target_rows,
    validate_phase6_provenance,
)


def _provenance(**overrides) -> ModelProvenance:
    payload = dict(
        backend="ollama",
        model_id=PHASE6_MODEL_ID,
        revision=PHASE6_OLLAMA_DIGEST,
        dtype="gguf",
        quantisation=PHASE6_QUANTISATION,
        decoding="greedy",
        max_new_tokens=PHASE6_MAX_NEW_TOKENS,
        seed=PHASE6_SEED,
        prompt_sha256=PHASE6_PROMPT_SHA256,
        openly_downloadable=True,
        weights_sha256=None,
        ollama_model_digest=PHASE6_OLLAMA_DIGEST,
    )
    payload.update(overrides)
    return ModelProvenance(**payload)


def test_phase7_requires_exact_phase6_translator_provenance():
    validate_phase6_provenance(_provenance())
    with pytest.raises(RuntimeError, match="does not match frozen Phase-6 provenance"):
        validate_phase6_provenance(_provenance(revision="different-digest"))
    with pytest.raises(RuntimeError, match="does not match frozen Phase-6 provenance"):
        validate_phase6_provenance(_provenance(seed=2027))


def test_original_b6_summary_counts_only_definite_high_confidence_cells():
    row = {}
    for target in TARGETS:
        row[f"{target}__state"] = "unmentioned"
        row[f"{target}__confidence"] = 0.0
    row["ACL__state"] = "positive"
    row["ACL__confidence"] = 0.90
    row["MCL__state"] = "negated"
    row["MCL__confidence"] = 0.90
    row["Effusion__state"] = "positive"
    row["Effusion__confidence"] = 0.50
    summary = _original_b6_summary(pd.Series(row))
    assert summary["usable_cells"] == 2
    assert summary["positive_cells"] == 1
    assert summary["negative_cells"] == 1


def test_target_rows_exports_only_usable_translated_b6_cells():
    snapshot = {"targets": {}}
    for target in TARGETS:
        snapshot["targets"][target] = {
            "state": "unmentioned",
            "confidence": 0.0,
            "probability": 0.5,
        }
    snapshot["ACL"] = None
    snapshot["targets"]["ACL"] = {
        "state": "positive",
        "confidence": 0.9,
        "probability": 0.97,
    }
    snapshot["targets"]["MCL"] = {
        "state": "negated",
        "confidence": 0.9,
        "probability": 0.03,
    }
    snapshot["targets"]["Effusion"] = {
        "state": "positive",
        "confidence": 0.2,
        "probability": 0.97,
    }
    rows = _target_rows("uid", "Greek", snapshot)
    assert [(row["target"], row["state"]) for row in rows] == [
        ("ACL", "positive"),
        ("MCL", "negated"),
    ]


def test_phase7_cache_rejects_duplicate_uids(tmp_path):
    path = tmp_path / "translation_cache.jsonl"
    row = {"StudyInstanceUID": "same", "translation": "x"}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="duplicate UID"):
        _load_cache(path)
