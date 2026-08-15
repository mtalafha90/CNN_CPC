"""Salvaging a partially completed labelling run into a declared pilot."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from rsna_knee.b23_llm_labels import (
    ExtractionCache,
    extraction_cache_key,
    load_frozen_b23_export,
    run_b23_export,
)
from rsna_knee.b23_local_llm import (
    BACKEND_OLLAMA,
    DECODING_GREEDY,
    ModelProvenance,
)
from rsna_knee.b23_pilot import (
    SCOPE_FULL,
    SCOPE_PILOT,
    build_pilot_export,
    cached_study_coverage,
    format_pilot,
)
from rsna_knee.constants import TARGETS

N_STUDIES = 30


def _provenance(**overrides) -> ModelProvenance:
    fields = dict(
        backend=BACKEND_OLLAMA,
        model_id="qwen3:14b",
        revision="a" * 64,
        dtype="gguf",
        quantisation="Q4_K_M",
        decoding=DECODING_GREEDY,
        max_new_tokens=4096,
        seed=2026,
        prompt_sha256="b" * 64,
        openly_downloadable=True,
    )
    fields.update(overrides)
    return ModelProvenance(**fields)


def _train_csv(tmp_path):
    rows = []
    for i in range(N_STUDIES):
        row = {"StudyInstanceUID": f"uid-{i:03d}", "Report": f"knee report number {i}"}
        for target in TARGETS:
            row[target] = np.nan
        rows.append(row)
    gold = {"StudyInstanceUID": "uid-gold", "Report": "gold knee report"}
    for j, target in enumerate(TARGETS):
        gold[target] = float(j % 2)
    rows.append(gold)
    path = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _seed_cache(tmp_path, train_csv, provenance, *, n_done):
    """Emulate a run that got through `n_done` reports and then died."""
    from rsna_knee.data import report_hash

    cache_path = tmp_path / "extraction_cache.jsonl"
    cache = ExtractionCache(cache_path)
    df = pd.read_csv(train_csv)
    for _, row in df.head(n_done).iterrows():
        key = extraction_cache_key(report_hash(str(row["Report"])), provenance)
        findings = {}
        for j, target in enumerate(TARGETS):
            state = "positive" if j % 3 == 0 else "negated"
            findings[target] = {"state": state, "confidence": 0.9, "evidence": "ev"}
        cache.put(key, {"cache_key": key, "report_sha1": report_hash(str(row["Report"])), "findings": findings})
    return cache_path


def test_coverage_reports_what_the_cache_can_already_supply(tmp_path):
    provenance = _provenance()
    train_csv = _train_csv(tmp_path)
    cache = _seed_cache(tmp_path, train_csv, provenance, n_done=12)

    coverage = cached_study_coverage(train_csv, cache, provenance)
    assert coverage["cached_studies"] == 12
    assert coverage["uncached_studies"] == N_STUDIES + 1 - 12
    assert coverage["cache_entries"] == 12


def test_a_changed_prompt_makes_every_cached_entry_a_miss(tmp_path):
    """Reusing entries after a prompt change would misdescribe the labels."""
    train_csv = _train_csv(tmp_path)
    cache = _seed_cache(tmp_path, train_csv, _provenance(), n_done=12)

    coverage = cached_study_coverage(train_csv, cache, _provenance(prompt_sha256="9" * 64))
    assert coverage["cached_studies"] == 0


def test_a_pilot_is_built_from_the_cache_without_any_model_call(tmp_path):
    provenance = _provenance()
    train_csv = _train_csv(tmp_path)
    cache = _seed_cache(tmp_path, train_csv, provenance, n_done=20)

    payload = build_pilot_export(
        train_csv, cache, provenance, out_root=tmp_path / "pilot", pilot_size=15
    )
    assert payload["scope"] == SCOPE_PILOT
    assert payload["pilot_size"] == 15
    assert payload["pilot_available_cached"] == 20
    assert payload["partial_smoke_test"] is False
    assert "no model calls" in payload["pilot_source"]


def test_a_pilot_export_is_valid_for_training(tmp_path):
    """Unlike a smoke test, a declared pilot loads."""
    provenance = _provenance()
    train_csv = _train_csv(tmp_path)
    cache = _seed_cache(tmp_path, train_csv, provenance, n_done=20)
    build_pilot_export(
        train_csv, cache, provenance, out_root=tmp_path / "pilot", pilot_size=15
    )

    frame, policy, audit = load_frozen_b23_export(tmp_path / "pilot")
    assert len(frame) == 15
    assert audit["scope"] == SCOPE_PILOT
    assert audit["gold_rows_in_training_targets"] == 0


def test_a_smoke_test_export_is_still_refused(tmp_path):
    """The distinction the pilot introduces must not weaken the smoke guard."""
    provenance = _provenance()
    train_csv = _train_csv(tmp_path)

    def _backend(system, user):
        findings = {
            t: {"state": "negated", "confidence": 0.9, "evidence": "n"} for t in TARGETS
        }
        return json.dumps({"findings": findings})

    run_b23_export(
        train_csv, _backend, out_root=tmp_path / "smoke", progress_every=0,
        provenance=provenance, limit=5,
    )
    with pytest.raises(ValueError, match="throwaway smoke test"):
        load_frozen_b23_export(tmp_path / "smoke")


def test_requesting_more_than_is_cached_is_capped_not_invented(tmp_path):
    provenance = _provenance()
    train_csv = _train_csv(tmp_path)
    cache = _seed_cache(tmp_path, train_csv, provenance, n_done=10)

    payload = build_pilot_export(
        train_csv, cache, provenance, out_root=tmp_path / "pilot", pilot_size=999
    )
    # The cache-only backend would raise rather than fabricate a label.
    assert payload["pilot_size"] == 10


def test_a_pilot_covering_everything_is_recorded_as_full(tmp_path):
    provenance = _provenance()
    train_csv = _train_csv(tmp_path)
    cache = _seed_cache(tmp_path, train_csv, provenance, n_done=N_STUDIES + 1)

    payload = build_pilot_export(train_csv, cache, provenance, out_root=tmp_path / "pilot")
    assert payload["scope"] == SCOPE_FULL


def test_an_empty_cache_is_refused_with_the_reason(tmp_path):
    provenance = _provenance()
    train_csv = _train_csv(tmp_path)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no cached extractions"):
        build_pilot_export(train_csv, empty, provenance, out_root=tmp_path / "pilot")


def test_the_pilot_summary_states_the_comparability_limit(tmp_path):
    provenance = _provenance()
    train_csv = _train_csv(tmp_path)
    cache = _seed_cache(tmp_path, train_csv, provenance, n_done=20)
    payload = build_pilot_export(
        train_csv, cache, provenance, out_root=tmp_path / "pilot", pilot_size=15
    )
    text = format_pilot(payload)
    assert "cache-only" in text
    # A pilot smaller than B6's 3,120 studies is not comparable to B20 directly.
    assert "not directly comparable to B20" in text


def test_the_pilot_is_deterministic_for_a_fixed_size(tmp_path):
    provenance = _provenance()
    train_csv = _train_csv(tmp_path)
    cache = _seed_cache(tmp_path, train_csv, provenance, n_done=20)

    first = build_pilot_export(
        train_csv, cache, provenance, out_root=tmp_path / "a", pilot_size=12
    )
    second = build_pilot_export(
        train_csv, cache, provenance, out_root=tmp_path / "b", pilot_size=12
    )
    frame_a = pd.read_csv(tmp_path / "a" / "training_targets.csv")
    frame_b = pd.read_csv(tmp_path / "b" / "training_targets.csv")
    assert first["pilot_size"] == second["pilot_size"]
    pd.testing.assert_frame_equal(frame_a, frame_b)
