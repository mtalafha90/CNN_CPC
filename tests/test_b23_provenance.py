"""Reproducibility guarantees for B23 labelling.

Competition use requires the label-generating function to be identifiable and
re-runnable. These tests pin that contract: only an openly downloadable
checkpoint, executed locally, pinned to an exact revision and decoded greedily
may produce an export that reaches training.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from rsna_knee.b23_llm_labels import SYSTEM_PROMPT, load_frozen_b23_export, run_b23_export
from rsna_knee.b23_local_llm import (
    BACKEND_HOSTED_API,
    BACKEND_LOCAL_TRANSFORMERS,
    BACKEND_LOCAL_VLLM,
    DECODING_GREEDY,
    DEFAULT_LOCAL_MODEL,
    SUGGESTED_LOCAL_MODELS,
    ModelProvenance,
    hash_local_weights,
    load_provenance,
    looks_openly_downloadable,
    prompt_sha256,
)
from rsna_knee.constants import TARGETS


def _provenance(**overrides) -> ModelProvenance:
    fields = dict(
        backend=BACKEND_LOCAL_TRANSFORMERS,
        model_id=DEFAULT_LOCAL_MODEL,
        revision="c" * 40,
        dtype="bfloat16",
        quantisation="none",
        decoding=DECODING_GREEDY,
        max_new_tokens=2048,
        seed=2026,
        prompt_sha256=prompt_sha256(SYSTEM_PROMPT),
        openly_downloadable=True,
    )
    fields.update(overrides)
    return ModelProvenance(**fields)


def test_the_default_model_is_an_openly_downloadable_checkpoint():
    assert looks_openly_downloadable(DEFAULT_LOCAL_MODEL)
    assert DEFAULT_LOCAL_MODEL in SUGGESTED_LOCAL_MODELS


@pytest.mark.parametrize(
    "model_id",
    ["Qwen/Qwen2.5-14B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3", "meta-llama/Llama-3.1-8B"],
)
def test_open_checkpoint_families_are_recognised(model_id):
    assert looks_openly_downloadable(model_id)


@pytest.mark.parametrize("model_id", ["", "claude-sonnet-5", "gpt-4o", "/opt/models/mystery"])
def test_unidentifiable_models_are_not_openly_downloadable(model_id):
    # A bare local path is rejected too: it tells a third party nothing about
    # which weights produced the labels.
    assert not looks_openly_downloadable(model_id)


def test_prompt_sha256_pins_the_instruction_half_of_the_function():
    assert prompt_sha256(SYSTEM_PROMPT) == hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
    assert prompt_sha256(SYSTEM_PROMPT) != prompt_sha256(SYSTEM_PROMPT + " ")


def test_a_pinned_local_greedy_run_is_reproducible():
    assert _provenance().reproducible


@pytest.mark.parametrize(
    "overrides",
    [
        {"backend": BACKEND_HOSTED_API},          # hosted weights can change
        {"openly_downloadable": False},           # not publicly obtainable
        {"revision": "unknown"},                  # not pinned to an artefact
        {"revision": ""},                         # not pinned at all
        {"decoding": "sampling"},                 # output depends on RNG
    ],
)
def test_unpinnable_configurations_are_not_reproducible(overrides):
    assert not _provenance(**overrides).reproducible


def test_local_vllm_is_also_a_reproducible_backend():
    assert _provenance(backend=BACKEND_LOCAL_VLLM).reproducible


def test_describe_surfaces_the_identifying_fields():
    text = _provenance().describe()
    for fragment in (DEFAULT_LOCAL_MODEL, "c" * 40, "bfloat16", DECODING_GREEDY):
        assert fragment in text


def test_hash_local_weights_is_stable_and_content_sensitive(tmp_path):
    (tmp_path / "model-00001.safetensors").write_bytes(b"weights-part-one")
    (tmp_path / "model-00002.safetensors").write_bytes(b"weights-part-two")
    first = hash_local_weights(tmp_path)
    assert first == hash_local_weights(tmp_path)

    (tmp_path / "model-00002.safetensors").write_bytes(b"weights-part-TWO")
    assert hash_local_weights(tmp_path) != first


def test_hash_local_weights_rejects_a_directory_without_shards(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="no .safetensors"):
        hash_local_weights(tmp_path)


def test_load_provenance_round_trips_through_json(tmp_path):
    original = _provenance(weights_sha256="d" * 64)
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"provenance": original.to_dict()}), encoding="utf-8")
    assert load_provenance(path) == original


def _train_csv(tmp_path):
    rows = []
    for i in range(3):
        row = {"StudyInstanceUID": f"uid-{i}", "Report": f"knee report {i}"}
        for target in TARGETS:
            row[target] = np.nan
        rows.append(row)
    gold = {"StudyInstanceUID": "uid-gold", "Report": "gold report"}
    for j, target in enumerate(TARGETS):
        gold[target] = float(j % 2)
    rows.append(gold)
    path = tmp_path / "train.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _backend(system, user):
    findings = {
        target: {"state": "negated", "confidence": 0.9, "evidence": "normal"} for target in TARGETS
    }
    return json.dumps({"findings": findings})


def test_export_refuses_to_run_without_provenance(tmp_path):
    with pytest.raises(ValueError, match="requires ModelProvenance"):
        run_b23_export(_train_csv(tmp_path), _backend, out_root=tmp_path / "e", progress_every=0)


def test_export_refuses_an_unreproducible_hosted_model(tmp_path):
    with pytest.raises(ValueError, match="not reproducible"):
        run_b23_export(
            _train_csv(tmp_path),
            _backend,
            out_root=tmp_path / "e",
            progress_every=0,
            provenance=_provenance(backend=BACKEND_HOSTED_API, openly_downloadable=False),
        )


def test_export_records_provenance_in_both_audit_and_policy(tmp_path):
    out = tmp_path / "export"
    audit = run_b23_export(
        _train_csv(tmp_path),
        _backend,
        out_root=out,
        progress_every=0,
        provenance=_provenance(weights_sha256="e" * 64),
    )
    assert audit["external_model_reproducible"] is True
    assert audit["provenance"]["model_id"] == DEFAULT_LOCAL_MODEL
    assert audit["provenance"]["revision"] == "c" * 40
    assert audit["provenance"]["weights_sha256"] == "e" * 64
    assert audit["provenance"]["prompt_sha256"] == prompt_sha256(SYSTEM_PROMPT)

    policy = json.loads((out / "policy.json").read_text(encoding="utf-8"))
    assert policy["provenance"]["decoding"] == DECODING_GREEDY
    assert load_provenance(out / "policy.json").reproducible


def test_an_unreproducible_export_cannot_be_loaded_for_training(tmp_path):
    out = tmp_path / "export"
    run_b23_export(
        _train_csv(tmp_path),
        _backend,
        out_root=out,
        progress_every=0,
        provenance=_provenance(backend=BACKEND_HOSTED_API, openly_downloadable=False),
        require_reproducible=False,  # development-only escape hatch
    )
    # The escape hatch lets the file be written, but never lets it reach training.
    with pytest.raises(ValueError, match="not produced by a reproducible"):
        load_frozen_b23_export(out)
    frame, _policy, _audit = load_frozen_b23_export(out, require_reproducible=False)
    assert len(frame) == 3


def test_a_partial_smoke_test_export_can_never_be_used_for_training(tmp_path):
    out = tmp_path / "smoke"
    audit = run_b23_export(
        _train_csv(tmp_path),
        _backend,
        out_root=out,
        progress_every=0,
        provenance=_provenance(),
        limit=2,
    )
    assert audit["partial_smoke_test"] is True
    assert audit["n_studies"] == 2
    assert audit["scope"] == "smoke"
    with pytest.raises(ValueError, match="throwaway smoke test"):
        load_frozen_b23_export(out)


def test_the_cache_key_binds_the_report_to_the_labelling_function():
    from rsna_knee.b23_llm_labels import extraction_cache_key

    base = extraction_cache_key("report-sha", _provenance())
    assert base == extraction_cache_key("report-sha", _provenance())
    # A different report, prompt, model, revision or decoding is a different key,
    # so a stale extraction can never be replayed under new provenance.
    assert base != extraction_cache_key("other-report", _provenance())
    assert base != extraction_cache_key("report-sha", _provenance(prompt_sha256="9" * 64))
    assert base != extraction_cache_key("report-sha", _provenance(model_id="qwen3:8b"))
    assert base != extraction_cache_key("report-sha", _provenance(revision="9" * 40))
    assert base != extraction_cache_key("report-sha", _provenance(max_new_tokens=4096))


def test_a_prompt_change_forces_re_extraction_rather_than_a_stale_replay(tmp_path):
    """The failure this prevents: an export that misdescribes its own labels."""
    calls = {"n": 0}

    def _counting(system, user):
        calls["n"] += 1
        return _backend(system, user)

    train = _train_csv(tmp_path)
    cache = tmp_path / "cache.jsonl"
    run_b23_export(
        train, _counting, out_root=tmp_path / "a", progress_every=0,
        cache_path=cache, provenance=_provenance(),
    )
    first = calls["n"]
    assert first > 0

    # Same cache file, same reports, but the prompt changed.
    run_b23_export(
        train, _counting, out_root=tmp_path / "b", progress_every=0,
        cache_path=cache, provenance=_provenance(prompt_sha256="0" * 64),
    )
    assert calls["n"] == first * 2, "a prompt change must re-extract, not replay"
