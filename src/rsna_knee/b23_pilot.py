"""Build a scoped B23 export from extractions already in the cache.

A long labelling run that stops part-way leaves real, paid-for work behind: the
cache holds every report that succeeded. This module turns that into a valid,
explicitly scoped export without calling the model again.

## Three scopes, and why the distinction matters

```text
full    every report in train.csv was labelled
pilot   a declared subset, deliberately scoped; VALID for training
smoke   a throwaway --limit run; REFUSED for training
```

A pilot is a legitimate experiment with a stated size. A smoke test is a
correctness check on twenty reports. Both are partial, but only one is
something you would draw a conclusion from, so they are marked differently and
the loader treats them differently.

## What a pilot can and cannot show

The weak pipeline already trains on the 3,120 report-only studies B6 activates.
A pilot smaller than that trains on *fewer* studies than B20 did, so its score
is not comparable to B20's 0.6672 -- the training-set size differs as well as
the labels.

What a pilot *can* do is compare B6 and B23 supervision on the same studies,
which is exactly the B24 matched design. A pilot answers "do these labels beat
those labels at this scale", not "is this better than B20".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .b23_llm_labels import (
    B23_STATES,
    ExtractionCache,
    TargetExtraction,
    extraction_cache_key,
    run_b23_export,
)
from .b23_local_llm import ModelProvenance, load_provenance
from .constants import TARGETS
from .data import load_train_csv, normalize_report, report_hash

SCOPE_FULL = "full"
SCOPE_PILOT = "pilot"
SCOPE_SMOKE = "smoke"


def cached_study_coverage(
    train_csv: str | Path,
    cache_path: str | Path,
    provenance: ModelProvenance,
) -> dict:
    """Report which studies the cache can already supply, without calling a model.

    Matching is by full cache key, not by report hash alone, so an entry
    produced by a different prompt or a different model is correctly treated as
    a miss rather than silently reused.
    """
    df = load_train_csv(train_csv)
    cache = ExtractionCache(cache_path)
    reports = df["Report"].fillna("").astype(str)
    uids = df["StudyInstanceUID"].astype(str).tolist()

    hit_uids: list[str] = []
    miss_uids: list[str] = []
    empty_uids: list[str] = []
    for uid, report in zip(uids, reports.tolist()):
        if not normalize_report(report):
            empty_uids.append(uid)
            continue
        key = extraction_cache_key(report_hash(report), provenance)
        (hit_uids if cache.get(key) is not None else miss_uids).append(uid)

    return {
        "total_studies": len(uids),
        "cached_studies": len(hit_uids),
        "uncached_studies": len(miss_uids),
        "empty_report_studies": len(empty_uids),
        "cache_entries": len(cache),
        "cached_uids": hit_uids,
        "uncached_uids": miss_uids,
        "provenance": provenance.to_dict(),
    }


class _CacheOnlyBackend:
    """A backend that serves the cache and refuses to invent anything.

    If a report is not already cached under this exact labelling function, that
    is an error rather than a fresh call: a pilot must be built only from work
    that has actually been done, or its declared size would be a fiction.
    """

    def __init__(self):
        self.misses = 0

    def __call__(self, system: str, user: str) -> str:  # pragma: no cover - guard
        self.misses += 1
        raise RuntimeError(
            "cache-only pilot hit an unlabelled report; restrict --pilot-size to "
            "the number of cached studies, or run the labeller to fill the gap"
        )


def build_pilot_export(
    train_csv: str | Path,
    cache_path: str | Path,
    provenance: ModelProvenance,
    *,
    out_root: str | Path,
    pilot_size: int | None = None,
    min_confidence: float = 0.75,
) -> dict:
    """Write a valid B23 export covering only the cached studies.

    The pilot keeps studies in `train.csv` order, so it is deterministic and
    reproducible: the same cache and the same size always give the same export.
    """
    coverage = cached_study_coverage(train_csv, cache_path, provenance)
    cached = coverage["cached_uids"]
    if not cached:
        raise ValueError(
            "no cached extractions match this labelling function. If the prompt "
            "or the model changed since the cache was written, those entries "
            "cannot be reused -- the export would misdescribe its own labels."
        )

    size = len(cached) if pilot_size is None else min(int(pilot_size), len(cached))
    if size < 2:
        raise ValueError("a pilot needs at least two studies")
    selected = set(cached[:size])

    df = load_train_csv(train_csv)
    df["StudyInstanceUID"] = df["StudyInstanceUID"].astype(str)
    subset = df.loc[df["StudyInstanceUID"].isin(selected)].copy()

    scratch = Path(out_root)
    scratch.mkdir(parents=True, exist_ok=True)
    subset_csv = scratch / "pilot_train_subset.csv"
    subset.to_csv(subset_csv, index=False)

    backend = _CacheOnlyBackend()
    audit = run_b23_export(
        subset_csv,
        backend,
        out_root=out_root,
        min_confidence=min_confidence,
        cache_path=cache_path,
        provenance=provenance,
        progress_every=0,
    )

    # Re-stamp the export as a declared pilot rather than a full run.
    audit_path = Path(out_root) / "audit.json"
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    payload["scope"] = SCOPE_PILOT if size < coverage["total_studies"] else SCOPE_FULL
    payload["pilot_size"] = int(size)
    payload["pilot_available_cached"] = int(len(cached))
    payload["pilot_source"] = "cache-only; no model calls were made"
    payload["partial_smoke_test"] = False
    payload["comparability_note"] = (
        "A pilot smaller than the 3,120 studies B6 activates trains on fewer "
        "studies than B20 did, so its score is not directly comparable to B20's "
        "0.6672. Use it to compare B6 against B23 supervision on the SAME "
        "studies, which is the B24 matched design."
    )
    audit_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["coverage"] = {
        k: v for k, v in coverage.items() if k not in ("cached_uids", "uncached_uids")
    }
    return payload


def format_pilot(payload: dict) -> str:
    coverage = payload.get("coverage", {})
    return "\n".join(
        [
            f"B23 pilot export ({payload.get('scope')})",
            f"  studies labelled            {payload.get('pilot_size')}",
            f"  cached and available        {payload.get('pilot_available_cached')}",
            f"  model calls made            0 (cache-only)",
            "",
            f"  usable cells                {payload.get('usable_cells_total')}"
            f" of {payload.get('possible_cells_total')}"
            f"  ({payload.get('cell_coverage', 0):.1%})",
            f"  gold rows in training       {payload.get('gold_rows_in_training_targets')}",
            f"  reproducible provenance     {payload.get('external_model_reproducible')}",
            "",
            f"  cache entries total         {coverage.get('cache_entries')}",
            f"  still unlabelled            {coverage.get('uncached_studies')}",
            "",
            "  " + str(payload.get("comparability_note", "")),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a B23 pilot export from cached extractions (no model calls)"
    )
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--cache", required=True, help="extraction_cache.jsonl from the run")
    parser.add_argument(
        "--provenance",
        required=True,
        help="policy.json or audit.json from the interrupted run, to match cache keys",
    )
    parser.add_argument("--out-root", required=True)
    parser.add_argument(
        "--pilot-size",
        type=int,
        default=None,
        help="declare a size; defaults to every cached study",
    )
    parser.add_argument("--min-confidence", type=float, default=0.75)
    args = parser.parse_args()

    provenance = load_provenance(args.provenance)
    payload = build_pilot_export(
        args.train_csv,
        args.cache,
        provenance,
        out_root=args.out_root,
        pilot_size=args.pilot_size,
        min_confidence=args.min_confidence,
    )
    print(format_pilot(payload))


if __name__ == "__main__":  # pragma: no cover
    main()
