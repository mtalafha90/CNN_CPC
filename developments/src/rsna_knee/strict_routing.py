"""Strict semantic routing for B9.

The historical dual-stream selector intentionally tried to populate both the
fluid and structural slot of a plane whenever multiple series were available.
That can place a fluid-sensitive series in a structural slot (or vice versa)
when a study has multiple acquisitions of only one contrast class.

B9 keeps the historical selector untouched for reproducibility and uses this
module instead.  A fluid slot may contain only ``Fluid_Sensitive == True`` and
a structural slot only ``Fluid_Sensitive == False``.  Unknown contrast stays
unselected after the normal metadata-repair pass.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .constants import DUAL_STREAMS
from .data import PLANES, build_series_index

STRICT_ROUTING_POLICY = "fluid_sensitive_exact_v1"


def _rank_indices(score: np.ndarray) -> list[int]:
    return np.argsort(-score, kind="mergesort").astype(int).tolist()


def _select_strict_from_study(part: pd.DataFrame) -> dict[str, str | None]:
    """Return six dual streams without cross-contrast substitution."""
    result: dict[str, str | None] = {}
    for plane in PLANES:
        p = part.loc[part["Anatomical_Plane"].eq(plane)].reset_index(drop=True)
        key = plane.lower()
        fluid_key = f"{key}_fluid"
        structural_key = f"{key}_structural"
        result[fluid_key] = None
        result[structural_key] = None
        if p.empty:
            continue

        known = p["Fluid_Sensitive"].notna().to_numpy()
        fluid_flag = p["Fluid_Sensitive"].fillna(False).astype(bool).to_numpy()
        fat_flag = p["Fat_Suppression"].fillna(False).astype(bool).to_numpy()

        # Keep the historical within-class ranking preference while forbidding
        # the historical cross-class fallback.  This makes B9's scientific
        # change routing semantics rather than a new ranking heuristic.
        fluid_candidates = np.flatnonzero(known & fluid_flag)
        if fluid_candidates.size:
            score = 2 * fluid_flag.astype(int) + 2 * fat_flag.astype(int)
            ranked = [i for i in _rank_indices(score) if i in set(fluid_candidates.tolist())]
            result[fluid_key] = str(p.at[ranked[0], "SeriesInstanceUID"])

        structural_candidates = np.flatnonzero(known & (~fluid_flag))
        if structural_candidates.size:
            score = 2 * (~fat_flag).astype(int) + (~fluid_flag).astype(int)
            ranked = [i for i in _rank_indices(score) if i in set(structural_candidates.tolist())]
            result[structural_key] = str(p.at[ranked[0], "SeriesInstanceUID"])

    return result


def build_strict_series_index(
    series_df: pd.DataFrame,
    studies: Iterable[str],
) -> dict[str, dict[str, str | None]]:
    """Build a six-stream index with exact Fluid_Sensitive semantics."""
    work = series_df.copy()
    work["StudyInstanceUID"] = work["StudyInstanceUID"].astype(str)
    work["SeriesInstanceUID"] = work["SeriesInstanceUID"].astype(str)
    grouped = {uid: part for uid, part in work.groupby("StudyInstanceUID", sort=False)}
    empty = work.iloc[0:0]
    return {
        str(uid): _select_strict_from_study(grouped.get(str(uid), empty))
        for uid in studies
    }


def routing_audit(series_df: pd.DataFrame, studies: Iterable[str]) -> dict:
    """Compare historical dual routing with B9 strict routing.

    The audit is label-free.  It uses only series metadata and therefore can be
    recorded before any gold evaluation.
    """
    uids = [str(uid) for uid in studies]
    work = series_df.copy()
    work["StudyInstanceUID"] = work["StudyInstanceUID"].astype(str)
    work["SeriesInstanceUID"] = work["SeriesInstanceUID"].astype(str)
    legacy = build_series_index(work, uids, mode="dual")
    strict = build_strict_series_index(work, uids)

    lookup: dict[tuple[str, str], bool | None] = {}
    for row in work.itertuples(index=False):
        value = getattr(row, "Fluid_Sensitive")
        semantic: bool | None
        if pd.isna(value):
            semantic = None
        else:
            semantic = bool(value)
        lookup[(str(getattr(row, "StudyInstanceUID")), str(getattr(row, "SeriesInstanceUID")))] = semantic

    per_stream: dict[str, dict[str, int]] = {}
    total_legacy = total_strict = legacy_mismatch = strict_mismatch = changed = 0
    removed_cross_contrast = 0
    for stream in DUAL_STREAMS:
        expected_fluid = stream.endswith("_fluid")
        row = {
            "legacy_selected": 0,
            "strict_selected": 0,
            "legacy_semantic_mismatch": 0,
            "strict_semantic_mismatch": 0,
            "changed_selection": 0,
            "removed_cross_contrast_substitution": 0,
        }
        for uid in uids:
            old = legacy.get(uid, {}).get(stream)
            new = strict.get(uid, {}).get(stream)
            if old:
                row["legacy_selected"] += 1
                total_legacy += 1
                semantic = lookup.get((uid, str(old)))
                if semantic is None or semantic != expected_fluid:
                    row["legacy_semantic_mismatch"] += 1
                    legacy_mismatch += 1
            if new:
                row["strict_selected"] += 1
                total_strict += 1
                semantic = lookup.get((uid, str(new)))
                if semantic is None or semantic != expected_fluid:
                    row["strict_semantic_mismatch"] += 1
                    strict_mismatch += 1
            if old != new:
                row["changed_selection"] += 1
                changed += 1
            if old and not new:
                semantic = lookup.get((uid, str(old)))
                if semantic is None or semantic != expected_fluid:
                    row["removed_cross_contrast_substitution"] += 1
                    removed_cross_contrast += 1
        per_stream[stream] = row

    if strict_mismatch:
        raise RuntimeError(f"strict routing audit found {strict_mismatch} semantic mismatch(es)")

    return {
        "routing_policy": STRICT_ROUTING_POLICY,
        "studies": len(uids),
        "legacy_selected_streams": total_legacy,
        "strict_selected_streams": total_strict,
        "legacy_semantic_mismatches": legacy_mismatch,
        "strict_semantic_mismatches": strict_mismatch,
        "changed_stream_assignments": changed,
        "removed_cross_contrast_substitutions": removed_cross_contrast,
        "per_stream": per_stream,
    }
