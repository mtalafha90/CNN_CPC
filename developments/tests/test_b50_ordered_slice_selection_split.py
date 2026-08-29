"""B50's fresh model-selection boundary must not reuse B48/B49 validation rows."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from rsna_knee.data import TARGETS
from rsna_knee.b50_ordered_slice_selection_split import (
    B50_SELECTION_SPLIT_VERSION,
    B50_SPLIT_EXCLUDED,
    B50_SPLIT_SEEN,
    B50_SPLIT_TRAIN,
    B50_SPLIT_UNSEEN,
    build_b50_selection_rows,
    load_b48_parent_split,
    verify_b50_selection_split,
    write_b50_selection_split,
)
from rsna_knee.domain_shift_split import DOMAIN_SPLIT_VERSION


def _surface(n_profiles: int = 6, studies_per_profile: int = 8):
    rows = []
    for profile in range(n_profiles):
        for study in range(studies_per_profile):
            rows.append(
                {
                    "StudyInstanceUID": f"study-{profile}-{study}",
                    "scanner_profile": f"scanner-{profile}",
                    "series_count": 2,
                    "distinct_profiles": 1,
                }
            )
    profiles = pd.DataFrame(rows)
    uids = profiles["StudyInstanceUID"].tolist()
    # Every scanner has both classes for every target, avoiding a random
    # synthetic failure unrelated to the split mechanism being tested.
    targets = np.asarray(
        [[float((index + target) % 3 == 0) for target in range(len(TARGETS))] for index in range(len(uids))]
    )
    weights = np.ones_like(targets)
    parent = profiles[["StudyInstanceUID", "scanner_profile"]].copy()
    parent["holdout"] = False
    parent["split"] = "train"
    parent.loc[parent["scanner_profile"].eq("scanner-4"), "split"] = "validation_seen_scanners"
    parent.loc[parent["scanner_profile"].eq("scanner-5"), ["split", "holdout"]] = [
        "holdout_unseen_scanners",
        True,
    ]
    return profiles, uids, targets, weights, parent


def test_b50_uses_only_parent_training_rows_for_its_new_gate():
    profiles, uids, targets, weights, parent = _surface()
    rows, summary, _holdout_profiles = build_b50_selection_rows(
        profiles, uids, targets, weights, parent
    )
    verify_b50_selection_split(rows)

    spent = rows["parent_b48_split"].ne("train")
    assert rows.loc[spent, "b50_split"].eq(B50_SPLIT_EXCLUDED).all()
    assert not rows.loc[~spent, "b50_split"].eq(B50_SPLIT_EXCLUDED).any()
    assert set(rows["b50_split"]) == {
        B50_SPLIT_TRAIN,
        B50_SPLIT_SEEN,
        B50_SPLIT_UNSEEN,
        B50_SPLIT_EXCLUDED,
    }
    assert summary["selection_pool_studies"] == 32
    assert summary["excluded_prior_surface_studies"] == 16
    assert summary["comparable_targets"] == list(TARGETS)


def test_b50_unseen_profiles_do_not_straddle_train_or_seen():
    profiles, uids, targets, weights, parent = _surface()
    rows, _summary, _holdout_profiles = build_b50_selection_rows(
        profiles, uids, targets, weights, parent
    )
    train = set(rows.loc[rows["b50_split"].eq(B50_SPLIT_TRAIN), "scanner_profile"])
    seen = set(rows.loc[rows["b50_split"].eq(B50_SPLIT_SEEN), "scanner_profile"])
    unseen = set(rows.loc[rows["b50_split"].eq(B50_SPLIT_UNSEEN), "scanner_profile"])
    assert unseen.isdisjoint(train)
    assert unseen.isdisjoint(seen)
    assert seen.issubset(train)


def test_b50_split_is_reproducible_when_parent_rows_arrive_shuffled():
    profiles, uids, targets, weights, parent = _surface()
    first, first_summary, first_profiles = build_b50_selection_rows(
        profiles, uids, targets, weights, parent
    )
    second, second_summary, second_profiles = build_b50_selection_rows(
        profiles.sample(frac=1.0, random_state=4).reset_index(drop=True),
        uids,
        targets,
        weights,
        parent.sample(frac=1.0, random_state=9).reset_index(drop=True),
    )
    assert first_profiles == second_profiles
    assert first_summary == second_summary
    assert first.sort_values("StudyInstanceUID").reset_index(drop=True).equals(
        second.sort_values("StudyInstanceUID").reset_index(drop=True)
    )


def test_b50_refuses_to_reuse_a_parent_validation_row():
    profiles, uids, targets, weights, parent = _surface()
    rows, _summary, _holdout_profiles = build_b50_selection_rows(
        profiles, uids, targets, weights, parent
    )
    victim = rows.index[rows["parent_b48_split"].ne("train")][0]
    rows.loc[victim, "b50_split"] = B50_SPLIT_TRAIN
    with pytest.raises(ValueError, match="reuse a B48/B49 validation row"):
        verify_b50_selection_split(rows)


def test_b50_refuses_a_profile_mismatch_against_the_frozen_parent():
    profiles, uids, targets, weights, parent = _surface()
    parent = parent.copy()
    parent.loc[parent.index[0], "scanner_profile"] = "changed-scanner"
    with pytest.raises(ValueError, match="scanner profiles differ"):
        build_b50_selection_rows(profiles, uids, targets, weights, parent)


def test_b50_writer_is_hashed_and_write_once(tmp_path):
    profiles, uids, targets, weights, parent = _surface()
    rows, summary, holdout_profiles = build_b50_selection_rows(
        profiles, uids, targets, weights, parent
    )
    payload = {
        "version": B50_SELECTION_SPLIT_VERSION,
        "summary": summary,
        "unseen_scanner_profiles": holdout_profiles,
    }
    path = write_b50_selection_split(payload, rows, out_root=tmp_path)
    assert json.loads(path.read_text())["version"] == B50_SELECTION_SPLIT_VERSION
    assert (tmp_path / "b50_selection_split.sha256").read_text().strip() == hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    with pytest.raises(FileExistsError, match="already exists"):
        write_b50_selection_split(payload, rows, out_root=tmp_path)


def test_parent_reader_rejects_a_tampered_hash(tmp_path):
    payload = tmp_path / "domain_split.json"
    payload.write_text(json.dumps({"version": DOMAIN_SPLIT_VERSION}))
    (tmp_path / "domain_split_by_study.csv").write_text(
        "StudyInstanceUID,scanner_profile,holdout,split\n"
        "a,scanner-a,False,train\n"
        "b,scanner-b,True,holdout_unseen_scanners\n"
    )
    (tmp_path / "domain_split.sha256").write_text("wrong\n")
    with pytest.raises(ValueError, match="SHA-256"):
        load_b48_parent_split(tmp_path)
