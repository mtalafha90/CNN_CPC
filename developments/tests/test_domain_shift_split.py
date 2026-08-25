"""The scanner-grouped validation surface.

Every split this project has frozen so far was made by hashing the study UID,
which scatters each scanner across both sides. This surface groups studies by
the machine that produced them so that no scanner profile is ever trained on and
then scored, which is the only way to measure whether the model's performance
travels to acquisition settings it has not seen.

The property everything else rests on is that no profile straddles the split, so
that is tested hardest.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from rsna_knee.data import TARGETS
from rsna_knee.domain_shift_split import (
    CONCENTRATION_WARNING_SHARE,
    DOMAIN_SPLIT_SALT,
    build_study_profiles,
    choose_holdout_profiles,
    choose_seen_scanner_validation,
    field_strength_bin,
    scanner_profile,
    summarise_split,
    verify_domain_split,
    write_domain_split,
)


@pytest.mark.parametrize(
    "value,expected",
    [
        (1.5, "1.5T"),
        (1.494, "1.5T"),  # scanners report real numbers, not round ones
        (1.4936, "1.5T"),
        (3.0, "3T"),
        (2.8936, "3T"),
        (0.35, "0.3T"),
        (1.0, "1.0T"),
        (7.0, "7.0T"),
        (None, "Missing"),
        ("", "Missing"),
        (float("nan"), "Missing"),
        (0.0, "Missing"),
        (-1.5, "Missing"),
    ],
)
def test_field_strength_is_bucketed_not_compared_exactly(value, expected):
    assert field_strength_bin(value) == expected


def test_one_scanner_reporting_slightly_different_strengths_stays_one_group():
    """An equality test here would split a single machine into several groups."""
    first = scanner_profile("SIEMENS", "MAGNETOM Aera", 1.5)
    second = scanner_profile("Siemens HealthCare", "MAGNETOM Aera", 1.494)
    assert first == second == "Siemens|MAGNETOM Aera|1.5T"


def test_missing_metadata_becomes_its_own_group_rather_than_vanishing():
    assert scanner_profile(None, None, None) == "Missing|Missing|Missing"


def test_different_machines_do_not_collide():
    assert scanner_profile("GE MEDICAL SYSTEMS", "Optima MR450w", 1.5) != scanner_profile(
        "SIEMENS", "MAGNETOM Aera", 1.5
    )
    assert scanner_profile("SIEMENS", "MAGNETOM Aera", 1.5) != scanner_profile(
        "SIEMENS", "MAGNETOM Aera", 3.0
    ), "field strength is part of the domain"


# --- fixtures -------------------------------------------------------------


def _population(n_profiles=6, studies_per_profile=8, seed=0):
    """A synthetic population: several scanners, several studies each."""
    rng = np.random.default_rng(seed)
    records, rows = [], []
    for profile_index in range(n_profiles):
        maker = ["SIEMENS", "GE MEDICAL SYSTEMS", "PHILIPS"][profile_index % 3]
        model = f"Model{profile_index}"
        strength = 1.5 if profile_index % 2 else 3.0
        for study_index in range(studies_per_profile):
            uid = f"study-{profile_index}-{study_index}"
            for series_index in range(2):
                records.append(
                    {
                        "StudyInstanceUID": uid,
                        "SeriesInstanceUID": f"{uid}-s{series_index}",
                        "manufacturer": maker,
                        "manufacturer_model": model,
                        "magnetic_field_strength_t": strength,
                    }
                )
            rows.append(uid)

    header = pd.DataFrame(records)
    train = pd.DataFrame({"StudyInstanceUID": rows})
    for target in TARGETS:
        train[target] = np.nan  # report-only studies carry no official label

    uids = list(rows)
    targets = (rng.random((len(uids), len(TARGETS))) < 0.4).astype(float)
    weights = np.ones_like(targets)
    return train, header, uids, targets, weights


@pytest.fixture
def population():
    return _population()


@pytest.fixture(autouse=True)
def _no_gold(monkeypatch):
    """The synthetic frames have no official labels, so nothing is gold."""
    import rsna_knee.domain_shift_split as module

    monkeypatch.setattr(module, "gold_mask", lambda frame: pd.Series(False, index=frame.index))


# --- the split ------------------------------------------------------------


def test_a_study_gets_the_profile_its_series_agree_on(population):
    train, header, *_ = population
    profiles = build_study_profiles(train, header)
    assert len(profiles) == 48
    assert profiles["scanner_profile"].nunique() == 6
    assert set(profiles["series_count"]) == {2}


def test_a_study_whose_series_disagree_still_lands_in_exactly_one_group(population):
    """A stitched or re-imported series must not put a study on both sides."""
    train, header, *_ = population
    header = header.copy()
    mixed = header["StudyInstanceUID"] == "study-0-0"
    header.loc[mixed & header["SeriesInstanceUID"].str.endswith("s1"), "manufacturer"] = "PHILIPS"

    profiles = build_study_profiles(train, header)
    row = profiles.loc[profiles["StudyInstanceUID"] == "study-0-0"].iloc[0]
    assert row["distinct_profiles"] == 2
    assert isinstance(row["scanner_profile"], str)
    assert len(profiles.loc[profiles["StudyInstanceUID"] == "study-0-0"]) == 1


def test_the_gold_studies_are_left_out(population):
    """The 58 clean studies belong to B46, not to this surface."""
    import rsna_knee.domain_shift_split as module

    train, header, *_ = population
    held_back = {"study-0-0", "study-1-1"}
    module.gold_mask = lambda frame: frame["StudyInstanceUID"].isin(held_back)
    try:
        profiles = build_study_profiles(train, header)
    finally:
        from rsna_knee.data import gold_mask as original

        module.gold_mask = original
    assert held_back.isdisjoint(set(profiles["StudyInstanceUID"]))
    assert len(profiles) == 46


def test_a_study_with_no_header_row_is_an_error_not_a_silent_drop(population):
    train, header, *_ = population
    header = header.loc[header["StudyInstanceUID"] != "study-2-3"]
    with pytest.raises(ValueError, match="no header rows"):
        build_study_profiles(train, header)


def test_a_header_table_missing_columns_is_refused(population):
    train, header, *_ = population
    with pytest.raises(ValueError, match="missing columns"):
        build_study_profiles(train, header.drop(columns=["manufacturer_model"]))


def test_no_scanner_profile_appears_on_both_sides(population):
    """The property the whole surface rests on."""
    train, header, uids, targets, weights = population
    profiles = build_study_profiles(train, header)
    chosen = choose_holdout_profiles(profiles, targets, weights, uids)

    rows = profiles.copy()
    rows["holdout"] = rows["scanner_profile"].isin(set(chosen))
    verify_domain_split(rows)

    held = set(rows.loc[rows["holdout"], "scanner_profile"])
    kept = set(rows.loc[~rows["holdout"], "scanner_profile"])
    assert held and kept
    assert held.isdisjoint(kept)


def test_a_straddling_profile_is_caught(population):
    train, header, uids, targets, weights = population
    profiles = build_study_profiles(train, header)
    rows = profiles.copy()
    rows["holdout"] = False
    rows.loc[rows.index[:1], "holdout"] = True  # one study, not one whole profile
    with pytest.raises(ValueError, match="both training and holdout"):
        verify_domain_split(rows)


def test_the_split_is_reproducible(population):
    train, header, uids, targets, weights = population
    profiles = build_study_profiles(train, header)
    first = choose_holdout_profiles(profiles, targets, weights, uids)
    second = choose_holdout_profiles(profiles, targets, weights, uids)
    assert first == second


def test_the_salt_decides_only_between_equally_balanced_choices():
    """Balance picks the split; the salt is the tie-break, as in B46's folds.

    When two profiles carry identical label statistics the balance objective
    cannot separate them, and something has to choose. That something is a hash
    of the frozen salt, so the choice is arbitrary but fixed rather than an
    accident of dictionary order.
    """
    train, header, uids, targets, weights = _population(n_profiles=4, studies_per_profile=6)
    profiles = build_study_profiles(train, header)
    # Every study identical, so no profile is better balanced than any other.
    targets = np.ones((len(uids), len(TARGETS)), dtype=float)
    weights = np.ones_like(targets)

    salts = {
        salt: choose_holdout_profiles(
            profiles, targets, weights, uids, holdout_fraction=0.25, salt=salt
        )
        for salt in ("salt-a", "salt-b", "salt-c", "salt-d")
    }
    assert len({tuple(chosen) for chosen in salts.values()}) > 1, (
        "with every choice equally balanced, the salt should be what decides"
    )


def test_the_split_does_not_depend_on_the_order_rows_arrive_in(population):
    """A validation surface that shuffling the input can change is not frozen."""
    train, header, uids, targets, weights = population
    profiles = build_study_profiles(train, header)
    shuffled = profiles.sample(frac=1.0, random_state=7).reset_index(drop=True)
    assert choose_holdout_profiles(profiles, targets, weights, uids) == (
        choose_holdout_profiles(shuffled, targets, weights, uids)
    )


def test_the_holdout_is_roughly_the_size_asked_for(population):
    train, header, uids, targets, weights = population
    profiles = build_study_profiles(train, header)
    for fraction in (0.2, 0.35, 0.5):
        chosen = choose_holdout_profiles(
            profiles, targets, weights, uids, holdout_fraction=fraction
        )
        summary = summarise_split(profiles, chosen, targets, weights, uids)
        # Whole groups only, so the size lands near the request, not exactly on it.
        assert abs(summary["holdout_study_fraction"] - fraction) < 0.2


def test_taking_every_profile_is_refused(population):
    train, header, uids, targets, weights = population
    profiles = build_study_profiles(train, header)
    with pytest.raises(ValueError, match="too large"):
        choose_holdout_profiles(profiles, targets, weights, uids, holdout_fraction=0.99)


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_a_nonsense_fraction_is_refused(population, fraction):
    train, header, uids, targets, weights = population
    profiles = build_study_profiles(train, header)
    with pytest.raises(ValueError, match="between 0 and 1"):
        choose_holdout_profiles(profiles, targets, weights, uids, holdout_fraction=fraction)


# --- what the split can and cannot measure --------------------------------


def test_a_target_with_no_positives_in_the_holdout_is_named_not_averaged_in():
    """An AUC needs both classes. A silent NaN would quietly distort the macro."""
    train, header, uids, targets, weights = _population()
    profiles = build_study_profiles(train, header)
    chosen = choose_holdout_profiles(profiles, targets, weights, uids)

    held = set(profiles.loc[profiles["scanner_profile"].isin(chosen), "StudyInstanceUID"])
    positions = [index for index, uid in enumerate(uids) if uid in held]
    targets = targets.copy()
    targets[positions, 0] = 0.0  # no positive case anywhere in the holdout

    summary = summarise_split(profiles, chosen, targets, weights, uids)
    assert TARGETS[0] in summary["unmeasurable_targets"]
    assert TARGETS[0] not in summary["measurable_targets"]


def test_a_target_present_on_both_classes_is_measurable(population):
    train, header, uids, targets, weights = population
    profiles = build_study_profiles(train, header)
    chosen = choose_holdout_profiles(profiles, targets, weights, uids)
    summary = summarise_split(profiles, chosen, targets, weights, uids)
    assert summary["measurable_targets"], "the balanced split should score something"


def test_cells_with_no_weight_are_not_counted(population):
    """Most of the label surface is blank; blank is not a negative."""
    train, header, uids, targets, weights = population
    profiles = build_study_profiles(train, header)
    chosen = choose_holdout_profiles(profiles, targets, weights, uids)

    full = summarise_split(profiles, chosen, targets, weights, uids)
    blanked = summarise_split(profiles, chosen, targets, np.zeros_like(weights), uids)
    assert full["holdout_unseen_scanners"]["usable_cells"] > 0
    assert blanked["holdout_unseen_scanners"]["usable_cells"] == 0
    assert set(blanked["unmeasurable_targets"]) == set(TARGETS)


def test_one_dominant_scanner_is_reported_rather_than_hidden():
    """If one profile holds most of the data, no grouped split can separate it."""
    train, header, uids, targets, weights = _population(n_profiles=2, studies_per_profile=10)
    header = header.copy()
    # Push almost everything onto a single machine.
    keep_small = header["StudyInstanceUID"].isin({"study-0-0", "study-0-1"})
    header.loc[~keep_small, "manufacturer"] = "SIEMENS"
    header.loc[~keep_small, "manufacturer_model"] = "OneBigScanner"
    header.loc[~keep_small, "magnetic_field_strength_t"] = 3.0

    profiles = build_study_profiles(train, header)
    summary = summarise_split(profiles, ["Siemens|Model0|3.0T"], targets, weights, uids)
    assert summary["largest_profile_share"] > CONCENTRATION_WARNING_SHARE
    assert summary["concentration_warning"] is True


# --- what gets written ----------------------------------------------------


def test_the_frozen_record_says_what_it_is_and_is_hashed(tmp_path, population):
    train, header, uids, targets, weights = population
    profiles = build_study_profiles(train, header)
    chosen = choose_holdout_profiles(profiles, targets, weights, uids)
    rows = profiles.copy()
    rows["holdout"] = rows["scanner_profile"].isin(set(chosen))

    payload = {
        "version": "test",
        "salt": DOMAIN_SPLIT_SALT,
        "holdout_scanner_profiles": chosen,
        "summary": summarise_split(profiles, chosen, targets, weights, uids),
    }
    path = write_domain_split(payload, rows, out_root=tmp_path)

    written = json.loads(path.read_text())
    assert written["holdout_scanner_profiles"] == chosen

    recorded = (tmp_path / "domain_split.sha256").read_text().strip()
    import hashlib

    assert recorded == hashlib.sha256(path.read_bytes()).hexdigest()

    table = pd.read_csv(tmp_path / "domain_split_by_study.csv")
    assert set(table.columns) >= {"StudyInstanceUID", "scanner_profile", "holdout"}
    assert len(table) == len(profiles)
    assert table["StudyInstanceUID"].is_monotonic_increasing


# --- the comparator that makes the holdout readable ------------------------


def _three_way(population, holdout_fraction=0.25):
    """Build the full three-way split the way build_domain_split does."""
    train, header, uids, targets, weights = population
    profiles = build_study_profiles(train, header)
    chosen = choose_holdout_profiles(
        profiles, targets, weights, uids, holdout_fraction=holdout_fraction
    )
    rows = profiles.copy()
    rows["holdout"] = rows["scanner_profile"].isin(set(chosen))
    seen = choose_seen_scanner_validation(
        rows.loc[~rows["holdout"]], int(rows["holdout"].sum())
    )
    rows["split"] = np.where(
        rows["holdout"],
        "holdout_unseen_scanners",
        np.where(rows["StudyInstanceUID"].isin(seen), "validation_seen_scanners", "train"),
    )
    return rows, chosen, seen, uids, targets, weights, profiles


def test_the_comparator_is_drawn_from_scanners_training_keeps(population):
    """Otherwise it is a second unseen holdout, not a comparison."""
    rows, _chosen, seen, *_ = _three_way(population)
    verify_domain_split(rows)

    trained = set(rows.loc[rows["split"].eq("train"), "scanner_profile"])
    comparator = set(rows.loc[rows["split"].eq("validation_seen_scanners"), "scanner_profile"])
    assert comparator, "there must be something to compare against"
    assert comparator.issubset(trained)


def test_a_comparator_from_an_unseen_scanner_is_caught(population):
    rows, *_ = _three_way(population)
    # Move every study of one training profile into the comparator, so that
    # profile no longer appears in training at all.
    victim = rows.loc[rows["split"].eq("train"), "scanner_profile"].iloc[0]
    rows.loc[rows["scanner_profile"].eq(victim), "split"] = "validation_seen_scanners"
    with pytest.raises(ValueError, match="not 'seen'"):
        verify_domain_split(rows)


def test_an_unknown_split_label_is_refused(population):
    rows, *_ = _three_way(population)
    rows.loc[rows.index[0], "split"] = "somewhere_else"
    with pytest.raises(ValueError, match="unknown split labels"):
        verify_domain_split(rows)


def test_no_study_lands_in_two_places(population):
    rows, _chosen, seen, *_ = _three_way(population)
    counts = rows["split"].value_counts()
    assert counts.sum() == len(rows)
    assert set(counts.index) == {"train", "validation_seen_scanners", "holdout_unseen_scanners"}
    assert seen.isdisjoint(
        set(rows.loc[rows["split"].eq("holdout_unseen_scanners"), "StudyInstanceUID"])
    )


def test_the_comparator_is_about_the_size_of_the_holdout(population):
    """Similar sizes mean similar noise, so the gap is not a sample artefact."""
    rows, *_ = _three_way(population)
    held = int(rows["split"].eq("holdout_unseen_scanners").sum())
    seen = int(rows["split"].eq("validation_seen_scanners").sum())
    assert seen == held


def test_the_comparator_is_reproducible(population):
    train, header, uids, targets, weights = population
    profiles = build_study_profiles(train, header)
    chosen = choose_holdout_profiles(profiles, targets, weights, uids)
    candidates = profiles.loc[~profiles["scanner_profile"].isin(set(chosen))]
    first = choose_seen_scanner_validation(candidates, 5)
    second = choose_seen_scanner_validation(candidates, 5)
    assert first == second
    assert len(first) == 5


def test_a_comparator_that_would_eat_the_training_side_is_refused(population):
    train, header, uids, targets, weights = population
    profiles = build_study_profiles(train, header)
    with pytest.raises(ValueError, match="consume the whole training side"):
        choose_seen_scanner_validation(profiles, len(profiles))
    with pytest.raises(ValueError, match="at least one study"):
        choose_seen_scanner_validation(profiles, 0)


def test_the_gap_is_only_claimed_for_targets_both_sides_can_score(population):
    rows, chosen, seen, uids, targets, weights, profiles = _three_way(population)
    targets = targets.copy()
    positions = [index for index, uid in enumerate(uids) if uid in seen]
    targets[positions, 0] = 0.0  # the comparator cannot score this target

    summary = summarise_split(
        profiles, chosen, targets, weights, uids, seen_validation_uids=seen
    )
    assert TARGETS[0] not in summary["comparable_targets"]
    assert set(summary["comparable_targets"]).issubset(set(summary["measurable_targets"]))


def test_with_no_comparator_no_gap_is_claimed(population):
    _rows, chosen, _seen, uids, targets, weights, profiles = _three_way(population)
    summary = summarise_split(profiles, chosen, targets, weights, uids)
    assert summary["comparable_targets"] == []
    assert summary["validation_seen_scanners"]["studies"] == 0


# --- end to end -----------------------------------------------------------


def test_build_domain_split_produces_a_usable_frozen_record(tmp_path, monkeypatch):
    """The whole entry point, with the two heavy loaders stood in for."""
    import rsna_knee.domain_shift_split as module
    from rsna_knee.domain_shift_split import build_domain_split

    train, header, uids, targets, weights = _population()
    header_csv = tmp_path / "header_by_series.csv"
    header.to_csv(header_csv, index=False)

    data_root = tmp_path / "data"
    data_root.mkdir()
    (data_root / "train.csv").write_text("StudyInstanceUID\n" + "\n".join(uids))

    monkeypatch.setattr(module, "load_train_csv", lambda path: train)
    monkeypatch.setitem(
        __import__("sys").modules,
        "rsna_knee.phase9_supervision",
        type(
            "Stub",
            (),
            {
                "load_fill_merged_export": staticmethod(lambda root: (None, None, None)),
                "prepare_all_report_only_supervision": staticmethod(
                    lambda train_df, frame: (uids, targets, weights, {})
                ),
            },
        ),
    )

    payload, rows = build_domain_split(
        data_root=data_root,
        header_csv=header_csv,
        labels_root=tmp_path / "labels",
        holdout_fraction=0.25,
    )

    assert payload["status"].startswith("frozen_before")
    assert payload["holdout_scanner_profiles"]
    assert "not a true centre holdout" in payload["group_key_is_a_scanner_proxy_not_a_site"]

    counts = rows["split"].value_counts()
    assert set(counts.index) == {"train", "validation_seen_scanners", "holdout_unseen_scanners"}
    assert counts["train"] > 0

    summary = payload["summary"]
    assert summary["holdout_unseen_scanners"]["studies"] > 0
    assert (
        summary["validation_seen_scanners"]["studies"]
        == summary["holdout_unseen_scanners"]["studies"]
    )
    # Nothing that trains is also scored.
    trained = set(rows.loc[rows["split"].eq("train"), "StudyInstanceUID"])
    scored = set(rows.loc[rows["split"].ne("train"), "StudyInstanceUID"])
    assert trained.isdisjoint(scored)

    path = write_domain_split(payload, rows, out_root=tmp_path / "out")
    assert json.loads(path.read_text())["version"] == payload["version"]
