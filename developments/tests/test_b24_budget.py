"""Budget planning must be measured, never guessed."""
from __future__ import annotations

import json

import pytest

from rsna_knee.b24_budget import (
    B24_ARMS,
    B24_EPOCHS_PER_ARM,
    MeasuredRates,
    format_plan,
    labelling_hours,
    max_reports_in,
    max_studies_in,
    measure_rates,
    plan,
    training_hours,
)


def _rates(report=None, train=None):
    return MeasuredRates(
        seconds_per_report=report,
        seconds_per_study_epoch=train,
        report_source="test",
        training_source="test",
    )


def test_nothing_is_estimated_when_nothing_has_been_measured():
    rates = _rates()
    assert labelling_hours(4349, rates) is None
    assert training_hours(3120, rates) is None
    assert max_reports_in(8.0, rates) is None
    assert max_studies_in(8.0, rates) is None
    assert len(rates.missing()) == 2


def test_the_plan_refuses_to_recommend_without_measurements():
    payload = plan(_rates(), cached_reports=1130)
    assert "MEASURE FIRST" in payload["recommended"]
    assert len(payload["unmeasured"]) == 2
    assert "cannot be guessed" in format_plan(payload)


def test_measured_rates_are_read_from_the_cache(tmp_path):
    cache = tmp_path / "cache.jsonl"
    with cache.open("w", encoding="utf-8") as handle:
        for seconds in (40.0, 44.0, 42.0):
            handle.write(json.dumps({"cache_key": str(seconds), "seconds": seconds}) + "\n")
    rates = measure_rates(cache_path=cache)
    # Median, so one stalled request cannot skew a budget.
    assert rates.seconds_per_report == pytest.approx(42.0)
    assert "measured from" in rates.report_source


def test_a_single_stall_does_not_skew_the_rate(tmp_path):
    cache = tmp_path / "cache.jsonl"
    with cache.open("w", encoding="utf-8") as handle:
        for seconds in (40.0, 41.0, 42.0, 43.0, 6000.0):
            handle.write(json.dumps({"cache_key": str(seconds), "seconds": seconds}) + "\n")
    assert measure_rates(cache_path=cache).seconds_per_report == pytest.approx(42.0)


def test_measured_rates_are_read_from_a_training_history(tmp_path):
    history = tmp_path / "history.json"
    history.write_text(
        json.dumps(
            {"history": [{"seconds_per_study": 1.8}, {"seconds_per_study": 2.2}]}
        ),
        encoding="utf-8",
    )
    rates = measure_rates(training_history=history)
    assert rates.seconds_per_study_epoch == pytest.approx(2.0)


def test_an_explicit_rate_overrides_an_artifact(tmp_path):
    cache = tmp_path / "cache.jsonl"
    cache.write_text(json.dumps({"cache_key": "a", "seconds": 99.0}) + "\n", encoding="utf-8")
    rates = measure_rates(cache_path=cache, seconds_per_report=10.0)
    assert rates.seconds_per_report == 10.0
    assert rates.report_source == "supplied"


def test_training_cost_covers_both_arms_and_the_evaluation():
    rates = _rates(train=1.0)
    # Two arms x two epochs, plus the cross-labeller evaluation pass.
    bare_epochs = B24_ARMS * B24_EPOCHS_PER_ARM
    hours = training_hours(1000, rates)
    assert hours > 1000 * bare_epochs / 3600.0
    assert hours == pytest.approx(1000 * (bare_epochs + 0.6) / 3600.0)


def test_labelling_and_training_are_both_charged_to_the_same_window():
    rates = _rates(report=45.0, train=2.0)
    # A 4,349-report labelling run and a full-scale training run cannot both fit.
    assert labelling_hours(4349, rates) > 8.5
    assert training_hours(4349, rates) > 8.5


def test_a_pilot_that_fits_is_recommended_over_more_labelling():
    rates = _rates(report=45.0, train=0.5)
    payload = plan(rates, cached_reports=1130, session_hours=8.5)
    assert payload["options"]["pilot_end_to_end"]["fits"] is True
    assert payload["recommended"].startswith("pilot_end_to_end")


def test_labelling_is_recommended_when_too_little_is_cached():
    rates = _rates(report=45.0, train=0.5)
    payload = plan(rates, cached_reports=20, session_hours=8.5)
    assert payload["recommended"].startswith("labelling_only")


def test_the_plan_reports_how_many_sessions_labelling_needs():
    rates = _rates(report=45.0, train=0.5)
    payload = plan(rates, cached_reports=1130, total_reports=4349)
    labelling = payload["options"]["labelling_only"]
    assert labelling["sessions_to_finish"] >= 1
    assert labelling["reports_this_session"] <= payload["remaining_reports"]


def test_full_scale_training_reports_the_ceiling_that_actually_fits():
    rates = _rates(report=45.0, train=2.0)
    payload = plan(rates, cached_reports=4349, total_reports=4349)
    full = payload["options"]["full_scale_training"]
    assert full["fits"] is False
    # The planner must say how large a run WOULD fit, not merely refuse.
    assert full["max_studies_that_fit"] > 0
    assert full["max_studies_that_fit"] < 4349


def test_a_session_of_nine_hours_or_more_is_refused():
    # RuntimeBudget itself rejects >= 9h, so the planner must agree.
    with pytest.raises(ValueError, match="strictly under 9 hours"):
        plan(_rates(report=1.0, train=1.0), session_hours=9.0)


def test_the_formatted_plan_marks_what_does_not_fit():
    rates = _rates(report=45.0, train=2.0)
    text = format_plan(plan(rates, cached_reports=4349, total_reports=4349))
    assert "DOES NOT FIT" in text
    assert "RECOMMENDED" in text
