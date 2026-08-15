"""Plan the B23/B24 pipeline against a hard wall-clock budget.

The repository records no wall time before this module existed, so any estimate
of what fits in a nine-hour window would have been a guess. This planner works
from **measured** rates only: seconds per report taken from the extraction
cache, and seconds per study-epoch taken from a completed training history. If
a rate has not been measured it says so and refuses to invent one.

## The structural finding this exists to surface

Labelling and training compete for the same GPU and the same window. At full
scale they almost certainly do not both fit, so the pipeline has to be split
across sessions — which is exactly why the extraction cache is resumable and
why a partially completed run can be salvaged into a pilot.

The planner's job is to say which of these fits:

```text
A  pilot end-to-end     label a subset, audit, train both arms, evaluate
B  labelling only       spend the window entirely on reports
C  training only        labels already complete; train both arms and evaluate
```
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .b23_llm_labels import ExtractionCache, observed_seconds_per_report

# RuntimeBudget refuses anything >= 9 hours, so a session is always shorter.
DEFAULT_SESSION_HOURS = 8.5
DEFAULT_RESERVE_MINUTES = 10.0

# B24 trains two arms for two epochs each.
B24_ARMS = 2
B24_EPOCHS_PER_ARM = 2
# Inference for the cross-labeller evaluation: two arms over two weak surfaces,
# each with 3-offset TTA. Expressed as a multiple of one training epoch's
# per-study cost, since inference has no backward pass but does run TTA.
EVAL_COST_FACTOR = 0.6


@dataclass(frozen=True)
class MeasuredRates:
    """Rates taken from artifacts, never assumed."""

    seconds_per_report: float | None
    seconds_per_study_epoch: float | None
    report_source: str
    training_source: str

    def missing(self) -> list[str]:
        gaps = []
        if self.seconds_per_report is None:
            gaps.append(
                "seconds_per_report -- run `rsna-knee-b23 --limit 20` and re-read "
                "the cache, or pass --seconds-per-report from a timed run"
            )
        if self.seconds_per_study_epoch is None:
            gaps.append(
                "seconds_per_study_epoch -- complete one B24 arm, or pass "
                "--seconds-per-study-epoch from a timed run"
            )
        return gaps


def measure_rates(
    *,
    cache_path: str | Path | None = None,
    training_history: str | Path | None = None,
    seconds_per_report: float | None = None,
    seconds_per_study_epoch: float | None = None,
) -> MeasuredRates:
    """Read what has actually been measured; override only when given a number."""
    report_rate, report_source = seconds_per_report, "supplied"
    if report_rate is None and cache_path is not None and Path(cache_path).is_file():
        report_rate = observed_seconds_per_report(ExtractionCache(cache_path))
        report_source = f"measured from {cache_path}"
    if report_rate is None:
        report_source = "NOT MEASURED"

    train_rate, train_source = seconds_per_study_epoch, "supplied"
    if train_rate is None and training_history is not None and Path(training_history).is_file():
        payload = json.loads(Path(training_history).read_text(encoding="utf-8"))
        per_study = [
            float(row["seconds_per_study"])
            for row in payload.get("history", [])
            if isinstance(row.get("seconds_per_study"), (int, float))
        ]
        if per_study:
            train_rate = float(np.median(np.asarray(per_study)))
            train_source = f"measured from {training_history}"
    if train_rate is None:
        train_source = "NOT MEASURED"

    return MeasuredRates(
        seconds_per_report=report_rate,
        seconds_per_study_epoch=train_rate,
        report_source=report_source,
        training_source=train_source,
    )


def labelling_hours(n_reports: int, rates: MeasuredRates) -> float | None:
    if rates.seconds_per_report is None:
        return None
    return float(n_reports) * rates.seconds_per_report / 3600.0


def training_hours(n_studies: int, rates: MeasuredRates) -> float | None:
    """Both B24 arms, plus the cross-labeller evaluation."""
    if rates.seconds_per_study_epoch is None:
        return None
    epochs = B24_ARMS * B24_EPOCHS_PER_ARM
    train = float(n_studies) * rates.seconds_per_study_epoch * epochs
    evaluate = float(n_studies) * rates.seconds_per_study_epoch * EVAL_COST_FACTOR
    return (train + evaluate) / 3600.0


def max_reports_in(hours: float, rates: MeasuredRates) -> int | None:
    if rates.seconds_per_report is None:
        return None
    return int(hours * 3600.0 / rates.seconds_per_report)


def max_studies_in(hours: float, rates: MeasuredRates) -> int | None:
    if rates.seconds_per_study_epoch is None:
        return None
    per_study = rates.seconds_per_study_epoch * (
        B24_ARMS * B24_EPOCHS_PER_ARM + EVAL_COST_FACTOR
    )
    return int(hours * 3600.0 / per_study)


def plan(
    rates: MeasuredRates,
    *,
    session_hours: float = DEFAULT_SESSION_HOURS,
    reserve_minutes: float = DEFAULT_RESERVE_MINUTES,
    cached_reports: int = 0,
    total_reports: int = 4349,
    pilot_studies: int | None = None,
) -> dict:
    """Work out which of the three shapes fits one session."""
    if session_hours >= 9.0:
        raise ValueError("a session must be strictly under 9 hours")
    usable = session_hours - reserve_minutes / 60.0
    remaining_reports = max(0, int(total_reports) - int(cached_reports))
    pilot = int(pilot_studies) if pilot_studies is not None else int(cached_reports)

    options = {}

    # A: pilot end-to-end, using labels that already exist.
    pilot_train = training_hours(pilot, rates)
    options["pilot_end_to_end"] = {
        "description": f"train both B24 arms on the {pilot} already-labelled studies",
        "labelling_hours": 0.0,
        "training_hours": pilot_train,
        "total_hours": pilot_train,
        "fits": None if pilot_train is None else bool(pilot_train <= usable),
        "note": "no new labelling; uses the cache as a declared pilot",
    }

    # B: spend the window labelling.
    reports_possible = max_reports_in(usable, rates)
    options["labelling_only"] = {
        "description": f"label as much of the remaining {remaining_reports} as fits",
        "reports_this_session": (
            None if reports_possible is None else min(reports_possible, remaining_reports)
        ),
        "sessions_to_finish": (
            None
            if reports_possible is None or reports_possible == 0
            else int(np.ceil(remaining_reports / reports_possible))
        ),
        "total_hours": usable,
        "fits": True,
        "note": "resumable; the cache means nothing is lost between sessions",
    }

    # C: full-scale training once labelling is complete.
    full_train = training_hours(int(total_reports), rates)
    options["full_scale_training"] = {
        "description": f"train both arms on all {total_reports} labelled studies",
        "training_hours": full_train,
        "total_hours": full_train,
        "fits": None if full_train is None else bool(full_train <= usable),
        "max_studies_that_fit": max_studies_in(usable, rates),
        "note": "requires labelling to be finished in earlier sessions",
    }

    recommended = _recommend(options, rates, cached_reports, remaining_reports)
    return {
        "session_hours": session_hours,
        "usable_hours": usable,
        "rates": {
            "seconds_per_report": rates.seconds_per_report,
            "seconds_per_report_source": rates.report_source,
            "seconds_per_study_epoch": rates.seconds_per_study_epoch,
            "seconds_per_study_epoch_source": rates.training_source,
        },
        "unmeasured": rates.missing(),
        "cached_reports": int(cached_reports),
        "remaining_reports": remaining_reports,
        "options": options,
        "recommended": recommended,
    }


def _recommend(options, rates, cached_reports, remaining_reports) -> str:
    if rates.missing():
        return (
            "MEASURE FIRST. Nothing can be planned honestly until the missing "
            "rates above exist."
        )
    if options["pilot_end_to_end"]["fits"] and cached_reports >= 200:
        return (
            "pilot_end_to_end -- it validates the whole pipeline inside one "
            "window using labels you have already paid for, and it is the only "
            "option that produces a result this session."
        )
    if remaining_reports > 0:
        return (
            "labelling_only -- there are not yet enough labels for a pilot worth "
            "training on; spend the window on reports, which is resumable."
        )
    return "full_scale_training -- labelling is complete."


def format_plan(payload: dict) -> str:
    lines = [
        f"B23/B24 budget plan ({payload['session_hours']}h session, "
        f"{payload['usable_hours']:.2f}h usable)",
        "",
        "  measured rates",
    ]
    rates = payload["rates"]
    spr = rates["seconds_per_report"]
    spe = rates["seconds_per_study_epoch"]
    lines.append(
        f"    seconds/report        "
        f"{'NOT MEASURED' if spr is None else f'{spr:.1f}'}"
        f"   ({rates['seconds_per_report_source']})"
    )
    lines.append(
        f"    seconds/study/epoch   "
        f"{'NOT MEASURED' if spe is None else f'{spe:.3f}'}"
        f"   ({rates['seconds_per_study_epoch_source']})"
    )
    if payload["unmeasured"]:
        lines.append("")
        lines.append("  MISSING MEASUREMENTS -- these cannot be guessed:")
        for gap in payload["unmeasured"]:
            lines.append(f"    - {gap}")

    lines.extend(["", f"  cached reports {payload['cached_reports']}"
                  f" | remaining {payload['remaining_reports']}", ""])
    for name, opt in payload["options"].items():
        fits = opt.get("fits")
        mark = "?" if fits is None else ("fits" if fits else "DOES NOT FIT")
        total = opt.get("total_hours")
        hours = "unknown" if total is None else f"{total:.2f}h"
        lines.append(f"  [{mark:>12}]  {name}  ({hours})")
        lines.append(f"                  {opt['description']}")
        if opt.get("sessions_to_finish"):
            lines.append(f"                  sessions to finish: {opt['sessions_to_finish']}")
        if opt.get("max_studies_that_fit") is not None:
            lines.append(
                f"                  studies that fit one session: {opt['max_studies_that_fit']}"
            )
        lines.append(f"                  {opt['note']}")
        lines.append("")
    lines.append(f"  RECOMMENDED: {payload['recommended']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan the B23/B24 pipeline against a wall-clock budget"
    )
    parser.add_argument("--session-hours", type=float, default=DEFAULT_SESSION_HOURS)
    parser.add_argument("--reserve-minutes", type=float, default=DEFAULT_RESERVE_MINUTES)
    parser.add_argument("--cache", default=None, help="extraction_cache.jsonl, to measure labelling")
    parser.add_argument("--training-history", default=None, help="a B24 history.json")
    parser.add_argument("--seconds-per-report", type=float, default=None)
    parser.add_argument("--seconds-per-study-epoch", type=float, default=None)
    parser.add_argument("--cached-reports", type=int, default=0)
    parser.add_argument("--total-reports", type=int, default=4349)
    parser.add_argument("--pilot-studies", type=int, default=None)
    args = parser.parse_args()

    rates = measure_rates(
        cache_path=args.cache,
        training_history=args.training_history,
        seconds_per_report=args.seconds_per_report,
        seconds_per_study_epoch=args.seconds_per_study_epoch,
    )
    cached = args.cached_reports
    if cached == 0 and args.cache and Path(args.cache).is_file():
        cached = len(ExtractionCache(args.cache))
    payload = plan(
        rates,
        session_hours=args.session_hours,
        reserve_minutes=args.reserve_minutes,
        cached_reports=cached,
        total_reports=args.total_reports,
        pilot_studies=args.pilot_studies,
    )
    print(format_plan(payload))


if __name__ == "__main__":  # pragma: no cover
    main()
