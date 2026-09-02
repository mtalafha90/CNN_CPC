"""Fill the cells the regex parser leaves silent, keeping every call it makes.

B23 replaces the B6 parser outright and was refused: it answers more cells and
ranks expert truth far better (state-only macro AUC 0.8125 against 0.7025,
paired 95% CI [+0.0681, +0.1532]), but its specificity is 0.5678 against B6's
0.6061, and the predeclared gate required specificity to improve as well.

B24X then measured which half of the change carries the benefit. Preserving
every B6 decision and using the LLM only where B6 is silent captured **103.3%**
of the full replacement's gain, and replacing B6's own calls added nothing
(B23 - Density, 95% CI [-0.0100, +0.0035]). So the value is coverage, not
correction.

That is what this builds. Every committed B6 cell survives unchanged, so the
frozen parser's specificity is carried through intact and the clause that
closed B23 does not apply: no B6 call is overridden, and the cells being added
are ones B6 declined to answer at all.

What it does not settle
-----------------------

The added cells carry the LLM's own error rate, and nobody has measured that on
this corpus outside the 58-study pilot. B26 is the standing warning: filling
Synovitis negatives passed a 100% manual label audit and still made expert
Synovitis AUC fall, because what looked like a missing-label defect was
reporting habit. This merge is therefore a supervision *candidate*, and the
hidden test is the only ruler that can judge it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .constants import TARGETS

MERGE_VERSION = "b6_preserved_plus_b23_fill_v1"

# A cell is committed when the labeller gave a definite answer it stands behind.
COMMITTED_STATES = ("positive", "negated")
MIN_CONFIDENCE = 0.75

# Which states the filler is allowed to supply. Measured on the 58 expert
# studies, the two are not remotely alike:
#
#     negated cells      137    97.8% correct
#     positive cells     305    67.2% correct
#
# and within the positives, the split is exactly the one this merge creates:
#
#     B6 also says positive    159    71.1% correct    <- a base cell, preserved
#     B6 stayed silent         145    62.8% correct    <- what filling adds
#
# So "only accept a positive the frozen parser corroborates" and "fill negated
# cells only" are the same rule stated two ways, because a filled cell is by
# definition one the base declined to answer. `NEGATED_ONLY` is that rule.
FILL_BOTH_STATES = ("positive", "negated")
FILL_NEGATED_ONLY = ("negated",)


def _committed(
    frame: pd.DataFrame,
    target: str,
    *,
    min_confidence: float,
    states: tuple[str, ...] = COMMITTED_STATES,
    min_model_confidence: float | None = None,
) -> pd.Series:
    """Cells this labeller stands behind, optionally narrowed by state or self-report.

    `min_model_confidence` reads `__model_confidence`, the labeller's own number,
    which the export records and the supervision pipeline never uses. On the 58
    expert studies it separates its own right answers from its wrong ones at
    about 0.61 AUC -- real but weak -- so it is available and off by default.
    """
    state = frame[f"{target}__state"].astype(str)
    confidence = pd.to_numeric(frame[f"{target}__confidence"], errors="coerce").fillna(0.0)
    keep = state.isin(states) & confidence.ge(min_confidence)
    if min_model_confidence is not None:
        column = f"{target}__model_confidence"
        if column not in frame.columns:
            raise ValueError(
                f"--min-model-confidence needs {column}, which training_targets.csv "
                "does not carry; point --filler at the export's structured_labels.csv"
            )
        model = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        keep = keep & model.ge(float(min_model_confidence))
    return keep


def _read_export(root: str | Path) -> tuple[pd.DataFrame, dict]:
    """Read an export directory, or a single labels CSV.

    `training_targets.csv` carries no gold rows and no `__model_confidence`.
    `structured_labels.csv` carries both.

    Gold rows are **kept** here, and excluded where it actually matters: at the
    write step, which emits `training_targets.csv` from the non-gold rows only
    and counts what it dropped rather than asserting it. Carrying them is what
    lets a merged export be measured against the 58 expert studies at all --
    without it, the only teacher nobody can audit is the one being trained on.
    """
    root = Path(root)
    if root.is_file():
        targets_path, audit_path = root, root.parent / "audit.json"
    else:
        targets_path, audit_path = root / "training_targets.csv", root / "audit.json"
    if not targets_path.is_file():
        raise FileNotFoundError(f"missing export artifact: {targets_path}")

    frame = pd.read_csv(targets_path)
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    if "is_gold" in frame.columns and frame["is_gold"].astype(bool).any():
        print(
            f"[merge] {targets_path.name}: carrying "
            f"{int(frame['is_gold'].astype(bool).sum())} gold studies for audit; "
            "they are excluded from training_targets.csv at the write step",
            flush=True,
        )
    if frame["StudyInstanceUID"].duplicated().any():
        raise ValueError(f"{targets_path} lists a study more than once")

    missing = [
        column
        for target in TARGETS
        for column in (target, f"{target}__confidence", f"{target}__state")
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"{targets_path} is missing columns: {', '.join(missing[:6])}")

    audit = (
        json.loads(audit_path.read_text(encoding="utf-8"))
        if audit_path.is_file()
        else {"note": f"no audit.json beside {targets_path.name}"}
    )
    return frame, audit


def merge_fill_only(
    base: pd.DataFrame,
    filler: pd.DataFrame,
    *,
    min_confidence: float = MIN_CONFIDENCE,
    exclude_targets: tuple[str, ...] = (),
    fill_states: tuple[str, ...] = FILL_BOTH_STATES,
    min_model_confidence: float | None = None,
    only_silent_studies: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Keep every committed base cell; take a filler cell only where base is silent.

    The base's study order is preserved exactly, and a study the filler never
    saw simply contributes nothing rather than dropping the study.

    `only_silent_studies` narrows that further: a filler cell is taken only if
    the base says *nothing whatever* about that study. It exists for the
    translation rescue, where the two cases are not the same decision. Phase 6
    froze "no translated cell may enter a B6-active study"; a study the LLM
    filler has since answered was not B6-active, so whether that clause reaches
    it is a judgement rather than a lookup. This flag is how a run declines to
    make that judgement, and it turns 2,725 available cells into 678.

    `exclude_targets` leaves a finding entirely on the base labeller. That
    exists for Synovitis: B26 filled its scarce negatives, passed a 100% manual
    label audit, and expert Synovitis AUC still fell, because the scarcity was
    reporting habit -- synovitis is stated when present and rarely negated --
    rather than a missing-label defect. A filler that supplies many Synovitis
    negatives is reproducing that experiment, not avoiding it.
    """
    unknown = [t for t in exclude_targets if t not in TARGETS]
    if unknown:
        raise ValueError(f"unknown target(s) to exclude: {', '.join(unknown)}")
    unknown_states = [s for s in fill_states if s not in COMMITTED_STATES]
    if unknown_states or not fill_states:
        raise ValueError(f"fill_states must be a non-empty subset of {COMMITTED_STATES}")

    merged = base.copy()
    filler_by_uid = filler.set_index("StudyInstanceUID")
    shared = merged["StudyInstanceUID"].isin(filler_by_uid.index)

    aligned = (
        filler_by_uid.reindex(merged["StudyInstanceUID"]).reset_index(drop=True)
    )

    # Study-level silence, measured on the base before a single cell is written,
    # so the first target filled cannot make the later ones ineligible.
    base_cells_per_study = sum(
        _committed(base, target, min_confidence=min_confidence).astype(int)
        for target in TARGETS
    )
    eligible_study = (
        base_cells_per_study.eq(0).reset_index(drop=True)
        if only_silent_studies
        else pd.Series(True, index=range(len(merged)))
    )

    per_target: dict[str, dict] = {}
    for target in TARGETS:
        base_committed = _committed(merged, target, min_confidence=min_confidence)
        if target in exclude_targets:
            per_target[target] = {
                "base_committed": int(base_committed.sum()),
                "filled": 0,
                "filled_positive": 0,
                "filled_negative": 0,
                "base_overridden": 0,
                "final_committed": int(base_committed.sum()),
                "excluded_from_fill": True,
            }
            continue

        filler_committed = (
            _committed(
                aligned,
                target,
                min_confidence=min_confidence,
                states=tuple(fill_states),
                min_model_confidence=min_model_confidence,
            )
            & shared.reset_index(drop=True)
        )
        fillable = ~base_committed & filler_committed & eligible_study

        carried = [target, f"{target}__confidence", f"{target}__state"]
        # The labeller's own confidence travels with the cell it belongs to, so a
        # merged export can be filtered on it later. Base cells keep NaN: the
        # regex parser has no self-report and pretending otherwise would invent
        # one. Base cells are never filled, so no filter ever consults it there.
        model_column = f"{target}__model_confidence"
        if model_column in aligned.columns:
            if model_column not in merged.columns:
                merged[model_column] = float("nan")
            carried.append(model_column)
        for column in carried:
            merged.loc[fillable, column] = aligned.loc[fillable, column].to_numpy()

        filled_states = aligned.loc[fillable, f"{target}__state"].astype(str)
        per_target[target] = {
            "base_committed": int(base_committed.sum()),
            "filled": int(fillable.sum()),
            "filled_positive": int((filled_states == "positive").sum()),
            "filled_negative": int((filled_states == "negated").sum()),
            "base_overridden": 0,
            "final_committed": int(base_committed.sum() + fillable.sum()),
        }

    possible = len(merged) * len(TARGETS)
    base_total = sum(item["base_committed"] for item in per_target.values())
    filled_total = sum(item["filled"] for item in per_target.values())
    audit = {
        "merge_version": MERGE_VERSION,
        "rule": "every committed base cell is preserved; the filler is used only where the base is silent",
        "min_confidence": float(min_confidence),
        "fill_states": list(fill_states),
        "min_model_confidence": (
            None if min_model_confidence is None else float(min_model_confidence)
        ),
        "only_silent_studies": bool(only_silent_studies),
        "studies_wholly_silent_in_base": int(base_cells_per_study.eq(0).sum()),
        "studies": int(len(merged)),
        "studies_seen_by_filler": int(shared.sum()),
        "possible_cells": int(possible),
        "base_committed_cells": base_total,
        "filled_cells": filled_total,
        "final_committed_cells": base_total + filled_total,
        "base_cells_overridden": 0,
        "base_coverage": base_total / possible if possible else 0.0,
        "final_coverage": (base_total + filled_total) / possible if possible else 0.0,
        "coverage_increase": (filled_total / base_total) if base_total else 0.0,
        "excluded_targets": list(exclude_targets),
        "targets": per_target,
    }
    return merged, audit


def write_merged_export(
    out_root: str | Path,
    merged: pd.DataFrame,
    audit: dict,
    *,
    base_audit: dict,
    filler_audit: dict,
) -> Path:
    """Write the files the training pipeline expects, plus one it can be audited by.

    `training_targets.csv` is what trains: non-gold rows only, and the count of
    what was withheld is computed here rather than declared, because
    `load_fill_merged_export` refuses any export that cannot certify zero.

    `structured_labels.csv` is written whenever the merge carried gold rows. It
    is the file `report_label_gold_audit` reads, so a merged teacher can be
    measured against the 58 expert studies instead of being inferred from its
    two halves.
    """
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)

    columns = ["StudyInstanceUID"]
    for target in TARGETS:
        columns.extend([target, f"{target}__confidence", f"{target}__state"])

    has_gold = "is_gold" in merged.columns
    if has_gold:
        structured_columns = [
            column
            for column in (
                ["StudyInstanceUID", "is_gold"]
                + [
                    name
                    for target in TARGETS
                    for name in (
                        target,
                        f"{target}__confidence",
                        f"{target}__model_confidence",
                        f"{target}__state",
                    )
                ]
            )
            if column in merged.columns
        ]
        merged[structured_columns].to_csv(out / "structured_labels.csv", index=False)
        training = merged.loc[~merged["is_gold"].astype(bool)]
    else:
        training = merged

    training[columns].to_csv(out / "training_targets.csv", index=False)

    withheld = int(len(merged) - len(training))
    remaining_gold = (
        int(training["is_gold"].astype(bool).sum()) if has_gold else 0
    )
    if remaining_gold:
        raise RuntimeError(
            f"{remaining_gold} gold studies reached training_targets.csv; refusing to write"
        )

    full_audit = {
        **audit,
        "gold_rows_in_training_targets": remaining_gold,
        "gold_rows_withheld_from_training": withheld,
        "structured_labels_written": bool(has_gold),
        # Both sources are recorded whole: this export is only as reproducible
        # as the labellers behind it, and the LLM's provenance lives in its own
        # audit rather than here.
        "base_audit": base_audit,
        "filler_audit": filler_audit,
    }
    (out / "audit.json").write_text(json.dumps(full_audit, indent=2), encoding="utf-8")

    policy = {
        "experiment": "b23_fill_merge",
        "version": MERGE_VERSION,
        "purpose": (
            "B6 decisions preserved exactly, LLM used only where B6 is silent; "
            "the B24X-Density formulation, which captured 103.3% of full "
            "replacement's gain without overriding a single B6 call"
        ),
        "base_version": str(base_audit.get("b6_version", base_audit.get("version", ""))),
        "filler_version": str(filler_audit.get("b23_version", "")),
        "filler_provenance": filler_audit.get("provenance"),
        "min_confidence_for_usable_cell": float(audit["min_confidence"]),
        "base_cells_overridden": 0,
    }
    (out / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    return out


def _report(audit: dict) -> None:
    print(f"\n{audit['studies']} studies, {audit['possible_cells']:,} possible cells\n")
    if audit.get("only_silent_studies"):
        print(
            f"  filling only the {audit['studies_wholly_silent_in_base']:,} studies "
            "the base says nothing about\n"
        )
    print(f"  parser committed   {audit['base_committed_cells']:>7,}"
          f"   {audit['base_coverage'] * 100:5.1f}% coverage")
    print(f"  filled by the LLM  {audit['filled_cells']:>7,}"
          f"   +{audit['coverage_increase'] * 100:.1f}% more supervision")
    print(f"  parser overridden  {audit['base_cells_overridden']:>7,}   (always zero, by rule)")
    print(f"  final              {audit['final_committed_cells']:>7,}"
          f"   {audit['final_coverage'] * 100:5.1f}% coverage\n")

    print(f"  {'target':<18}{'parser':>8}{'filled':>8}{'+pos':>7}{'+neg':>7}{'final':>8}")
    for target, item in audit["targets"].items():
        print(
            f"  {target:<18}{item['base_committed']:>8}{item['filled']:>8}"
            f"{item['filled_positive']:>7}{item['filled_negative']:>7}"
            f"{item['final_committed']:>8}"
        )

    # B26 filled Synovitis negatives at 100% manual label accuracy and expert
    # Synovitis AUC still fell, because the scarcity was reporting habit rather
    # than a missing-label defect. Anyone reading this table should see that.
    synovitis = audit["targets"].get("Synovitis")
    if synovitis and synovitis["filled_negative"] > synovitis["base_committed"]:
        print(
            "\n  Note: this adds more Synovitis negatives than the parser found "
            "cells.\n  B26 did that deliberately, passed a 100% manual label "
            "audit, and expert\n  Synovitis AUC fell anyway. Watch that target."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preserve every parser call and fill only its silent cells"
    )
    parser.add_argument("--base", required=True, help="the frozen parser export to preserve")
    parser.add_argument("--filler", required=True, help="the LLM export used for silent cells")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--min-confidence", type=float, default=MIN_CONFIDENCE)
    parser.add_argument(
        "--fill-states",
        choices=("both", "negated"),
        default="both",
        help=(
            "which states the filler may supply. 'negated' keeps every base call "
            "and adds only negatives, which is the same rule as refusing any "
            "positive the base parser does not corroborate"
        ),
    )
    parser.add_argument(
        "--min-model-confidence",
        type=float,
        default=None,
        help=(
            "also require the labeller's own confidence. Needs a filler export "
            "carrying __model_confidence, i.e. structured_labels.csv"
        ),
    )
    parser.add_argument(
        "--exclude-target",
        action="append",
        default=[],
        help=(
            # argparse %-formats help text, so a literal percent must be doubled
            # or --help raises instead of printing.
            "leave this finding entirely on the base labeller; repeat the flag. "
            "Synovitis is the known case: B26 filled its negatives at 100%% manual "
            "label accuracy and expert AUC fell anyway"
        ),
    )
    parser.add_argument(
        "--only-silent-studies",
        action="store_true",
        help=(
            "fill only studies the base says nothing whatever about. For the "
            "translation rescue: filling a wholly silent study is the frozen "
            "Phase-8 policy, filling one the LLM has already reached is a new "
            "and unmeasured one"
        ),
    )
    args = parser.parse_args()

    base, base_audit = _read_export(args.base)
    filler, filler_audit = _read_export(args.filler)

    merged, audit = merge_fill_only(
        base,
        filler,
        min_confidence=args.min_confidence,
        exclude_targets=tuple(args.exclude_target),
        fill_states=(
            FILL_NEGATED_ONLY if args.fill_states == "negated" else FILL_BOTH_STATES
        ),
        min_model_confidence=args.min_model_confidence,
        only_silent_studies=args.only_silent_studies,
    )
    _report(audit)

    out = write_merged_export(
        args.out_root, merged, audit, base_audit=base_audit, filler_audit=filler_audit
    )
    print(out)


if __name__ == "__main__":
    main()
