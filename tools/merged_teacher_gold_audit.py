"""Audit the fill-merged teacher that actually trained B51, against expert truth.

B51's labels came from neither export alone. Its recorded policy is
`b6_preserved_plus_b23_fill_v1`: every committed B6 cell preserved exactly, the
B23 LLM used only where B6 is silent. That merged export writes only
`training_targets.csv`, `audit.json` and `policy.json` -- and
`training_targets.csv` excludes the 58 expert studies by design. So the teacher
that trained B51 has no artefact recording what it would have said about the
only cells where expert truth exists.

It can be reconstructed, because the merge is deterministic and both source
exports keep their gold rows. This applies `merge_fill_only` -- the real
function, not a reimplementation -- to the full frames including gold, then
audits the gold rows.

A reconstruction is only worth as much as its check, so before reporting
anything this verifies that the reconstruction reproduces the recorded
`training_targets.csv` exactly on every report-only cell. If a single state or
confidence disagrees, it refuses to report the gold numbers, because then the
rule being applied is not the rule that produced B51's labels.

    python -m tools.merged_teacher_gold_audit \\
      --train-csv /path/to/train.csv \\
      --base      runs/011_.../b6_report_labels_v121 \\
      --filler    runs/033_.../b23_full \\
      --merged    runs/0XX_.../<the fill-merge export>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "developments" / "src"))

from rsna_knee.b23_fill_merge import MERGE_VERSION, merge_fill_only  # noqa: E402
from rsna_knee.constants import TARGETS  # noqa: E402
from rsna_knee.report_label_gold_audit import (  # noqa: E402
    audit_export_against_gold,
    _print_summary,
)

CELL_COLUMNS = [f"{target}__state" for target in TARGETS] + [
    f"{target}__confidence" for target in TARGETS
]


def _structured(root: Path) -> pd.DataFrame:
    path = root / "structured_labels.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; the gold rows live only in structured_labels.csv"
        )
    frame = pd.read_csv(path)
    frame["StudyInstanceUID"] = frame["StudyInstanceUID"].astype(str)
    return frame


def _verify_against_recorded(reconstructed: pd.DataFrame, merged_root: Path) -> dict:
    """The reconstruction must reproduce the real export on every shared cell."""
    recorded = pd.read_csv(merged_root / "training_targets.csv")
    recorded["StudyInstanceUID"] = recorded["StudyInstanceUID"].astype(str)

    shared = reconstructed.loc[
        reconstructed["StudyInstanceUID"].isin(set(recorded["StudyInstanceUID"]))
    ].copy()
    if len(shared) != len(recorded):
        raise RuntimeError(
            f"reconstruction covers {len(shared)} of the recorded export's "
            f"{len(recorded)} studies; the sources do not match this merge"
        )

    left = shared.set_index("StudyInstanceUID").sort_index()
    right = recorded.set_index("StudyInstanceUID").sort_index()

    disagreements = {}
    for column in CELL_COLUMNS:
        if column.endswith("__state"):
            same = left[column].astype(str).eq(right[column].astype(str))
        else:
            same = (
                pd.to_numeric(left[column], errors="coerce")
                .sub(pd.to_numeric(right[column], errors="coerce"))
                .abs()
                .le(1e-6)
            )
        if not same.all():
            disagreements[column] = int((~same).sum())

    if disagreements:
        raise RuntimeError(
            "the reconstruction does not reproduce the recorded export: "
            f"{dict(list(disagreements.items())[:5])}. The gold numbers would "
            "describe a merge that never trained anything, so they are not reported."
        )
    return {"studies_checked": len(recorded), "cells_checked": len(recorded) * len(CELL_COLUMNS)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct and audit the fill-merged teacher on the expert studies"
    )
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--base", required=True, help="the B6 export directory")
    parser.add_argument("--filler", required=True, help="the B23 export directory")
    parser.add_argument("--merged", required=True, help="the fill-merge export directory")
    parser.add_argument("--out-root", default="runs/label_gold_audit/b23_fill_merge")
    args = parser.parse_args()

    merged_root = Path(args.merged)
    recorded_audit = json.loads((merged_root / "audit.json").read_text(encoding="utf-8"))
    if recorded_audit.get("merge_version") != MERGE_VERSION:
        raise SystemExit(
            f"expected a {MERGE_VERSION} export; got {recorded_audit.get('merge_version')!r}"
        )

    min_confidence = float(recorded_audit["min_confidence"])
    exclude = tuple(recorded_audit.get("excluded_targets") or ())
    print(f"merge rule    : {recorded_audit['rule']}")
    print(f"min confidence: {min_confidence}")
    print(f"excluded      : {list(exclude) or 'none'}")

    reconstructed, audit = merge_fill_only(
        _structured(Path(args.base)),
        _structured(Path(args.filler)),
        min_confidence=min_confidence,
        exclude_targets=exclude,
    )

    check = _verify_against_recorded(reconstructed, merged_root)
    print(
        f"reconstruction: VERIFIED against the recorded export "
        f"({check['studies_checked']:,} studies, {check['cells_checked']:,} cells)"
    )

    out = Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)
    reconstructed_path = out / "structured_labels.csv"
    reconstructed.to_csv(reconstructed_path, index=False)

    result = audit_export_against_gold(
        args.train_csv, reconstructed_path, label="b23_fill_merge", out_root=out
    )
    result["merge_audit"] = audit
    (out / "gold_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _print_summary(result)


if __name__ == "__main__":
    main()
