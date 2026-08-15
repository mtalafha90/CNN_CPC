#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from rsna_knee.chatgpt_hybrid_supervision import build_hybrid_export


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a ChatGPT-created hybrid report-label cache into an "
            "exploratory B7-compatible supervision export"
        )
    )
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument(
        "--out-root",
        default="runs/chatgpt_hybrid_supervision",
    )
    args = parser.parse_args()

    audit = build_hybrid_export(
        args.train_csv,
        args.cache,
        out_root=args.out_root,
    )

    print("=" * 72)
    print("CHATGPT HYBRID SUPERVISION EXPORT")
    print("=" * 72)
    for key in (
        "cache_file_sha256",
        "cache_entries",
        "cache_hashes_matching_train",
        "cache_hashes_not_in_train",
        "n_studies",
        "n_gold_audit_only",
        "n_report_only_training",
        "matched_non_gold_studies",
        "matched_gold_studies",
        "active_training_studies",
        "usable_cells_total",
        "possible_cells_total",
        "cell_coverage",
        "gold_rows_in_training_targets",
    ):
        print(f"{key:34s}: {audit[key]}")

    print()
    print("EXPLORATORY ONLY")
    print("formal B23 compatible :", audit["formal_b23_compatible"])
    print("formal B24 eligible   :", audit["formal_b24_eligible"])
    print("gold acceptance       :", audit["gold_acceptance_allowed"])
    print()
    print(json.dumps(audit["targets"], indent=2))


if __name__ == "__main__":
    main()
