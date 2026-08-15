from __future__ import annotations

import argparse
import json
from pathlib import Path

from .report_teacher import run_report_teacher_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(
        "rsna-knee-report-teacher",
        description="Benchmark and export the fold-safe competition-data-only report teacher",
    )
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--out-dir", default="runs/report_teacher")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    payload = run_report_teacher_benchmark(
        args.train_csv,
        out_dir=args.out_dir,
        n_folds=3,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
    )
    print(json.dumps(payload["oof"], indent=2))
    print(Path(args.out_dir) / "metrics.json")


if __name__ == "__main__":
    main()
