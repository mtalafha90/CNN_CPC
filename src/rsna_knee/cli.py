from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .constants import TARGETS
from .data import gold_mask, load_series_csv, load_train_csv
from .ensemble import ensemble_predictions
from .evaluation import bootstrap_macro_auc, compare_runs, load_oof
from .fusion import tune_alpha
from .inference import infer_checkpoints, validate_submission
from .preflight import run_preflight
from .report_labels import label_dataframe
from .runtime import resolve_runtime
from .training import train_fold


def read_config(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser("rsna-knee")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect")
    p.add_argument("--data-root", required=True)

    p = sub.add_parser("preflight", help="decode a representative DICOM sample before training")
    p.add_argument("--data-root", required=True)
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--series-csv", default=None)
    p.add_argument("--sample-size", type=int, default=24)
    p.add_argument("--stream-mode", choices=["best", "dual"], default="best")
    p.add_argument("--max-failure-rate", type=float, default=0.05)
    p.add_argument("--no-strict", action="store_true")
    p.add_argument("--out", default=None)

    p = sub.add_parser("pseudo-label")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("train")
    p.add_argument("--config", required=True)
    p.add_argument("--fold", type=int, required=True)

    p = sub.add_parser("tune-fusion")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--oof", nargs="+", required=True)
    p.add_argument("--out", default="fusion.json")

    p = sub.add_parser("ensemble", help="rank-average or mean multiple aligned prediction files")
    p.add_argument("--predictions", nargs="+", required=True)
    p.add_argument("--method", choices=["rank", "mean"], default="rank")
    p.add_argument("--out", required=True)

    p = sub.add_parser("infer")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--out", default="submission.csv")

    p = sub.add_parser("evaluate", help="bootstrap the gold macro-AUC of saved OOF predictions")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--oof", nargs="+", required=True, help="one or more oof.csv files")
    p.add_argument("--compare-oof", nargs="+", default=None, help="a second run, for a paired comparison")
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--out", default=None, help="optional JSON output path")

    sub.add_parser("runtime", help="report the GPU, precision and worker setup")
    args = parser.parse_args()

    if args.cmd == "inspect":
        root = Path(args.data_root)
        df = load_train_csv(root / "train.csv")
        series = load_series_csv(root / "train_series.csv")
        gm = gold_mask(df)
        print(
            f"studies={len(df)} gold={gm.sum()} unlabeled={(~gm).sum()} "
            f"reports_present={df['Report'].notna().sum()}"
        )
        print(f"series={len(series)}")
        print(df.loc[gm, TARGETS].fillna(0).sum().sort_values(ascending=False).to_string())

    elif args.cmd == "preflight":
        result = run_preflight(
            args.data_root,
            split=args.split,
            series_csv=args.series_csv,
            sample_size=args.sample_size,
            stream_mode=args.stream_mode,
            max_failure_rate=args.max_failure_rate,
            strict=not args.no_strict,
        )
        print(result.summary())
        if args.out:
            Path(args.out).write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
            print(f"wrote {args.out}")

    elif args.cmd == "pseudo-label":
        df = load_train_csv(args.train_csv)
        probs, conf = label_dataframe(df)
        out = pd.DataFrame({"StudyInstanceUID": df["StudyInstanceUID"].astype(str)})
        for j, target in enumerate(TARGETS):
            out[target] = probs[:, j]
            out[f"{target}__confidence"] = conf[:, j]
        out.to_csv(args.out, index=False)

    elif args.cmd == "train":
        print(train_fold(read_config(args.config), args.fold))

    elif args.cmd == "tune-fusion":
        best = tune_alpha(args.train_csv, args.oof)
        Path(args.out).write_text(json.dumps(best, indent=2), encoding="utf-8")
        print(best)

    elif args.cmd == "ensemble":
        out = ensemble_predictions(args.predictions, method=args.method)
        out.to_csv(args.out, index=False)
        print(args.out)

    elif args.cmd == "evaluate":
        y_true, predictions, uids = load_oof(args.train_csv, args.oof)
        result = bootstrap_macro_auc(y_true, predictions, n_bootstrap=args.n_bootstrap)
        print(f"scored {len(uids)} gold studies from {len(args.oof)} OOF file(s)")
        print(result.summary())
        print("\nper-target AUC (worst first):")
        for name, value in sorted(
            result.per_target.items(),
            key=lambda kv: (np.isfinite(kv[1]) is False, kv[1]),
        ):
            print(f"  {name:<20} {'undefined' if not np.isfinite(value) else f'{value:.4f}'}")
        payload = result.to_dict()
        if args.compare_oof:
            _, other, _ = load_oof(args.train_csv, args.compare_oof, restrict_to=uids)
            comparison = compare_runs(y_true, predictions, other, n_bootstrap=args.n_bootstrap)
            payload["comparison"] = comparison
            print(
                f"\npaired comparison: median difference {comparison['median_difference']:+.4f} "
                f"[{comparison['ci_lower']:+.4f}, {comparison['ci_upper']:+.4f}], "
                f"second run better in {comparison['probability_b_better']:.0%} of resamples"
            )
        if args.out:
            Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"\nwrote {args.out}")

    elif args.cmd == "runtime":
        print(resolve_runtime({}).describe())

    elif args.cmd == "infer":
        config = read_config(args.config)
        subm = infer_checkpoints(config["data_root"], args.checkpoints, config, args.alpha)
        validate_submission(subm)
        subm.to_csv(args.out, index=False)
        print(args.out)


if __name__ == "__main__":
    main()
