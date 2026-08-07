from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .constants import TARGETS
from .data import gold_mask, load_series_csv, load_train_csv
from .evaluation import bootstrap_macro_auc, compare_runs, load_oof
from .inference import infer_checkpoints
from .preflight import run_preflight
from .report_labels import label_dataframe
from .runtime import resolve_runtime
from .training import train_fold


def read_config(path: str | Path) -> dict:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a YAML mapping: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect", help="summarize competition CSVs")
    p.add_argument("--data-root", required=True)

    p = sub.add_parser("preflight", help="decode a representative DICOM sample")
    p.add_argument("--data-root", required=True)
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--series-csv", default=None)
    p.add_argument("--sample-size", type=int, default=24)
    p.add_argument("--max-decode-failure-rate", type=float, default=0.05)
    p.add_argument("--no-strict", action="store_true")
    p.add_argument("--out", default=None)

    p = sub.add_parser("pseudo-label", help="export report-teacher probabilities")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("train", help="train one leakage-safe fold")
    p.add_argument("--config", required=True)
    p.add_argument("--fold", type=int, required=True)

    p = sub.add_parser("evaluate", help="bootstrap gold-only OOF macro AUC")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--oof", nargs="+", required=True)
    p.add_argument("--compare-oof", nargs="+", default=None)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--out", default=None)

    p = sub.add_parser("infer", help="average fold checkpoints into a submission")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--out", default="submission.csv")

    p = sub.add_parser("runtime", help="show resolved device/precision/data-loader settings")
    p.add_argument("--config", default=None)

    args = parser.parse_args()

    if args.cmd == "inspect":
        root = Path(args.data_root)
        df = load_train_csv(root / "train.csv")
        series = load_series_csv(root / "train_series.csv")
        gm = gold_mask(df)
        print(
            f"studies={len(df)} gold={int(gm.sum())} unlabeled={int((~gm).sum())} "
            f"reports_present={int(df['Report'].notna().sum())} series={len(series)}"
        )
        print(df.loc[gm, TARGETS].sum(axis=0, skipna=True).sort_values(ascending=False).to_string())
        return

    if args.cmd == "preflight":
        result = run_preflight(
            args.data_root,
            split=args.split,
            series_csv=args.series_csv,
            sample_size=args.sample_size,
            stream_mode="dual",
            max_decode_failure_rate=args.max_decode_failure_rate,
            strict=not args.no_strict,
        )
        print(result.summary())
        if args.out:
            Path(args.out).write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        return

    if args.cmd == "pseudo-label":
        df = load_train_csv(args.train_csv)
        probabilities, confidence = label_dataframe(df)
        out = pd.DataFrame({"StudyInstanceUID": df["StudyInstanceUID"].astype(str)})
        for j, target in enumerate(TARGETS):
            out[target] = probabilities[:, j]
            out[f"{target}__confidence"] = confidence[:, j]
        out.to_csv(args.out, index=False)
        print(args.out)
        return

    if args.cmd == "train":
        print(train_fold(read_config(args.config), args.fold))
        return

    if args.cmd == "evaluate":
        y_true, predictions, uids = load_oof(args.train_csv, args.oof)
        result = bootstrap_macro_auc(y_true, predictions, n_bootstrap=args.n_bootstrap)
        print(f"scored {len(uids)} gold studies from {len(args.oof)} OOF file(s)")
        print(result.summary())
        print("\nper-target AUC (worst first):")
        ordered = sorted(
            result.per_target.items(),
            key=lambda kv: (not np.isfinite(kv[1]), kv[1] if np.isfinite(kv[1]) else np.inf),
        )
        for name, value in ordered:
            print(f"  {name:<20} {'undefined' if not np.isfinite(value) else f'{value:.4f}'}")
        payload = result.to_dict()
        if args.compare_oof:
            _, other, other_uids = load_oof(args.train_csv, args.compare_oof, restrict_to=uids)
            if other_uids != uids:
                raise ValueError("paired comparison OOF study ordering mismatch")
            comparison = compare_runs(y_true, predictions, other, n_bootstrap=args.n_bootstrap)
            payload["comparison"] = comparison
            print(
                f"\npaired delta(B-A)={comparison['median_difference']:+.4f} "
                f"[{comparison['ci_lower']:+.4f}, {comparison['ci_upper']:+.4f}], "
                f"P(B>A)={comparison['probability_b_better']:.1%}"
            )
        if args.out:
            Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(args.out)
        return

    if args.cmd == "infer":
        config = read_config(args.config)
        submission = infer_checkpoints(config["data_root"], args.checkpoints, config)
        submission.to_csv(args.out, index=False)
        print(args.out)
        return

    if args.cmd == "runtime":
        config = read_config(args.config) if args.config else {}
        print(resolve_runtime(config).describe())
        return


if __name__ == "__main__":
    main()
