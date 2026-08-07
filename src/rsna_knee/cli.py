from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd
import yaml

from .constants import TARGETS
from .data import gold_mask, load_series_csv, load_train_csv
from .fusion import tune_alpha
from .inference import infer_checkpoints, validate_submission
from .report_labels import label_dataframe
from .training import train_fold


def read_config(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser("rsna-knee")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inspect"); p.add_argument("--data-root", required=True)
    p = sub.add_parser("pseudo-label"); p.add_argument("--train-csv", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("train"); p.add_argument("--config", required=True); p.add_argument("--fold", type=int, required=True)
    p = sub.add_parser("tune-fusion"); p.add_argument("--train-csv", required=True); p.add_argument("--oof", nargs="+", required=True); p.add_argument("--out", default="fusion.json")
    p = sub.add_parser("infer"); p.add_argument("--config", required=True); p.add_argument("--checkpoints", nargs="+", required=True); p.add_argument("--alpha", type=float, default=0.7); p.add_argument("--out", default="submission.csv")
    args = parser.parse_args()

    if args.cmd == "inspect":
        root = Path(args.data_root)
        df = load_train_csv(root / "train.csv")
        series = load_series_csv(root / "train_series.csv")
        gm = gold_mask(df)
        print(f"studies={len(df)} gold={gm.sum()} unlabeled={(~gm).sum()} reports_present={df['Report'].notna().sum()}")
        print(f"series={len(series)}")
        print(df.loc[gm, TARGETS].fillna(0).sum().sort_values(ascending=False).to_string())
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
    elif args.cmd == "infer":
        config = read_config(args.config)
        subm = infer_checkpoints(config["data_root"], args.checkpoints, config, args.alpha)
        validate_submission(subm)
        subm.to_csv(args.out, index=False)
        print(args.out)


if __name__ == "__main__":
    main()
