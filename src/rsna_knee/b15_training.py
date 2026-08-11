"""CLI for B15 MRI-SSL initialized downstream training on weak-holdout-v2."""
from __future__ import annotations
import argparse
from .b7_weak_supervision import _read_config
from .b15_downstream import train_v2_downstream

def main() -> None:
    parser = argparse.ArgumentParser("rsna-knee-b15")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--b6-root", required=True)
    parser.add_argument("--series-policy", required=True)
    parser.add_argument("--weak-holdout-root", required=True)
    parser.add_argument("--ssl-checkpoint", required=True)
    parser.add_argument("--out-root", default="runs/b15_mri_ssl/downstream")
    args = parser.parse_args()
    config = _read_config(args.config)
    if args.data_root:
        config = dict(config)
        config["data_root"] = args.data_root
    path = train_v2_downstream(
        config,
        mode="b15",
        b6_root=args.b6_root,
        series_policy_path=args.series_policy,
        weak_holdout_root=args.weak_holdout_root,
        out_root=args.out_root,
        ssl_checkpoint=args.ssl_checkpoint,
    )
    print(path)

if __name__ == "__main__":
    main()
