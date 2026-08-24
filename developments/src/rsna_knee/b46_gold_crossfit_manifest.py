"""CLI to create the one frozen B46 official-gold fold manifest."""
from __future__ import annotations

import argparse

from .b46_gold_crossfit import B46_MANIFEST_NAME, B46_RUN_ROOT, build_gold_fold_manifest


def main() -> None:
    parser = argparse.ArgumentParser("Create frozen B46 five-fold official-gold manifest")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", default=f"{B46_RUN_ROOT}/{B46_MANIFEST_NAME}")
    args = parser.parse_args()
    build_gold_fold_manifest(args.data_root, out_path=args.out)


if __name__ == "__main__":
    main()
