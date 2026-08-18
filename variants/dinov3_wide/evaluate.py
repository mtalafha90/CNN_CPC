"""Validation and test-set prediction for wide-encoder checkpoints.

The supported stages rebuild a model from its checkpoint using the 768-d
builder, which cannot reconstruct a wider one. These entry points supply the
variant's loader instead and reuse the rest of each stage unchanged, so scoring
and submission writing are the same code as the supported path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from model._implementation import read_config, run_directory
from testing.test import predict_test_set
from validation.validate import evaluate

from .model import load_wide_checkpoint


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="config/current_model.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--name")


def _config(args) -> dict:
    config = read_config(args.config)
    config["data_root"] = str(Path(args.data_root).resolve())
    return config


def _stem(args) -> str:
    return args.name or Path(args.checkpoint).resolve().parent.name


def validate() -> None:
    parser = argparse.ArgumentParser(description="Score a wide-encoder checkpoint")
    _common(parser)
    args = parser.parse_args()

    result = evaluate(_config(args), checkpoint=args.checkpoint, loader=load_wide_checkpoint)
    out = run_directory(args.experiment, "validate") / f"{_stem(args)}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(out)


def predict() -> None:
    parser = argparse.ArgumentParser(description="Predict the test set with a wide encoder")
    _common(parser)
    args = parser.parse_args()

    out_path = run_directory(args.experiment, "test") / f"{_stem(args)}.csv"
    predict_test_set(
        _config(args),
        checkpoint=args.checkpoint,
        out_path=out_path,
        loader=load_wide_checkpoint,
    )
