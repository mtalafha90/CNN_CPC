"""Visible-test equivalence audit for hidden-safe B39/B41 streaming execution.

Runs the already-audited fast wrapper and the new hidden-safe streaming wrapper
on the same visible Kaggle test surface, then requires identical UID/column order
and bit-for-bit identical probability arrays before hidden resubmission.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .b7_weak_supervision import _read_config
from .b35_training import sha256_file


def _release_between_runs() -> None:
    gc.collect()
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            with torch.cuda.device(index):
                torch.cuda.empty_cache()


def _compare(reference_path: Path, streaming_path: Path) -> dict:
    reference = pd.read_csv(reference_path)
    streaming = pd.read_csv(streaming_path)
    if list(reference.columns) != list(streaming.columns):
        raise RuntimeError("streaming equivalence column order changed")
    if not reference.iloc[:, 0].astype(str).equals(streaming.iloc[:, 0].astype(str)):
        raise RuntimeError("streaming equivalence StudyInstanceUID order changed")
    a = reference.iloc[:, 1:].to_numpy(np.float64)
    b = streaming.iloc[:, 1:].to_numpy(np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError("streaming equivalence probability matrix is invalid")
    delta = np.abs(a - b)
    max_abs = float(delta.max()) if delta.size else 0.0
    exact = bool(np.array_equal(a, b))
    if not exact:
        raise RuntimeError(
            "hidden-safe streaming probabilities are not bit-for-bit equal to the "
            f"audited fast wrapper; max|delta|={max_abs:.12g}"
        )
    return {
        "rows": int(len(reference)),
        "probability_columns": int(a.shape[1]),
        "exact_probability_array_equal": exact,
        "max_abs_probability_delta": max_abs,
        "reference_submission_sha256": sha256_file(reference_path),
        "streaming_submission_sha256": sha256_file(streaming_path),
    }


def run_visible_equivalence(
    endpoint: str,
    config: dict,
    *,
    data_root: str | Path,
    checkpoint: str | Path,
    base_checkpoint: str | Path,
    out_root: str | Path,
) -> dict:
    endpoint = str(endpoint).strip().lower()
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    reference_path = out / f"{endpoint}_audited_fast.csv"
    streaming_path = out / f"{endpoint}_hidden_safe_streaming.csv"

    # Endpoint imports are intentionally lazy so a B41-only Kaggle artifact does
    # not need to contain B39 modules, and vice versa.
    if endpoint == "b39":
        from .b39_b37_five_offset_tta_dualgpu_fast import (
            generate_b39_submission_dual_gpu_fast,
        )
        from .b39_b37_five_offset_tta_dualgpu_streaming import (
            B39_STREAMING_EXECUTION_VERSION,
            generate_b39_submission_dual_gpu_streaming,
        )

        generate_b39_submission_dual_gpu_fast(
            config,
            data_root=data_root,
            checkpoint=checkpoint,
            base_checkpoint=base_checkpoint,
            out_path=reference_path,
        )
        _release_between_runs()
        generate_b39_submission_dual_gpu_streaming(
            config,
            data_root=data_root,
            checkpoint=checkpoint,
            base_checkpoint=base_checkpoint,
            out_path=streaming_path,
        )
        execution_version = B39_STREAMING_EXECUTION_VERSION
    elif endpoint == "b41":
        from .b41_highres_aspect_sparse_submission_dualgpu_fast import (
            generate_b41_submission_dual_gpu_fast,
        )
        from .b41_highres_aspect_sparse_submission_dualgpu_streaming import (
            B41_STREAMING_EXECUTION_VERSION,
            generate_b41_submission_dual_gpu_streaming,
        )

        generate_b41_submission_dual_gpu_fast(
            config,
            data_root=data_root,
            checkpoint=checkpoint,
            base_checkpoint=base_checkpoint,
            out_path=reference_path,
        )
        _release_between_runs()
        generate_b41_submission_dual_gpu_streaming(
            config,
            data_root=data_root,
            checkpoint=checkpoint,
            base_checkpoint=base_checkpoint,
            out_path=streaming_path,
        )
        execution_version = B41_STREAMING_EXECUTION_VERSION
    else:
        raise ValueError("endpoint must be b39 or b41")

    result = {
        "endpoint": endpoint.upper(),
        "execution_version": execution_version,
        **_compare(reference_path, streaming_path),
    }
    audit_path = out / f"{endpoint}_hidden_safe_equivalence.json"
    audit_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"{endpoint.upper()} HIDDEN-SAFE STREAMING EQUIVALENCE: PASS", flush=True)
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", choices=("b39", "b41"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args(argv)
    run_visible_equivalence(
        args.endpoint,
        dict(_read_config(args.config)),
        data_root=args.data_root,
        checkpoint=args.checkpoint,
        base_checkpoint=args.base_checkpoint,
        out_root=args.out_root,
    )


if __name__ == "__main__":
    main()


__all__ = ["run_visible_equivalence"]
