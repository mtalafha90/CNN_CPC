"""Visible-test audit for hidden-safe B39/B41 streaming execution.

A fresh fp16 dual-GPU re-execution is useful telemetry, but it is not a reliable
bit-for-bit reference: the same frozen endpoint can exhibit tiny numerical drift
between independent CUDA executions.  The pass criterion is therefore the SHA-256
of the independently audited public submission artifact produced before the
hidden-safe implementation existed.  A streaming run must reproduce that
canonical CSV byte-for-byte.

The optional fresh fast-wrapper run is still compared and reported, but any tiny
re-execution drift is telemetry only when the streaming artifact matches the
canonical public SHA exactly.
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

# Independently audited visible-test artifacts from the original fast submission
# runs.  These were recorded before the hidden-safe streaming implementation.
B41_CANONICAL_PUBLIC_SUBMISSION_SHA256 = (
    "70c4166f589884aa45f4eeb493589245fa15571771ac272e4a5d4685a008c754"
)
B39_CANONICAL_PUBLIC_SUBMISSION_SHA256 = (
    "8a45f8969270aede1456d4ab9b2afd3db21a52af5a42e71d34f100e86e441225"
)
CANONICAL_PUBLIC_SHA256 = {
    "b39": B39_CANONICAL_PUBLIC_SUBMISSION_SHA256,
    "b41": B41_CANONICAL_PUBLIC_SUBMISSION_SHA256,
}


def _release_between_runs() -> None:
    gc.collect()
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            with torch.cuda.device(index):
                torch.cuda.empty_cache()


def _endpoint_streaming_runner(endpoint: str):
    endpoint = str(endpoint).strip().lower()
    if endpoint == "b39":
        from .b39_b37_five_offset_tta_dualgpu_streaming import (
            B39_STREAMING_EXECUTION_VERSION,
            generate_b39_submission_dual_gpu_streaming,
        )

        return B39_STREAMING_EXECUTION_VERSION, generate_b39_submission_dual_gpu_streaming
    if endpoint == "b41":
        from .b41_highres_aspect_sparse_submission_dualgpu_streaming import (
            B41_STREAMING_EXECUTION_VERSION,
            generate_b41_submission_dual_gpu_streaming,
        )

        return B41_STREAMING_EXECUTION_VERSION, generate_b41_submission_dual_gpu_streaming
    raise ValueError("endpoint must be b39 or b41")


def _endpoint_fast_runner(endpoint: str):
    endpoint = str(endpoint).strip().lower()
    if endpoint == "b39":
        from .b39_b37_five_offset_tta_dualgpu_fast import (
            generate_b39_submission_dual_gpu_fast,
        )

        return generate_b39_submission_dual_gpu_fast
    if endpoint == "b41":
        from .b41_highres_aspect_sparse_submission_dualgpu_fast import (
            generate_b41_submission_dual_gpu_fast,
        )

        return generate_b41_submission_dual_gpu_fast
    raise ValueError("endpoint must be b39 or b41")


def _validate_streaming_against_canonical(endpoint: str, streaming_path: Path) -> dict:
    endpoint = str(endpoint).strip().lower()
    expected = CANONICAL_PUBLIC_SHA256[endpoint]
    observed = sha256_file(streaming_path)
    frame = pd.read_csv(streaming_path)
    if frame.empty or frame.shape[1] < 2:
        raise RuntimeError("hidden-safe streaming submission is empty or malformed")
    values = frame.iloc[:, 1:].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("hidden-safe streaming submission contains non-finite probabilities")
    matches = bool(observed == expected)
    if not matches:
        raise RuntimeError(
            f"{endpoint.upper()} hidden-safe streaming did not reproduce the canonical "
            f"audited public artifact: expected SHA={expected}, observed SHA={observed}"
        )
    return {
        "canonical_public_submission_sha256": expected,
        "streaming_submission_sha256": observed,
        "streaming_matches_canonical_public": matches,
        "rows": int(len(frame)),
        "probability_columns": int(values.shape[1]),
    }


def _compare_reexecution(reference_path: Path, streaming_path: Path) -> dict:
    reference = pd.read_csv(reference_path)
    streaming = pd.read_csv(streaming_path)
    if list(reference.columns) != list(streaming.columns):
        raise RuntimeError("fresh-fast telemetry column order changed")
    if not reference.iloc[:, 0].astype(str).equals(streaming.iloc[:, 0].astype(str)):
        raise RuntimeError("fresh-fast telemetry StudyInstanceUID order changed")
    a = reference.iloc[:, 1:].to_numpy(np.float64)
    b = streaming.iloc[:, 1:].to_numpy(np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError("fresh-fast telemetry probability matrix is invalid")
    delta = np.abs(a - b)
    return {
        "fresh_fast_submission_sha256": sha256_file(reference_path),
        "fresh_fast_matches_streaming_exactly": bool(np.array_equal(a, b)),
        "fresh_fast_vs_streaming_max_abs_probability_delta": (
            float(delta.max()) if delta.size else 0.0
        ),
    }


def run_canonical_visible_check(
    endpoint: str,
    config: dict,
    *,
    data_root: str | Path,
    checkpoint: str | Path,
    base_checkpoint: str | Path,
    out_root: str | Path,
) -> dict:
    """Run streaming only and require exact reproduction of the canonical public CSV."""
    endpoint = str(endpoint).strip().lower()
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    streaming_path = out / f"{endpoint}_hidden_safe_streaming.csv"
    execution_version, streaming_runner = _endpoint_streaming_runner(endpoint)
    streaming_runner(
        config,
        data_root=data_root,
        checkpoint=checkpoint,
        base_checkpoint=base_checkpoint,
        out_path=streaming_path,
    )
    result = {
        "endpoint": endpoint.upper(),
        "execution_version": execution_version,
        **_validate_streaming_against_canonical(endpoint, streaming_path),
        "fresh_fast_reexecution_used_for_pass_fail": False,
    }
    audit_path = out / f"{endpoint}_hidden_safe_canonical_check.json"
    audit_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"{endpoint.upper()} HIDDEN-SAFE CANONICAL EQUIVALENCE: PASS", flush=True)
    return result


def run_visible_equivalence(
    endpoint: str,
    config: dict,
    *,
    data_root: str | Path,
    checkpoint: str | Path,
    base_checkpoint: str | Path,
    out_root: str | Path,
) -> dict:
    """Run fresh fast + streaming, but pass/fail only on the canonical public SHA."""
    endpoint = str(endpoint).strip().lower()
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    reference_path = out / f"{endpoint}_fresh_fast_telemetry.csv"
    streaming_path = out / f"{endpoint}_hidden_safe_streaming.csv"

    fast_runner = _endpoint_fast_runner(endpoint)
    execution_version, streaming_runner = _endpoint_streaming_runner(endpoint)

    fast_runner(
        config,
        data_root=data_root,
        checkpoint=checkpoint,
        base_checkpoint=base_checkpoint,
        out_path=reference_path,
    )
    _release_between_runs()
    streaming_runner(
        config,
        data_root=data_root,
        checkpoint=checkpoint,
        base_checkpoint=base_checkpoint,
        out_path=streaming_path,
    )

    canonical = _validate_streaming_against_canonical(endpoint, streaming_path)
    telemetry = _compare_reexecution(reference_path, streaming_path)
    canonical_sha = canonical["canonical_public_submission_sha256"]
    telemetry["fresh_fast_matches_canonical_public"] = bool(
        telemetry["fresh_fast_submission_sha256"] == canonical_sha
    )

    result = {
        "endpoint": endpoint.upper(),
        "execution_version": execution_version,
        **canonical,
        **telemetry,
        "fresh_fast_reexecution_used_for_pass_fail": False,
        "interpretation": (
            "PASS is based only on exact reproduction of the independently audited "
            "canonical public CSV. Fresh fp16 dual-GPU re-execution differences are "
            "reported as numerical telemetry and do not redefine the reference."
        ),
    }
    audit_path = out / f"{endpoint}_hidden_safe_equivalence.json"
    audit_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(f"{endpoint.upper()} HIDDEN-SAFE CANONICAL EQUIVALENCE: PASS", flush=True)
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", choices=("b39", "b41"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument(
        "--with-fresh-fast-telemetry",
        action="store_true",
        help="also rerun the old fp16 fast wrapper for non-authoritative telemetry",
    )
    args = parser.parse_args(argv)
    runner = run_visible_equivalence if args.with_fresh_fast_telemetry else run_canonical_visible_check
    runner(
        args.endpoint,
        dict(_read_config(args.config)),
        data_root=args.data_root,
        checkpoint=args.checkpoint,
        base_checkpoint=args.base_checkpoint,
        out_root=args.out_root,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "B39_CANONICAL_PUBLIC_SUBMISSION_SHA256",
    "B41_CANONICAL_PUBLIC_SUBMISSION_SHA256",
    "run_canonical_visible_check",
    "run_visible_equivalence",
]
