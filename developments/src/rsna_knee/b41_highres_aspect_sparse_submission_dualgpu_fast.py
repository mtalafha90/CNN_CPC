"""Fast Kaggle-only execution wrapper for the frozen B41 endpoint.

Scientific endpoint and model execution chunk are unchanged. This wrapper keeps
the historical B41/B37 encoder_chunk_size=4, reuses the audited B41 dual-GPU
study sharding, removes three-fold repeated full-volume normalization, and keeps
PyTorch's CUDA allocator cache warm between studies.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import b41_highres_aspect_sparse_submission_dualgpu as _impl
from .b41_kaggle_fast_preprocess import B41KaggleNormalizeOnceDataset

B41_FAST_ENCODER_CHUNK_SIZE = 4
B41_FAST_EXECUTION_VERSION = "b41_hidden_dual_t4_normonce_chunk4_cache_reuse_v3"


def generate_b41_submission_dual_gpu_fast(
    config: dict,
    *,
    data_root: str | Path,
    checkpoint: str | Path,
    base_checkpoint: str | Path,
    out_path: str | Path = "submission.csv",
) -> Path:
    """Run frozen B41 with submission-only redundant-preprocessing removal."""
    original_dataset = _impl.B41HighResAspectSparseDataset
    original_release = _impl._release_worker_memory

    def fast_release(device):
        # Completed tensors are released by normal reference counting. Keep CUDA
        # allocator blocks cached between studies; model arithmetic is unchanged.
        return None

    _impl.B41HighResAspectSparseDataset = B41KaggleNormalizeOnceDataset
    _impl._release_worker_memory = fast_release
    try:
        output = _impl.generate_b41_submission_dual_gpu(
            config,
            data_root=data_root,
            checkpoint=checkpoint,
            base_checkpoint=base_checkpoint,
            out_path=out_path,
        )
    finally:
        _impl.B41HighResAspectSparseDataset = original_dataset
        _impl._release_worker_memory = original_release

    output = Path(output)
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "execution_version": B41_FAST_EXECUTION_VERSION,
            "execution_only_change": True,
            "checkpoint_encoder_chunk_size": 4,
            "execution_encoder_chunk_size": 4,
            "cuda_cache_reused_between_studies": True,
            "per_study_cuda_empty_cache": False,
            "historical_volume_normalizations_per_series": 3,
            "execution_volume_normalizations_per_series": 1,
            "tta_resize_calls_per_series": 3,
            "runtime_acceleration_scope": (
                "Inference execution only: compute the exact full-native-volume "
                "normalization once per series and reuse it for the same three B41 "
                "centre offsets; each historical 90% crop, aspect-preserving "
                "antialiased resize-to-fit and zero-pad operation remains separate. "
                "Encoder chunk remains 4."
            ),
        }
    )
    manifest["governance"] = (
        "Exact frozen B41 fixed-E2 native-aspect endpoint. Kaggle-fast v3 retains "
        "the trained encoder execution chunk of 4 and removes only redundant "
        "repeated native-volume normalization (three identical passes -> one reused "
        "result), while retaining CUDA allocator blocks between studies. Checkpoint, "
        "90% native crop, aspect-preserving resize/pad, offsets [-1,0,1], sparse-MIL, "
        "sigmoid probability averaging, thresholds and blending are unchanged."
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        "[B41 fast submit] normalize_once=True execution_chunk=4 "
        "cuda_cache_reuse=True",
        flush=True,
    )
    return output


__all__ = [
    "B41_FAST_ENCODER_CHUNK_SIZE",
    "B41_FAST_EXECUTION_VERSION",
    "generate_b41_submission_dual_gpu_fast",
]
