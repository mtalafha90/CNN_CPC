"""Fast Kaggle-only execution wrapper for the frozen B39 endpoint.

Scientific endpoint and model execution chunk are unchanged.  This wrapper keeps
the historical B37 encoder_chunk_size=4, reuses the audited B39 dual-GPU study
sharding, removes five-fold repeated full-volume normalization, and keeps
PyTorch's CUDA allocator cache warm between studies.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import b39_b37_five_offset_tta_dualgpu as _impl
from .b39_kaggle_fast_preprocess import B39KaggleNormalizeOnceDataset

B39_FAST_ENCODER_CHUNK_SIZE = 4
B39_FAST_EXECUTION_VERSION = "b39_hidden_dual_t4_normonce_chunk4_cache_reuse_v4"


def generate_b39_submission_dual_gpu_fast(
    config: dict,
    *,
    data_root: str | Path,
    checkpoint: str | Path,
    base_checkpoint: str | Path,
    out_path: str | Path = "submission.csv",
) -> Path:
    """Run frozen B39 with submission-only redundant-preprocessing removal."""
    original_dataset = _impl.B37HighResSparseDataset
    original_release = _impl._release_worker_memory

    def fast_release(device):
        # Normal Python reference counting releases completed tensors.  Keep CUDA
        # allocator blocks cached between studies; this changes no model arithmetic.
        return None

    _impl.B37HighResSparseDataset = B39KaggleNormalizeOnceDataset
    _impl._release_worker_memory = fast_release
    try:
        output = _impl.generate_b39_submission_dual_gpu(
            config,
            data_root=data_root,
            checkpoint=checkpoint,
            base_checkpoint=base_checkpoint,
            out_path=out_path,
        )
    finally:
        _impl.B37HighResSparseDataset = original_dataset
        _impl._release_worker_memory = original_release

    output = Path(output)
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "execution_version": B39_FAST_EXECUTION_VERSION,
            "execution_only_change": True,
            "checkpoint_encoder_chunk_size": 4,
            "execution_encoder_chunk_size": 4,
            "cuda_cache_reused_between_studies": True,
            "per_study_cuda_empty_cache": False,
            "historical_volume_normalizations_per_series": 5,
            "execution_volume_normalizations_per_series": 1,
            "tta_resize_calls_per_series": 5,
            "runtime_acceleration_scope": (
                "Inference execution only: compute the exact full-native-volume "
                "normalization once per series and reuse it for the same five B39 "
                "centre offsets; each historical per-offset crop and antialiased "
                "448x448 resize remains separate. Encoder chunk remains 4."
            ),
        }
    )
    manifest["governance"] = (
        "Exact frozen B39 five-offset endpoint. Kaggle-fast v4 retains the trained "
        "encoder execution chunk of 4 and removes only redundant repeated native-"
        "volume normalization (five identical passes -> one reused result), while "
        "retaining CUDA allocator blocks between studies. Checkpoint, 448 crop/"
        "resize, offsets [-2,-1,0,1,2], sparse-MIL, sigmoid probability averaging, "
        "thresholds and blending are unchanged."
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        "[B39 fast submit] normalize_once=True execution_chunk=4 "
        "cuda_cache_reuse=True",
        flush=True,
    )
    return output


__all__ = [
    "B39_FAST_ENCODER_CHUNK_SIZE",
    "B39_FAST_EXECUTION_VERSION",
    "generate_b39_submission_dual_gpu_fast",
]
