"""Fast Kaggle-only execution wrapper for the frozen B41 endpoint.

Scientific endpoint is unchanged.  This wrapper reuses the audited B41 dual-GPU
study sharding but increases the inference-only ConvNeXt chunk size from 4 to 16
and keeps PyTorch's CUDA allocator cache warm between studies.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import b41_highres_aspect_sparse_submission_dualgpu as _impl
from .b37_kaggle_fast_runtime import (
    B37_KAGGLE_FAST_ENCODER_CHUNK_SIZE,
    enable_b37_kaggle_fast_runtime,
)

B41_FAST_EXECUTION_VERSION = "b41_hidden_dual_t4_chunk16_cache_reuse_v2"


def generate_b41_submission_dual_gpu_fast(
    config: dict,
    *,
    data_root: str | Path,
    checkpoint: str | Path,
    base_checkpoint: str | Path,
    out_path: str | Path = "submission.csv",
) -> Path:
    """Run frozen B41 with submission-only memory-for-speed execution changes."""
    original_load = _impl._load_replica
    original_release = _impl._release_worker_memory

    def fast_load(checkpoint_path, base_path, device):
        model, payload = original_load(checkpoint_path, base_path, device)
        runtime_state = enable_b37_kaggle_fast_runtime(model)
        model._kaggle_fast_runtime_state = runtime_state
        return model, payload

    def fast_release(device):
        # Completed tensors are dropped by normal Python ref-counting.  Keep CUDA
        # allocator blocks cached so the next study reuses them instead of forcing
        # expensive cudaMalloc/cudaFree cycles after every case.
        return None

    _impl._load_replica = fast_load
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
        _impl._load_replica = original_load
        _impl._release_worker_memory = original_release

    output = Path(output)
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "execution_version": B41_FAST_EXECUTION_VERSION,
            "execution_only_change": True,
            "checkpoint_encoder_chunk_size": 4,
            "execution_encoder_chunk_size": int(B37_KAGGLE_FAST_ENCODER_CHUNK_SIZE),
            "cuda_cache_reused_between_studies": True,
            "per_study_cuda_empty_cache": False,
            "runtime_acceleration_scope": (
                "Inference execution only: larger ConvNeXt chunking of independent "
                "triplets and CUDA allocator cache reuse. Checkpoint, B41 aspect-"
                "preserving preprocessing, three TTA offsets, sparse-MIL and "
                "probability aggregation unchanged."
            ),
        }
    )
    manifest["governance"] = (
        "Exact frozen B41 endpoint. Kaggle-fast execution changes only the partition "
        "of independent encoder images (chunk 4 -> 16) and retains CUDA allocator "
        "cache between studies. Checkpoint, native-aspect preprocessing, offsets "
        "[-1,0,1], sigmoid probability averaging, sparse-MIL, thresholds and "
        "blending are unchanged."
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"[B41 fast submit] execution_chunk={B37_KAGGLE_FAST_ENCODER_CHUNK_SIZE} "
        "cuda_cache_reuse=True",
        flush=True,
    )
    return output


__all__ = [
    "B41_FAST_EXECUTION_VERSION",
    "generate_b41_submission_dual_gpu_fast",
]
