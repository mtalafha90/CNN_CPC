"""Fast Kaggle-only execution wrapper for the frozen B39 endpoint.

Scientific endpoint is unchanged. This wrapper reuses the audited B39 dual-GPU
study sharding but increases the inference-only ConvNeXt chunk size from 4 to 64
and keeps PyTorch's CUDA allocator cache warm between studies.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import b39_b37_five_offset_tta_dualgpu as _impl
from .b37_kaggle_fast_runtime import enable_b37_kaggle_fast_runtime

B39_FAST_ENCODER_CHUNK_SIZE = 64
B39_FAST_EXECUTION_VERSION = "b39_hidden_dual_t4_chunk64_cache_reuse_v3"


def generate_b39_submission_dual_gpu_fast(
    config: dict,
    *,
    data_root: str | Path,
    checkpoint: str | Path,
    base_checkpoint: str | Path,
    out_path: str | Path = "submission.csv",
) -> Path:
    """Run frozen B39 with submission-only memory-for-speed execution changes."""
    original_load = _impl._load_replica
    original_release = _impl._release_worker_memory

    def fast_load(checkpoint_path, base_path, device):
        model, payload = original_load(checkpoint_path, base_path, device)
        runtime_state = enable_b37_kaggle_fast_runtime(
            model,
            execution_chunk_size=B39_FAST_ENCODER_CHUNK_SIZE,
        )
        model._kaggle_fast_runtime_state = runtime_state
        return model, payload

    def fast_release(device):
        # Do not call gc.collect()/empty_cache() after every study. Normal Python
        # reference counting releases completed tensors while CUDA keeps reusable
        # blocks cached. This is the intended memory-for-speed tradeoff.
        return None

    _impl._load_replica = fast_load
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
        _impl._load_replica = original_load
        _impl._release_worker_memory = original_release

    output = Path(output)
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "execution_version": B39_FAST_EXECUTION_VERSION,
            "execution_only_change": True,
            "checkpoint_encoder_chunk_size": 4,
            "execution_encoder_chunk_size": int(B39_FAST_ENCODER_CHUNK_SIZE),
            "cuda_cache_reused_between_studies": True,
            "per_study_cuda_empty_cache": False,
            "runtime_acceleration_scope": (
                "Inference execution only: larger ConvNeXt chunking of independent "
                "triplets and CUDA allocator cache reuse. Checkpoint, preprocessing, "
                "five TTA offsets, sparse-MIL and probability aggregation unchanged."
            ),
        }
    )
    manifest["governance"] = (
        "Exact frozen B39 five-offset endpoint. Kaggle-fast execution changes only "
        "the partition of independent encoder images (chunk 4 -> 64) and retains "
        "CUDA allocator cache between studies. B37 checkpoint, 448 preprocessing, "
        "offsets [-2,-1,0,1,2], sigmoid probability averaging, sparse-MIL, "
        "thresholds and blending are unchanged."
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"[B39 fast submit] execution_chunk={B39_FAST_ENCODER_CHUNK_SIZE} "
        "cuda_cache_reuse=True",
        flush=True,
    )
    return output


__all__ = [
    "B39_FAST_ENCODER_CHUNK_SIZE",
    "B39_FAST_EXECUTION_VERSION",
    "generate_b39_submission_dual_gpu_fast",
]
