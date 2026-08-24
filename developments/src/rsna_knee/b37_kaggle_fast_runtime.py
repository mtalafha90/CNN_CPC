"""Submission-only runtime acceleration for B37-family hidden inference.

This module does not alter a checkpoint, preprocessing, TTA offsets, sparse-MIL
aggregation, thresholds, calibration, or labels.  It only changes how independent
448x448 triplets are grouped for ConvNeXt inference and whether PyTorch's CUDA
allocator cache is discarded between studies.

The trained B37 endpoint used encoder_chunk_size=4 because training had to retain
activation graphs.  Hidden inference runs under torch.inference_mode(), so the
same independent images can be encoded in larger chunks without changing the
mathematical model.  A T4 has substantially more usable inference memory than
that conservative training/runtime setting needs.
"""
from __future__ import annotations

import gc

import torch

from .b37_highres_sparse_mil import B37_ENCODER_CHUNK_SIZE

# Conservative speed-oriented value for 14.6-GiB Kaggle T4s.  This is 4x the
# historical training chunk while still leaving substantial margin for the
# frozen model, multi-view input tensors and sparse-MIL activations.
B37_KAGGLE_FAST_ENCODER_CHUNK_SIZE = 16
B37_KAGGLE_FAST_GC_INTERVAL_STUDIES = 50


def enable_b37_kaggle_fast_runtime(model) -> dict:
    """Increase only the inference execution chunk of a verified B37-family model."""
    original = int(getattr(model, "encoder_chunk_size", -1))
    if original != int(B37_ENCODER_CHUNK_SIZE):
        raise RuntimeError(
            "fast Kaggle runtime expected the frozen B37 encoder_chunk_size="
            f"{B37_ENCODER_CHUNK_SIZE}, got {original}"
        )
    fast = int(B37_KAGGLE_FAST_ENCODER_CHUNK_SIZE)
    model.encoder_chunk_size = fast
    # Keep the mirrored metadata used by the historical base encoder consistent
    # with the execution partition even though B37's custom forward uses the
    # residual wrapper's encoder_chunk_size directly.
    if hasattr(model, "base"):
        model.base.encoder_batch_size = fast
    return {
        "checkpoint_encoder_chunk_size": original,
        "execution_encoder_chunk_size": fast,
        "cuda_cache_reused_between_studies": True,
    }


def light_study_cleanup(device: torch.device, completed_studies: int) -> None:
    """Release Python cycles occasionally but keep CUDA allocator blocks cached.

    Tensor references are freed by normal Python ref-counting in the inference
    loop.  Calling torch.cuda.empty_cache() after every study forces expensive
    allocator teardown/reallocation and is unnecessary when memory is stable.
    """
    completed = int(completed_studies)
    if completed > 0 and completed % B37_KAGGLE_FAST_GC_INTERVAL_STUDIES == 0:
        gc.collect()
    # Intentionally do not call torch.cuda.empty_cache().
    # It does not increase memory available to PyTorch itself and destroys the
    # allocation cache that makes repeated same-shape inference fast.


def force_release(device: torch.device) -> None:
    """Emergency/final cleanup only; not part of the per-study fast path."""
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


__all__ = [
    "B37_KAGGLE_FAST_ENCODER_CHUNK_SIZE",
    "B37_KAGGLE_FAST_GC_INTERVAL_STUDIES",
    "enable_b37_kaggle_fast_runtime",
    "light_study_cleanup",
    "force_release",
]
