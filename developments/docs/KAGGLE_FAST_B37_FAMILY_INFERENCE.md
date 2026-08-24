# Kaggle-fast B37-family hidden inference

## Purpose

Local training/evaluation remains on the memory-safe historical runtime.  Kaggle
hidden inference may use more of each 14.6-GiB T4 to reduce wall-clock time while
keeping the frozen scientific endpoint unchanged.

The B37-family model encodes independent 448x448 triplets in chunks.  Training
used `encoder_chunk_size=4` because activation graphs had to be retained.  Hidden
inference runs under `torch.inference_mode()`, so a larger execution chunk is
possible.  The fast Kaggle path uses chunk 16 and keeps the PyTorch CUDA allocator
cache warm between studies instead of calling `gc.collect()` and
`torch.cuda.empty_cache()` after every case.

These are execution-only changes.  Checkpoints, preprocessing, TTA offsets,
sparse-MIL, sigmoid probability averaging, thresholds and blending do not
change.

## Frozen fast runtime

- two visible Kaggle T4 GPUs;
- deterministic complete-study sharding by test-row index modulo 2;
- checkpoint encoder chunk recorded as 4;
- Kaggle execution encoder chunk fixed to 16;
- no per-study `torch.cuda.empty_cache()`;
- B39 offsets remain `[-2,-1,0,1,2]`;
- B41 offsets remain `[-1,0,1]`;
- batch semantics and probability aggregation remain unchanged.

## Files

- `developments/src/rsna_knee/b37_kaggle_fast_runtime.py`
- `developments/src/rsna_knee/b39_b37_five_offset_tta_dualgpu_fast.py`
- `developments/src/rsna_knee/b41_highres_aspect_sparse_submission_dualgpu_fast.py`

## Governance

Do not use this fast path for training.  Before hidden submission, run a public
smoke/equivalence check and record wall time, CUDA peak and maximum probability
difference against the historical chunk-4 execution.  If chunk 16 is unstable or
exceeds safe T4 memory, fall back to the audited chunk-4 path rather than tuning
hidden data.  Do not increase chunk size after hidden evidence.
