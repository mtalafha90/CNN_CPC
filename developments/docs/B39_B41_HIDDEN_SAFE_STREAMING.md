# B39/B41 hidden-safe streaming execution

## Status

**IMPLEMENTED / EXECUTION-ONLY / PUBLIC EQUIVALENCE REQUIRED BEFORE RESUBMISSION.**

B39 and B41 both passed their visible three-study Kaggle notebooks but Kaggle
reported `Notebook Threw Exception` while rerunning on the hidden test set.  No
hidden traceback was exposed.  Because the two endpoints use different TTA
recipes but share the same high-resolution all-view study materialization and
proactive runtime-abort infrastructure, this hardening changes only those shared
execution mechanics.

Scientific checkpoints and predictions remain frozen.

## Shared hidden-scale risks being removed

The historical B37-family dataset returns an all-view tensor with shape

```text
[V, K, 32, 3, 448, 448]
```

where `V=5` for B39 and `V=3` for B41 and `K` is the number of eligible series.
For large hidden studies this can transiently multiply host RAM because the
per-series tensors, `torch.stack`, `permute`, and contiguous result overlap in
lifetime.

The historical dual-GPU workers also raised a `RuntimeError` when a timing
projection based on the last few completed studies exceeded the remaining
budget.  On a much larger hidden set an unusually slow early study can therefore
turn a conservative forecast into an actual notebook exception.

## Hidden-safe execution contract

The new modules are:

```text
developments/src/rsna_knee/kaggle_hidden_streaming_highres.py
developments/src/rsna_knee/b39_b37_five_offset_tta_dualgpu_streaming.py
developments/src/rsna_knee/b41_highres_aspect_sparse_submission_dualgpu_streaming.py
```

The execution sequence per study is now:

```text
decode each native series once
-> normalize each complete native volume once
-> retain normalized native arrays only
-> construct one TTA view across all series
-> frozen model inference
-> release resized view
-> next TTA view
-> average the same sigmoid probabilities
-> release normalized study arrays
-> trim host allocator arenas
```

The implementation never constructs an all-TTA high-resolution study tensor.
The CUDA allocator remains warm between studies.  Runtime projections are still
printed but are telemetry only and cannot raise a timing exception.

The following remain unchanged:

```text
B39 checkpoint                         frozen B37 fixed-E2
B39 offsets                            [-2,-1,0,1,2]
B41 checkpoint                         frozen B41 fixed-E2
B41 offsets                            [-1,0,1]
full-native percentile normalization   unchanged
90% native crop                        unchanged
B39 direct 448 resize                  unchanged
B41 aspect-preserving resize/pad       unchanged
encoder chunk                          unchanged
sparse-MIL                             unchanged
sigmoid probability averaging          unchanged
thresholding/blending                   none
strict DICOM                            true
```

## Exact preprocessing equivalence

Run locally before creating a new Kaggle artifact:

```bash
cd /media/talafha/Disk_1/CNN_CPC
conda activate rsna-knee
export PYTHONPATH="$PWD/developments/src:${PYTHONPATH:-}"
pytest -q developments/tests/test_kaggle_hidden_streaming_highres.py
```

The tests compare every B39 and B41 streamed TTA tensor against the already
audited normalize-once helper using `torch.equal`, not an approximate tolerance.

## Required public Kaggle equivalence

Before any hidden resubmission, run both the previous fast wrapper and the new
streaming wrapper on the visible three-study test set and compare:

```text
StudyInstanceUID order          identical
columns                         identical
all probabilities               exact or max|delta| <= 1e-7
preprocessing unit tests        exact tensor equality
checkpoint SHA                  unchanged
base-checkpoint SHA             unchanged
```

A hidden submission is permitted only after that public comparison passes.

## New execution versions

```text
B39: b39_hidden_dual_t4_streaming_views_normonce_noabort_v5
B41: b41_hidden_dual_t4_streaming_views_normonce_noabort_v4
```

These version changes describe execution infrastructure only and do not create
new scientific endpoints.
