# B37 — 448 high-resolution pathology-specific sparse MIL

## Status

**PROSPECTIVE / NOT YET RUN.**

B37 is frozen before its first expert-58 result.  It is a joint mechanism test,
not a clean ablation of resolution alone.

The first training attempt on 2026-08-22 was externally killed by
`systemd-oomd` during epoch 1 at step 1,650/2,175.  The terminal cgroup had
reached 22.3 GiB under sustained memory pressure.  No epoch completed, no
checkpoint was written and no expert evaluation was performed, so this was an
infrastructure interruption rather than an experimental result.  The relaunch
retains the frozen scientific protocol and seed while disabling worker prefetch
and pinned host buffers.  Completed variable-size batches are explicitly
released before the next batch is constructed, and free host arenas are trimmed
at each 50-step telemetry point.

## Why this experiment exists

The completed native-DICOM audit inspected 24,371 training series and 819,078
headers with zero header failures.  The key geometry results were:

- median matrix: 512 x 512;
- matrix p05/p50/p95 roughly 320 / 512 / 960 rows;
- median PixelSpacing: 0.3125 mm/pixel;
- PixelSpacing p05-p95: 0.1667-0.5625 mm/pixel;
- median physical FOV: about 160 mm;
- 90% native crop median maximum dimension: 461 pixels;
- padding-only to 640 would preserve only about 85.3% of series.

Therefore a universal no-resize canvas is neither computationally attractive nor
physically scale-standardized.  A single resize remains necessary, but 448 is
close to native median sampling after the fixed 90% focus crop:

```text
160 mm x 0.90 / 448 ~= 0.321 mm/pixel
```

which is close to the observed 0.3125 mm/pixel median acquisition sampling.

B36 then supplied the architectural motivation: explicit sparse top-k MIL learned
non-uniform pathology-specific locations, but its frozen 224 representation did
not improve the focal-six expert mean.  B37 asks whether the same sparse mechanism
becomes useful when local image information is much richer and the final encoder
stage is allowed to adapt.

## Frozen B37 contract

```text
input preprocessing
  full native-volume percentile normalization
  -> same deterministic 32 B35/B36 centres
  -> 2.5D triplets, gap=1
  -> centered 90% crop at native resolution
  -> ONE antialiased bilinear resize to 448x448

representation
  ConvNeXt encoder from the current full-fill checkpoint
  final encoder stage + output norm trainable
  all earlier encoder blocks frozen
  B34 non-encoder hierarchy frozen and always evaluated in deployed mode

local mechanism
  32 slice positions
  6x6 local feature grid per slice
  pathology-specific B36 evidence classifiers
  top-k = 8
  temperature = 1.0
  top-k log-mean-exp pooling
  direct local auxiliary loss weight = 1.0
  zero-start target-wise residual gate

optimization
  report-only studies = 4,349
  MRI series = 24,035
  supervision cells = 34,010
  full fill-only LLM supervision
  fixed seed = 2026
  micro-batch = 2
  fixed epochs = 2
  sparse-head LR = 1e-4
  encoder-tail LR = 5e-6 (0.05x)
  no expert gradients
  no expert checkpoint selection
  no resolution/grid/top-k sweep
```

The B34 hierarchy itself is frozen.  Encoder-tail gradients are produced through
both the reconstructed global B34 logit and the local auxiliary branch.  This
means B37 genuinely adapts the image representation while retaining the deployed
B34 aggregation function.

## Why 6x6

At 224 input, ConvNeXt's final spatial map is roughly 7x7 and B36 used a 3x3
adaptive grid.  At 448 the final map is roughly 14x14.  Scaling the B36 grid from
3x3 to 6x6 approximately preserves local region granularity instead of throwing
away the extra spatial information immediately.

Per series:

```text
B36: 32 x 3 x 3 =   288 local locations
B37: 32 x 6 x 6 = 1,152 local locations
```

Only the strongest eight locations per pathology enter the local MIL logit, so
this does not reintroduce B35's dense-softmax dilution mechanism.

## Memory preflight

Do not launch the fixed two-epoch run before the exact forward/backward preflight
passes on the intended RTX A4500 configuration.  The preflight deterministically
selects the two studies with the largest eligible-series counts, so it exercises
the worst-case padded 14-series micro-batch rather than an arbitrary first batch.

```bash
cd /media/talafha/Disk_1/CNN_CPC
conda activate rsna-knee
export PYTHONPATH="$PWD/developments/src:${PYTHONPATH:-}"

python -m rsna_knee.b37_highres_sparse_training \
  --config config/b37_highres_sparse_448.yaml \
  --data-root "$DATA_ROOT" \
  --labels-root "$LLM_LABELS" \
  --series-policy "$SERIES_POLICY" \
  --base-checkpoint "$BASE_MODEL" \
  --out-root runs/071_Experiment_B37_highres_448_sparse_mil/b37_highres_sparse_mil \
  --preflight-only
```

The preflight performs one real forward/backward pass but **no optimizer step**.
It reports process RSS, available host memory, and current/peak CUDA memory.
A later training launch starts from a fresh process/seed and therefore remains
the prospective endpoint.  Training reports the same memory telemetry every 50
steps.

## Primary expert-58 decision rule

Expert-58 is reused development evidence, not independent validation.

Primary endpoint:

```text
B37 combined 448 sparse-MIL
minus
historical 224 full-fill B34
```

**STRONG GO**

```text
macro delta >= +0.020
OR
focal-six delta >= +0.030 with macro delta >= 0
```

**KILL joint mechanism as the main explanation**

```text
macro delta <= +0.005
AND
focal-six delta < +0.010
```

The evaluator also reports the B37 448 **global-only** branch.  That line is
mechanistic decomposition only and is not an additional selection gate.

Focal six are frozen as:

- ACL
- MCL
- Medial Meniscus
- Lateral Meniscus
- Contusion
- Fracture

## Governance

Do not use expert-58 to change 448, 6x6, top-k=8, crop fraction, local auxiliary
weight, epoch count, target subset, or encoder depth.  A promoted candidate still
requires hidden competition evidence.
