# B36 — pathology-specific sparse top-k spatial MIL

## Motivation

B35 was a valid negative mechanism test.  It reproduced the frozen B34 base
exactly, but expert-58 macro AUC changed only about +0.0009 and focal-six mean
AUC about +0.0088.  More importantly, B35's dense attention did not localize:
normalized attention entropy was approximately 1.0 for every target and the
largest attention weights were close to the uniform value.

B35 also used a zero residual gate.  That protected the deployed B34 prediction,
but it meant the local attention/query parameters received little or no combined
loss gradient until the gate moved away from zero.

B36 is designed to test the narrower hypothesis that focal MRI abnormalities are
better represented as sparse multiple-instance evidence than as a dense weighted
average over all local positions.

## Architecture

The B34 base and ConvNeXt encoder remain frozen.  B36 reuses the validated B35
image path:

- 32 deterministic sampled centres per real series;
- the first 16 centres exactly reproduce the historical B34 path;
- 2.5D triplets at every centre;
- the current 0.9 center-crop focus policy;
- 3x3 spatial ConvNeXt features per sampled centre;
- exact B34-matched encoder ordering/chunk boundaries for the first 16 centres.

For each study, local features are flattened across real series, centres and 3x3
regions.  Each of the 12 pathologies has an independent linear evidence scorer.
Invalid/missing-series tokens are masked.  For every pathology, only the eight
highest-scoring local tokens are retained.  The local pathology logit is a
log-mean-exp over those eight evidence logits.

This makes the aggregation explicitly sparse: irrelevant locations cannot dilute
focal evidence simply because they outnumber the lesion locations.

## Optimization

The deployed candidate logit remains

```
combined = frozen_B34 + tanh(gate[target]) * local_MIL[target]
```

and every gate starts at exactly zero.  Therefore B36 prediction at initialization
is exactly the frozen B34 prediction.

Unlike B35, B36 gives the local MIL output its own direct weak-supervision loss:

```
L_total = L_combined + 1.0 * L_local
```

Both losses use the same target-balanced weak BCE, label confidence, target
multipliers and 34,010-cell report-derived supervision surface.  The auxiliary
loss solves the zero-gate gradient-starvation problem without altering the base
prediction at initialization.

## Fixed protocol

- Base: completed `llm_fill` B34 one-stage-finetuned checkpoint.
- Base/encoder: frozen.
- Report-only studies: 4,349.
- Eligible MRI series: 24,035.
- Supervision cells: 34,010.
- Expert/gold labels in gradient: 0.
- Dense centres: 32.
- Spatial grid: 3x3.
- Sparse top-k: 8.
- MIL temperature: 1.0.
- Local auxiliary loss weight: 1.0.
- Batch size: 2.
- Gradient accumulation: 1.
- Head LR: 1e-4.
- AdamW weight decay: 1e-4.
- Epochs: exactly 2.
- Seed: existing experiment seed (2026 plus fixed B36 offsets).
- Scheduler: none.

Low-memory DataLoader settings may be used (`num_workers=2`,
`persistent_workers=false`, `prefetch_factor=1`, reduced series cache).  These
settings change data delivery only, not the mathematical experiment.

## Runtime integrity

The first training batch must report

```
[B36] exact-base reconstruction max|delta|=...
```

and the value must be <= 0.002.  The expected result after the B35 exact-batch
fix is 0.

Training also aborts if:

- a frozen B34 parameter receives gradient;
- the residual gate never receives gradient;
- the sparse evidence scorer never receives direct auxiliary gradient;
- an epoch does not cover all 4,349 studies, 24,035 series and 34,010 cells.

`recovery_latest.pt` is written after each completed epoch only for infrastructure
recovery.  It is not a selectable endpoint.  The scientific endpoint remains the
fixed two-epoch `b36_model.pt`.

## Evaluation

Expert-58 remains a reused development diagnostic only.  Evaluation uses the
same three center offsets `[-1, 0, +1]`, evaluates the frozen B34 branch and B36
candidate through the same path, and reports:

- base macro AUC;
- B36 macro AUC and delta;
- all 12 per-target AUCs;
- focal-six mean AUC and delta;
- gate state;
- mean top-1 evidence, mean top-k evidence and top-1-to-kth separation;
- mean number of unique selected local locations across all 12 targets.

## Prospective stop/go rule

This rule is recorded before viewing B36 expert-58 results.

**Strong GO**

- macro delta >= +0.03; **or**
- focal-six delta >= +0.04 with macro delta >= 0.

**Weak / investigate**

- macro delta between about +0.01 and +0.03, or clear sparse-localization benefit
  that is not yet large enough to promote.

**Kill as the main mechanism**

- macro delta <= +0.01 **and** focal-six delta < +0.02.

No target-specific post-hoc hybrid will be selected from expert-58.  A promoted
architecture still requires hidden competition evaluation before replacing the
current model.
