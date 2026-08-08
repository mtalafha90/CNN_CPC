# Dataset and DICOM handling

## CSV contracts

`train.csv` contains `StudyInstanceUID`, `Report`, and the 12 target columns. `test.csv` requires `StudyInstanceUID`; reports are not required at inference.

Series CSVs require `StudyInstanceUID`, `SeriesInstanceUID`, `Fluid_Sensitive`, `Fat_Suppression`, and `Anatomical_Plane`. Duplicate study/series rows and missing UIDs are rejected.

## Nullable sequence metadata

Missing `Fluid_Sensitive` and `Fat_Suppression` values remain **unknown** when the CSV is loaded. DICOM backfill independently repairs:

- anatomical plane from image orientation;
- fluid sensitivity from timing/weighting;
- fat suppression from acquisition metadata.

A populated CSV field remains authoritative. If a field remains unknown after repair, routing uses a conservative `False` fallback only at final scoring.

## Six-stream routing

```text
sagittal_fluid       sagittal_structural
coronal_fluid        coronal_structural
axial_fluid          axial_structural
```

Fluid/fat-suppressed candidates rank toward the fluid slot; non-fluid/non-fat-suppressed candidates rank toward the structural slot. When alternatives exist, the same series is not placed in both semantic slots. Missing streams remain absent and are masked before ConvNeXt.

## DICOM decoding

The reader supports `.dcm`, `.dicom`, `.ima`, suffix-less, mixed-suffix, and enhanced multi-frame layouts. It applies rescale slope/intercept, `MONOCHROME1` inversion, mixed-size center crop/pad, and physical slice ordering from orientation/position when available.

## Worker-side decoded-volume cache

Persistent DataLoader workers own a bounded LRU cache of recently decoded raw series. The default cap is:

```yaml
series_cache_mb_per_worker: 256
```

This is deliberately an in-memory per-process cache rather than an unbounded disk cache: MRI volumes can be large, and writing all selected series to `/kaggle/working` would create a second storage bottleneck. The cache is opportunistic and never changes numerical preprocessing.

## Deterministic 2.5D sampling

A series is normalized with finite 1st/99th percentiles and mapped to `[0,1]`. Triplets use:

```text
[z-gap, z, z+gap]
```

Training defaults to gap choices `{1,2}` and center jitter ±2 slices. The worker-local NumPy jitter generator is seeded from PyTorch's seeded worker RNG. For a fixed seed, worker count, and configuration, stochastic sampling is reproducible.

## MRI-specific augmentation

Training applies mild series-consistent perturbations:

- small rotation;
- small translation and scale;
- gamma variation;
- low-frequency multiplicative bias field;
- Gaussian noise;
- slice dropout.

Validation and inference disable stochastic augmentation.

## One-decode TTA

At inference, all requested slice-center offsets are generated immediately after a series is decoded. A study therefore returns:

```text
[V, K, S, 3, H, W]
```

where `V` is the number of TTA center offsets and `K` is the six-stream dimension. All fold models consume those views before the batch is released, so three folds × three TTA views still require only one DICOM decode per selected series for that study.

## Preflight

```bash
rsna-knee preflight --data-root DATA_ROOT --split train --sample-size 24
```

Preflight executes real DICOM decoding and the production 2.5D transform. It distinguishes legitimate missing semantic streams from selected-stream path/decode failures and reports metadata repair counts.

## Full audit

```bash
rsna-knee audit --config configs/train.yaml --out-dir runs/audit
```

The full audit adds:

- report-state/confidence statistics;
- fold target counts;
- six-stream availability;
- every selected training series' decode status;
- total candidate DICOM files and failed instances;
- partial-series corruption counts.

Pixel audit work runs in CPU processes. If it cannot finish before the configured wall-clock reserve, it writes partial CSV output and then fails; it does not label the audit complete.

## Gold, inner, outer, and weak cross-fit roles

For fold `k`:

- `outer_oof`: official gold used only for final fold evaluation;
- `inner_selection`: gold used only to choose training duration in Phase A;
- `gold_train_selection`: trusted gold used in Phase-A training;
- `weak_oof`: non-gold `crossfit_fold=k`, excluded so fold `k` can produce an independent image teacher;
- `weak_train`: report-supervised rows.

After epoch selection, Phase B retrains a fresh model using all non-outer gold while continuing to exclude `weak_oof`.

For Stage-2 outer fold `k`, only Stage-1 `fold{k}/weak_oof.csv` may be used. Other folds' weak predictions are rejected because their models may have trained on outer-gold fold `k`.

## Self-supervised data scope

`rsna-knee pretrain` uses non-gold competition studies only by default. External pretrained weights are disabled in the conservative production configuration until the exact competition-specific rule is explicitly verified.

## Training versus inference

Reports, report calibration, and co-training consensus are training-only. Final submission inference uses MRI images plus self-describing checkpoints and is Internet-independent.
