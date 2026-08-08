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

Series-quality ranking still intentionally remains simple at this baseline; slice count, spacing/FOV, localizer detection, and physical-resolution ranking are later ablations, not prerequisites for the first controlled production run.

## DICOM decoding

The reader supports `.dcm`, `.dicom`, `.ima`, suffix-less, mixed-suffix, and enhanced multi-frame pixel arrays. It applies rescale slope/intercept, `MONOCHROME1` inversion, mixed-size center crop/pad, and physical slice ordering from orientation/position when available.

The full data audit determines whether the real competition corpus exposes codec or enhanced-multiframe edge cases that need further handling. Do not add speculative codec dependencies before that audit.

## Worker-side decoded-volume cache

Persistent DataLoader workers own a bounded LRU cache of recently decoded raw series. The default cap is:

```yaml
series_cache_mb_per_worker: 256
```

This is an in-memory per-process cache. It reduces repeated decoding while keeping RAM bounded and never changes numerical preprocessing.

## 2.5D sampling

A series is normalized with finite 1st/99th percentiles and mapped to `[0,1]`. Triplets use:

```text
[z-gap, z, z+gap]
```

Training defaults to gap choices `{1,2}` and center jitter ±2 slices. NumPy center jitter is seeded from the deterministic worker-local PyTorch RNG.

Current gaps are defined in slice indices rather than millimetres. Physical-spacing-aware sampling remains a post-baseline robustness ablation.

## MRI-specific augmentation

Training applies mild series-consistent perturbations:

- small rotation;
- small translation and scale;
- gamma variation;
- low-frequency multiplicative bias field;
- Gaussian noise;
- slice dropout.

Validation and inference disable stochastic augmentation.

## Validation/submission TTA parity

All requested center-offset views are generated immediately after a series is decoded. A study with TTA therefore returns:

```text
[V, K, S, 3, H, W]
```

where `V` is the number of TTA offsets and `K` is the six-stream dimension.

Production config locks:

```yaml
tta_center_offsets: [-1, 0, 1]
validation_tta_offsets: [-1, 0, 1]
```

Inner epoch selection and primary outer OOF use this same TTA policy. Thus:

- `oof.csv` = production-policy TTA OOF;
- `oof_center.csv` = center-only diagnostic;
- final submission = the same requested TTA contract, unless the runtime fallback is required to guarantee completion.

All views are built from one DICOM decode; TTA multiplies model forwards, not DICOM reads.

Stage-1 weak teacher generation defaults to:

```yaml
weak_oof_tta_offsets: [0]
```

because those predictions cover roughly one third of all non-gold studies and are a major runtime cost. This weak-teacher policy is separate from the primary validation/submission metric policy.

## Preflight

```bash
rsna-knee preflight --data-root DATA_ROOT --split train --sample-size 24
```

Preflight executes real DICOM decoding and the production 2.5D transform. It independently gates selected-stream failures and partial candidate-file corruption.

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
- per-series and global partial-corruption rates.

Pixel audit work uses CPU processes. The audit fails if incomplete, if any selected series cannot decode, or if configured corruption thresholds are exceeded.

## Gold, inner, outer, and weak cross-fit roles

For Stage-1 outer fold `k`:

- `outer_oof`: official gold used only for final fold evaluation;
- `inner_selection`: gold used only to select the epoch count;
- `gold_train_selection`: trusted gold used in Phase A;
- `weak_oof`: non-gold `crossfit_fold=k`, excluded so Stage-1 fold `k` can produce an independent image teacher;
- `weak_train`: remaining report-supervised rows.

After Phase A, Stage-1 Phase B starts from a fresh model, uses all non-outer gold, continues to exclude the `weak_oof` subset, and then writes `fold{k}/weak_oof.csv`.

For Stage-2 outer fold `k`, only a safe Stage-1 fold-`k` teacher is permitted. If multiple Stage-1 candidates exist (for example random vs SSL), candidate selection uses only each candidate's fold-`k` inner AUC. Candidates must share the same inner-fold and validation-TTA contracts.

Stage-2 Phase A stays report-only. In fresh Phase B, `crossfit_fold=k` weak studies are included with image/report consensus targets.

## Stage-2 image-only rescue supervision

Report silence remains zero-weight report supervision. However, because the image teacher is cross-fitted and independent for the permitted weak subset, a highly confident image prediction can contribute a **modest** BCE weight when report confidence is near zero.

Default gates are:

```yaml
cotrain_report_low_confidence: 0.10
cotrain_image_only_positive_threshold: 0.95
cotrain_image_only_negative_threshold: 0.05
cotrain_image_only_weight: 0.20
cotrain_image_only_blend: 0.75
```

The 0.20 weight remains below the ranking-loss confidence gate and far below the trusted-study threshold, so image-only rescue supervision does not become equivalent to gold or high-confidence report/image agreement.

Every Stage-2 fold writes `stage2_supervision.json`, including per-target:

- report non-zero-weight cells;
- Stage-2 non-zero-weight cells;
- `zero_to_nonzero_weight` cells;
- high-confidence Stage-2 cells;
- probability-change counts and magnitudes.

Stage 2 does not emit another `weak_oof.csv`, because its image-teacher rows were used for training.

## Effective supervision diagnostics

Every fold writes:

```text
supervision_plan.json
training_diagnostics.json
```

`supervision_plan.json` describes per-target planned weak/gold supervision mass before optimization. `training_diagnostics.json` records actual per-target weight mass, non-zero cells, participating batches, planned epoch weight, and ranking-pair counts across selection and retraining.

These diagnostics determine whether rare pathologies and the ranking auxiliary are actually active on real data.

## Self-supervised data scope

`rsna-knee pretrain` uses non-gold competition studies only by default. External pretrained weights are disabled in the conservative production configuration unless competition-specific permission is explicitly verified. SSL checkpoints carry and validate their source/training configuration.

## Training versus inference

Reports, report calibration, and co-training consensus are training-only. Final submission inference uses MRI images plus self-describing checkpoints and is Internet-independent.
