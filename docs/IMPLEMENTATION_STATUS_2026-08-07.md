# Review Implementation Status — 2026-08-07

This file maps the recommendations in `REPO_REVIEW_2026-08-07.md` to the code now present on `main`.

## Implemented

- [x] Strict DICOM preflight with real pixel decoding and a configurable failure threshold.
- [x] DICOM decode statistics and clearer failure messages.
- [x] Metadata backfilling integrated before both training and inference routing.
- [x] Image-only test inference by default (`alpha=1.0`); report fusion requires explicit opt-in.
- [x] Per-target-cell gold overrides so partially missing labels are never converted to negatives.
- [x] Fold-safe report-teacher calibration.
- [x] Gold-only OOF validation and bootstrap uncertainty.
- [x] Paired bootstrap run comparison.
- [x] Runtime metadata including elapsed time and peak CUDA memory.
- [x] Frozen experiment configs rather than manual multi-factor edits.
- [x] E01 2D ResNet18 reference configuration.
- [x] E02 true neighboring-slice 2.5D triplets.
- [x] E03 target-specific stream attention.
- [x] E04 six-stream fluid/structural routing.
- [x] E05 ConvNeXt-Tiny.
- [x] E06 optional pairwise AUC-surrogate ranking loss.
- [x] E07 Top-K feature pooling.
- [x] E08 frozen DINOv2/timm pathway, using the checkpoint's 518x518 input definition.
- [x] E09 compact complementary 3D arm.
- [x] Rank/mean prediction ensemble utility and CLI.
- [x] GitHub Actions pytest workflow.
- [x] Kaggle training template updated to select a frozen experiment and preflight first.
- [x] Kaggle submission template updated to image-only inference.
- [x] README synchronized with the current pipeline.
- [x] New tests for 2.5D tensors, target attention, Top-K pooling, partial gold labels, 3D forward pass, and rank ensembling.

## Intentionally not claimed as completed

The following require the real competition data and suitable compute. The repository provides the commands/configurations, but no result should be invented before the runs finish:

- [ ] Run E01 across all folds and freeze its OOF result.
- [ ] Run E02-E09 with the same fold assignments.
- [ ] Compare each candidate against its predecessor using paired bootstrap.
- [ ] Measure real wall-clock runtime and peak memory on the target GPU(s).
- [ ] Determine whether DINOv2, Top-K, ranking loss, or the 3D arm actually improves gold OOF macro-AUC.
- [ ] Build the final heterogeneous ensemble only from components that improve OOF.
- [ ] Generate and submit the final Kaggle `submission.csv`.

## Recommended run order

```text
preflight
  -> E01
  -> E02 (2.5D)
  -> E03 (target attention)
  -> E04 (dual streams)
  -> E05 (ConvNeXt)
  -> E06 (ranking loss)
  -> E07 (Top-K alternative)
  -> E08 (frozen DINOv2 alternative)
  -> E09 (3D complementary arm)
  -> OOF rank/mean ensembles
```

Do not advance an architectural change merely because it is more sophisticated. Retain it only when the frozen gold-only evaluation supports it.
