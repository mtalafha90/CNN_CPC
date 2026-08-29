# Maintained operational model: B42

This document identifies the maintained operational benchmark. It does not
rewrite the historical B20--B34 interface as if it were B42.

## Status

The maintained operational baseline is **B42 constant-area native-aspect ragged
sparse MIL**. B37, B41, and B42 are tied at a displayed hidden Kaggle macro AUC
of `0.714`; B42 is retained as the reference because it keeps correct native
aspect ratio without square-padding every series. This is an operational choice,
not a scientific promotion above B37 or B41.

The exact contract and results are frozen in
[`B42_CONSTANT_AREA_ASPECT_SPARSE_MIL.md`](../developments/docs/B42_CONSTANT_AREA_ASPECT_SPARSE_MIL.md).
For endpoint identity, required artefacts, and hidden-safe inference, see
[`ACTIVE_ENDPOINTS.md`](ACTIVE_ENDPOINTS.md).

## What it does

Each study is a knee MRI examination made up of several acquired series, and
the model predicts twelve binary findings for the study as a whole. The
competition scores the unweighted mean of the twelve ROC AUCs.

```text
MRI study
  -> every eligible real MRI series
  -> full-volume normalization and native 90% central crop
  -> aspect-preserving constant-area resize near 448^2 pixels
  -> thin reflection padding only to ConvNeXt stride 32
  -> 32 deterministic 2.5D slice centres for local sparse evidence
  -> ragged per-series ConvNeXt encoding
  -> frozen B34 global hierarchy plus B37 sparse TopK=8 local residual
  -> 12 probabilities
```

Studies carry a varying number of series, so batches are padded and carry a
presence mask rather than being truncated to a fixed count.

## Design decisions worth knowing

**Only the permitted encoder tail is trainable.** B42 fine-tunes the final
ConvNeXt stage and output norm under the fixed-E2 protocol. Checkpoint and
encoder fingerprints are verified before inference.

**Geometry is fixed, but not square.** The 90% native crop is resized once with
one isotropic scale factor; the resulting rectangle is reflection-padded only
to a multiple of 32. Training and inference share this contract exactly.

**The local branch is sparse and target-specific.** Each pathology pools its
strongest eight evidence cells; it is not a full volumetric sequence model.
That limitation motivates any future ordered-slice experiment, rather than a
new B42 geometry variant.

**Training stops at exactly two epochs.** No B42 checkpoint is selected by a
labelled score.

**Expert labels never enter the gradient.** The 58 studies are diagnostic only
and no longer a clean future architecture-selection surface.

## Labels

Training uses the frozen report-only weak-supervision surface. The official gold
studies are excluded from gradients. B46's cross-fitted gold-anchor test did
not support changing that contract by a post-hoc weight sweep.

## Interface

```text
config/b42_constant_area_aspect_sparse.yaml
developments/src/rsna_knee/b42_constant_area_aspect_sparse_mil.py
developments/src/rsna_knee/b42_constant_area_aspect_sparse_training.py
developments/src/rsna_knee/b42_constant_area_aspect_sparse_submission_dualgpu_fast.py
```

The top-level `model/`, `training/`, `validation/`, and `testing/` packages are
the preserved B34 compatibility interface. They are not the B42 submission
entrypoint.
