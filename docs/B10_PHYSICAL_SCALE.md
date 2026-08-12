# B10 — physical-scale normalization

> **Status — 2026-08-12:** **COMPLETED / REJECTED GLOBALLY.** B10 gold macro AUC was `0.5523982721`; it did not replace B7.1. B13 later became the reused-gold development champion, and completed B15 now points the next diagnostic toward supervision-state quality.

## Scientific question

B10 tested whether scanner/protocol scale harmonization could improve generalization while preserving the B7.1 architecture and supervision recipe.

Historical B7.1 preprocessing resized every selected MRI slice to `224 x 224` regardless of native physical spacing. B10 inserted one label-free geometry stage:

```text
native DICOM pixels
-> plane-specific canonical in-plane PixelSpacing
-> center crop/pad to canonical physical FOV
-> 224 x 224 resize
-> unchanged B7.1 model
```

B9 strict routing was not inherited because B9 had already reduced global development performance.

## Frozen controls

B10 retained:

```text
B5 encoder initialization
B6 v1.2.1 supervision
3120 active weak-training studies
14123 usable cells
6871 positive / 7252 negative
historical B7.1 routing
16 sampled positions/stream
batch size 2
4 full epochs
encoder LR 1e-5
head LR 1e-4
TTA [-1,0,1]
5000 bootstrap replicates
zero gold gradients / early stopping
```

The geometry policy was derived from training DICOM metadata only; gold labels were not used to choose target spacing or FOV.

## Completed result

```text
B10 macro AUC   0.5523982721
B7.1 macro AUC  0.5644802945
```

B10 therefore provided no evidence to replace B7.1 globally. Target-wise physical-scale tuning was not pursued.

## Successor context through B15

```text
B12 gold  0.5660915179
B13 gold  0.6293565948  retained champion
B14 gold  0.6197914249  rejected globally
B15 gold  0.6209002783  no global improvement
```

B15 passed frozen weak-v2 teacher agreement strongly (`0.7319060415` versus matched control `0.5652498118`, paired median `+0.1675245839`) but did not improve expert-gold macro AUC. That divergence makes report-state/supervision quality a higher-priority diagnostic than revisiting B10 spacing/FOV choices.

## Decision

B10 remains **rejected as a global replacement**. Do not search target-specific spacing/FOV, combine B10 with B13/B15 target winners, or retune its geometry from reused gold.

In-plane resolution at higher pixel dimensions remains a distinct future question; B10 tested physical-scale normalization under the existing `224 x 224` output size.

Current status: [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md). Post-B15 roadmap: [`RAISING_AUC.md`](RAISING_AUC.md).