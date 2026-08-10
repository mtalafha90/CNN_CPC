# Dataset and DICOM handling

> **Snapshot: 2026-08-10.** Data/audit facts below are verified on the downloaded competition release. Experiment scores live in [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md). B6 v1.2.1 is the frozen structured weak-label source; B7.1 is the current development leader; B8 spatial-anatomy training is in progress.

## CSV contract

`train.csv` contains `StudyInstanceUID`, `Report`, and the 12 targets. `test.csv` contains study UIDs and does not provide the report supervision used during training.

Series metadata contain:

- `StudyInstanceUID`;
- `SeriesInstanceUID`;
- `Fluid_Sensitive`;
- `Fat_Suppression`;
- `Anatomical_Plane`.

Duplicate study/series rows and missing UIDs are rejected.

## Verified release snapshot

```text
training studies       4,407
fully gold-labelled       58
report-only studies    4,349
reports present        4,407
training series rows  24,371
```

All 58 gold studies have all 12 official target cells populated. There are no partially gold-labelled rows in the current download.

The local test metadata currently contains 3 studies and 15 series rows.

## Gold class counts

| Target | Positive | Negative |
|---|---:|---:|
| ACL | 24 | 34 |
| MCL | 9 | 49 |
| Medial Meniscus | 26 | 32 |
| Lateral Meniscus | 23 | 35 |
| Medial OA | 15 | 43 |
| Lateral OA | 11 | 47 |
| PF OA | 21 | 37 |
| Effusion | 35 | 23 |
| Synovitis | 27 | 31 |
| Baker's | 12 | 46 |
| Contusion | 19 | 39 |
| Fracture | 18 | 40 |

## Six-stream routing

```text
sagittal_fluid       sagittal_structural
coronal_fluid        coronal_structural
axial_fluid          axial_structural
```

Observed coverage:

| Stream | Selected | Missing | Coverage |
|---|---:|---:|---:|
| sagittal_fluid | 4,401 | 6 | 99.86% |
| sagittal_structural | 4,294 | 113 | 97.44% |
| coronal_fluid | 4,250 | 157 | 96.44% |
| coronal_structural | 3,440 | 967 | 78.06% |
| axial_fluid | 4,407 | 0 | 100.00% |
| axial_structural | 1,094 | 3,313 | 24.82% |

Missing streams are normal and represented by explicit presence masks. They are never synthesized.

In this release `Fluid_Sensitive` and `Fat_Suppression` are perfectly coupled in the metadata table, although the implementation keeps them separate for robustness.

## Metadata repair

Missing plane/sequence metadata can be backfilled from DICOM headers using orientation, timing/weighting and acquisition cues. Populated CSV fields remain authoritative.

The verified selected-series audit required no metadata repair for the production-selected surface.

## DICOM decoding

The reader supports common DICOM naming and enhanced/multi-frame arrays. Processing includes:

1. physical slice ordering from orientation/position;
2. `InstanceNumber` fallback;
3. deterministic filename fallback;
4. rescale slope/intercept;
5. `MONOCHROME1` inversion;
6. mixed-size center crop/pad;
7. finite percentile clipping/normalization.

Physical ordering uses the image-plane normal:

```text
n = row_direction x column_direction
z_i = ImagePositionPatient_i . n
```

## Verified train preflight

```text
studies sampled               24
streams possible             144
streams selected             121
streams decoded              121
candidate files            4,045
file failures                  0
missing stream rate       0.1597
```

## Verified complete local-test preflight

```text
studies sampled                3
streams possible              18
streams selected              14
streams decoded               14
candidate files              533
file failures                  0
missing stream rate       0.2222
```

## Verified full selected-series audit

```text
selected series checked             21,886
selected series decoded             21,886
selected series failed                   0
series with partial file failures        2
candidate DICOM files              732,556
failed DICOM files                       2
global file failure rate       2.7302e-06
per-series failure limit               0.20
global failure limit                   0.02
```

Two selected series each contain one unreadable file (35/36 and 37/38 frames decoded). Both remain valid and are retained.

## 2.5D representation

Each active series is normalized and sampled into triplets:

```text
[I_(c-gap), I_c, I_(c+gap)]
```

Typical production settings:

```yaml
n_slices: 16
image_size: 224
triplet_gap: 1
train_gap_choices: [1, 2]
center_jitter: 2
```

Training may add mild affine, gamma, bias-field, Gaussian-noise and slice-dropout perturbations. Validation/inference are deterministic apart from predeclared TTA.

## Strong SSL data scope

The completed strong MRI SSL checkpoint uses only the 4,349 non-gold competition studies. The 58 gold studies are excluded from representation pretraining.

Checkpoint:

```text
runs/ssl_strong/ssl_encoder.pt
```

## B4/B5 frozen-feature data contract

B4/B5 extract deterministic features only after representation pretraining. The verified gold cache contract is:

```text
study_uids = (58,)
features   = (58, 6, 2304)
present    = (58, 6)
finite     = true
```

Mean, standard deviation and maximum are pooled per stream. Presence flags are explicitly available to the classical classifier.

B5 representation training uses the 4,349 report-only studies and excludes all 58 gold studies.

## B6 structured report data contract

Each report/target cell is represented as:

```text
positive
negated
uncertain
unmentioned
```

Report silence is not a negative.

Frozen B6 v1.2.1 report-only training export:

```text
report-only rows                  4349
active studies                    3120
inactive zero-usable studies      1229
usable cells                     14123
positive cells                    6871
negative cells                    7252
confidence threshold              0.75
gold rows in training_targets.csv    0
```

Per-target usable counts:

| Target | Positive | Negative | Usable |
|---|---:|---:|---:|
| ACL | 572 | 1,089 | 1,661 |
| MCL | 271 | 1,089 | 1,360 |
| Medial Meniscus | 1,126 | 536 | 1,662 |
| Lateral Meniscus | 448 | 1,182 | 1,630 |
| Medial OA | 484 | 334 | 818 |
| Lateral OA | 402 | 382 | 784 |
| PF OA | 682 | 372 | 1,054 |
| Effusion | 1,338 | 757 | 2,095 |
| Synovitis | 399 | 17 | 416 |
| Baker's | 557 | 476 | 1,033 |
| Contusion | 389 | 466 | 855 |
| Fracture | 203 | 552 | 755 |

B6 is frozen after its gold audit. Do not alter parser rules or confidence thresholds from later B7/B8 gold outcomes.

## B7/B7.1 training data contract

B7/B7.1 use only the 3,120 report-only studies with at least one usable B6 cell. In the audited run:

```text
active studies before MRI filter  3120
studies without selected MRI         0
training studies                  3120
training usable cells            14123
```

B7-v1 sampled only 1,000 study draws/epoch because of the 500-batch cap.

B7.1 uses:

```text
batch size          2
batches/epoch    1560
study draws/epoch 3120
epochs              4
```

so every complete epoch covers the entire active weak-training pool once.

B7.1 development result:

```text
macro AUC = 0.5644802945
```

## B8 spatial-token data contract

B8 keeps exactly the same study/series/B6 supervision pool as B7.1. The data change is not a new cohort or new label source; it is the representation retained from each sampled MRI slice.

```text
B7.1: 1 globally pooled ConvNeXt token/slice
B8:   2x2 ConvNeXt spatial grid = 4 tokens/slice
```

For the standard six streams and 16 sampled slices:

```text
B7.1 memory tokens/study = 96
B8 memory tokens/study   = 384
```

No extra MRI studies, external images, external labels or external language resources are introduced by B8.

B8 real-data training is currently in progress. No B8 gold-development result is recorded in this data document.

## Validation data roles

The 58 gold studies have supported multiple controlled method decisions. They are excluded from B5 representation training and B6 weak-training export, and they do not enter B7/B7.1/B8 gradients or early stopping.

However, the B6 audit and later model choices have used the same 58 studies for development decisions. Therefore current experiment tables should be interpreted as **model-selection cross-validation/development estimates**, not an untouched independent test set.
