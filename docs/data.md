# Dataset and DICOM handling

> **Snapshot: 2026-08-09.** Data/audit facts below are verified on the downloaded competition release. Experiment scores live in [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md). B5 is currently running and uses only the 4,349 report-only training studies for representation learning.

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

## B4 frozen-feature data contract

B4 extracts deterministic features only after representation pretraining. The verified gold cache is:

```text
study_uids = (58,)
features   = (58, 6, 2304)
present    = (58, 6)
finite     = true
```

Mean, standard deviation and maximum are pooled per stream. Presence flags are explicitly available to the classical classifier.

## Report data and OA parsing

Each report/target cell is represented as `positive`, `negated`, `uncertain`, or `unmentioned`. Report silence receives zero direct report weight by default.

The compartment-aware OA parser produces:

| Target | Positive | Negated | Unmentioned |
|---|---:|---:|---:|
| Medial OA | 492 | 339 | 3,576 |
| Lateral OA | 409 | 387 | 3,611 |
| PF OA | 695 | 379 | 3,333 |

These remain weak labels rather than gold-equivalent labels.

## B5 report-only representation scope

B5 uses the 4,349 report-only studies for both MRI and report representation learning and excludes all 58 gold studies.

The report semantic space is fitted only on competition reports:

```text
normalized report
-> word TF-IDF (1-2 grams)
-> TruncatedSVD (<=256 dimensions)
-> L2-normalized semantic embedding
```

No external corpus or language model is used. Exact duplicate normalized report hashes are tracked and masked as false negatives in the report contrastive queue.

The B5 text branch is training-only; the downstream artifact is an MRI encoder and final inference remains MRI-only.

## Validation data roles

Gold outer folds are evaluation rows, not representation-training rows. The original neural Stage-1 path also separates inner selection, gold training and weak cross-fit roles. B4/B5 representation pretraining excludes all gold rows entirely.

Because the same 58 gold studies have now been used across multiple controlled model decisions, current experiment tables should be interpreted as model-selection cross-validation rather than an untouched independent test set.
