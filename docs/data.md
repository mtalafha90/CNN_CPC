# Dataset and DICOM handling

This document describes the data contract implemented by `CNN_CPC` and records the verified real-data audit performed on 2026-08-08.

## Official CSV contract

`train.csv` contains:

- `StudyInstanceUID`;
- `Report`;
- the 12 target columns.

`test.csv` requires `StudyInstanceUID`; report text is not required at inference.

The series CSVs contain:

- `StudyInstanceUID`;
- `SeriesInstanceUID`;
- `Fluid_Sensitive`;
- `Fat_Suppression`;
- `Anatomical_Plane`.

Duplicate study/series rows and missing UIDs are rejected.

## Verified real-data snapshot

The downloaded training metadata resolves to:

```text
training studies       4,407
fully gold-labeled        58
report-only studies    4,349
reports present        4,407
training series rows  24,371
```

All 58 gold studies have all 12 official target cells populated. There are no partially labeled rows in the current download.

The local test metadata supplied with this download contains 3 studies and 15 series rows; `sample_submission.csv` has the exact required study order and 12-target output schema.

## Gold target positives

Among the 58 official gold studies:

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

The imbalance across targets is why the production loss is macro-balanced rather than dominated by the most frequently supervised pathology.

## Nullable sequence metadata and repair

Missing `Fluid_Sensitive`, `Fat_Suppression`, or plane metadata can be backfilled from DICOM headers. The repair path independently uses:

- image orientation for anatomical plane;
- timing/weighting cues for fluid sensitivity;
- acquisition metadata for fat suppression.

A populated CSV field remains authoritative. If a field is still unknown after repair, the final series-routing score uses a conservative fallback.

The full real-data audit reported zero required metadata fields missing or repaired for the selected series surface used in the verification run.

## Six-stream routing

The model routes each study into up to six semantic slots:

```text
sagittal_fluid       sagittal_structural
coronal_fluid        coronal_structural
axial_fluid          axial_structural
```

Observed training coverage:

| Stream | Selected | Missing | Coverage |
|---|---:|---:|---:|
| sagittal_fluid | 4,401 | 6 | 99.86% |
| sagittal_structural | 4,294 | 113 | 97.44% |
| coronal_fluid | 4,250 | 157 | 96.44% |
| coronal_structural | 3,440 | 967 | 78.06% |
| axial_fluid | 4,407 | 0 | 100.00% |
| axial_structural | 1,094 | 3,313 | 24.82% |

Missing semantic streams are therefore normal, especially `axial_structural`. The model uses an explicit presence mask; it does not invent missing image content.

A notable property of the current metadata release is that `Fluid_Sensitive` and `Fat_Suppression` are perfectly coupled in the series table: the observed combinations are `(0,0)` and `(1,1)`. They therefore do not contribute independent routing information in this release, although the code keeps them as separate fields for robustness to other data.

## DICOM decoding

The reader supports common DICOM filename conventions and enhanced/multi-frame arrays. It applies:

1. physical slice ordering from orientation and position when available;
2. `InstanceNumber` fallback;
3. deterministic filename fallback;
4. `RescaleSlope` / `RescaleIntercept`;
5. `MONOCHROME1` inversion;
6. mixed-size center crop/pad;
7. finite 1st/99th percentile clipping and normalization.

Physical slice ordering uses the image-plane normal

```text
n = row_direction × column_direction
z_i = ImagePositionPatient_i · n
```

before sorting.

## Verified train preflight

The strict 24-study train preflight produced:

```text
studies_sampled               24
streams_possible             144
streams_selected             121
streams_missing               23
directories_found            121
streams_decoded              121
candidate_files            4,045
file_decode_failures           0
decoded_frames             4,045
metadata_fields_missing        0
metadata_fields_repaired       0
decode_failure_rate          0.0
file_decode_failure_rate     0.0
missing_stream_rate       0.1597
```

The 15.97% missing-stream rate reflects genuinely absent semantic slots, not decode errors.

## Verified complete local test preflight

All three locally supplied test studies were checked:

```text
studies_sampled                3
streams_possible              18
streams_selected              14
streams_missing                4
directories_found             14
streams_decoded               14
candidate_files              533
file_decode_failures           0
decoded_frames               533
decode_failure_rate          0.0
file_decode_failure_rate     0.0
missing_stream_rate       0.2222
```

Because the local test metadata has only three studies, this preflight covers the entire local test set.

## Verified full selected-series audit

The full CPU audit checked every selected training series:

```text
selected series checked             21,886
selected series decoded             21,886
selected series failed                   0
series with partial file failures        2
series above per-series failure gate     0
candidate DICOM files              732,556
failed DICOM files                       2
global file failure rate       2.7302e-06
configured global limit               0.02
configured per-series limit            0.20
```

The two partial cases were:

```text
Study 1.2.826.0.1.3680043.8.498.34685905030370793639196564723935583035
Series 1.2.826.0.1.3680043.8.498.39396636671446532796538574315802928348
35 / 36 frames decoded; 1 file failed

Study 1.2.826.0.1.3680043.8.498.37833587429731221455928642963031995680
Series 1.2.826.0.1.3680043.8.498.31160785238781353848727311763596115703
37 / 38 frames decoded; 1 file failed
```

Both series remain valid and are retained. Removing an entire study because one slice failed would discard substantially more valid information than the configured corruption policy requires.

## 2.5D representation

Each active series is normalized and sampled at distributed centers. A three-channel input is formed as:

```text
[I_(c-gap), I_c, I_(c+gap)]
```

Production defaults:

```yaml
n_slices: 16
image_size: 224
triplet_gap: 1
train_gap_choices: [1, 2]
center_jitter: 2
```

Training also uses mild MRI-compatible perturbations:

- rotation;
- translation;
- scale jitter;
- gamma variation;
- low-frequency multiplicative bias field;
- Gaussian noise;
- slice dropout.

Validation and inference disable stochastic augmentation.

## Worker-side decoded-volume cache

Persistent workers own a bounded LRU cache of raw decoded series:

```yaml
series_cache_mb_per_worker: 256
```

The cache changes only I/O cost, not preprocessing semantics.

## Validation/submission TTA parity

The production policy is:

```yaml
tta_center_offsets: [-1, 0, 1]
validation_tta_offsets: [-1, 0, 1]
weak_oof_tta_offsets: [0]
```

All requested TTA views are built after one DICOM decode. TTA therefore multiplies model forward passes, not DICOM reads.

`oof.csv` uses the submission-policy TTA. `oof_center.csv` is diagnostic only. Stage-1 weak OOF generation uses one center view to protect runtime.

## Report states and OA-specific parsing

Each report/target cell is assigned one of:

```text
positive
negated
uncertain
unmentioned
```

`unmentioned` receives zero direct report weight by default.

The OA parser is compartment-aware. The initial narrow lexicon produced no useful OA weak supervision, so the current implementation recognizes compartment-specific OA/arthrosis, cartilage loss, chondrosis/chondromalacia, osteophytes and related degenerative terminology while avoiding generic meniscal degeneration.

Verified post-fix state counts:

| Target | Positive | Negated | Uncertain | Unmentioned |
|---|---:|---:|---:|---:|
| Medial OA | 492 | 339 | 0 | 3,576 |
| Lateral OA | 409 | 387 | 0 | 3,611 |
| PF OA | 695 | 379 | 0 | 3,333 |

These pseudo-labels remain deliberately lower-confidence than official gold labels.

## Gold, inner, outer and weak cross-fit roles

For Stage-1 outer fold `k`:

- `outer_oof`: official gold held out for final fold evaluation;
- `inner_selection`: official gold used to select epoch count;
- `gold_train_selection`: trusted gold available in Phase A;
- `weak_oof`: non-gold `crossfit_fold=k`, excluded so the Stage-1 model can later predict it independently;
- `weak_train`: remaining report-supervised studies.

Phase B starts from a fresh model. It uses all non-outer gold while continuing to exclude the Stage-1 weak-OOF subset.

For Stage 2, only the corresponding safe fold-local Stage-1 `weak_oof.csv` may become an image teacher.

## Effective-supervision diagnostics

Every training fold writes:

```text
supervision_plan.json
training_diagnostics.json
```

They record per target:

- planned weight mass;
- actual weight mass;
- nonzero supervised cells;
- participating batches;
- ranking-pair counts.

These diagnostics are important because weak supervision is sparse and target-dependent.

## Training versus inference

Reports, report calibration and Stage-2 consensus are training-only. Final inference is MRI-only and reconstructs its model contract from self-describing checkpoints.