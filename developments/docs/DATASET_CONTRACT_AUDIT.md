# Official dataset contract audit

## Purpose

Architecture development is paused after the successful frozen B34/PV2 mechanism test. Before defining B35 or making a hidden-test submission, the next work item is a descriptive audit of the competition data contract itself.

This audit is intentionally independent of model selection. It does not train a model, change B6, inspect PV1/PV2 target-wise outcomes for architecture design, or promote a checkpoint.

## Phase 1 status: COMPLETE — labels, reports and supplied series metadata

Recorded in:

```text
developments/docs/DATASET_CONTRACT_AUDIT_PHASE1_RESULT.md
```

Key findings:

```text
training studies                         4407
fully labelled studies                     58
partially labelled studies                  0
report-only studies                      4349
B6-active report-only studies            3120
B6 usable cells                         14123
Latin-script share of report-only       87.70%
Latin-script share of B6-active         98.75%
Latin-script share of usable B6 cells   99.72%
listed MRI series                       24371
```

Greek- and Cyrillic-script reports account for about 12.3% of report-only studies but only 40 of 14,123 usable B6 cells. This is a script-associated coverage/selection finding, not a language or institution inference.

All 58 rows selected by the repository `gold_mask()` happen to have all twelve official labels populated in this exact release; there are no partially labelled rows.

`Fluid_Sensitive` and `Fat_Suppression` are perfectly redundant in the supplied training metadata even though the competition contract permits discordant values in other data.

## Phase 2 status: COMPLETE — physical DICOM slice counts

Recorded in:

```text
developments/docs/DATASET_CONTRACT_AUDIT_PHASE2_RESULT.md
```

```text
series scanned                         24371
missing series directories                 0
zero-DICOM series                         0
mean slices/series                     33.61
median slices/series                      30
95th percentile                           45
99th percentile                          160
maximum                                  320
series >78 slices                        763  (3.13%)
series >100 slices                       709  (2.91%)
series >200 slices                        88  (0.36%)
```

The current 16-center, gap-1 2.5D sampler can touch up to 48 distinct source slices in one evaluation view and up to 78 across frozen TTA `[-1,0,+1]`. Approximately 96.87% of series are fully coverable by that three-view policy. A global increase above 16 centers is therefore **NO-GO** from slice counts alone.

## Phase 3 status: COMPLETE — DICOM scanner/header heterogeneity

Recorded in:

```text
developments/docs/DATASET_CONTRACT_AUDIT_PHASE3_RESULT.md
```

Every one of the 24,371 representative headers was read successfully and every ImageOrientationPatient-derived canonical plane agreed with the supplied `Anatomical_Plane`.

The decisive long-tail result is:

```text
known 2D series                          22329
known 3D series                            836
acquisition type missing                  1206

all known 3D series have >48 slices
known 3D >78 slices                 763 / 836  (91.27%)
known 3D >100 slices                709 / 836  (84.81%)
known 3D >200 slices                 88 / 836  (10.53%)

all >78-slice series are 3D
all >100-slice series are 3D
all >200-slice series are 3D
```

The >200-slice tail is a thin-slice 3D family: 85/88 are sagittal, median SliceThickness is about 0.8 mm, and median available SpacingBetweenSlices is about 0.4 mm. Therefore the extreme long tail is not ordinary 2D MRI with more slices.

The release is also broad in physical geometry: PixelSpacing spans roughly 0.073–1.172 mm, matrix sizes span 160–1280 rows and 160–1444 columns, and obliquity reaches about 41 degrees while still agreeing with the supplied closest anatomical plane.

### Transfer-syntax deployment warning

All 24,371 representative training headers use:

```text
1.2.840.10008.1.2.1  Explicit VR Little Endian
```

Thus the local training release does not exercise compressed DICOM pixel decoding even though the competition contract permits JPEG Lossless and JPEG 2000 in hidden data. Codec capability must be tested separately before submission.

## Phase 4: B6 supervision × MRI acquisition-domain intersection

Implemented in:

```text
developments/src/rsna_knee/dataset_domain_intersection_audit.py
```

This is still a descriptive data audit. It asks whether the studies that receive usable frozen B6 supervision occupy the same MRI acquisition domain as report-only studies receiving zero usable B6 cells.

It combines:

```text
train.csv report script bucket
gold / report-only status
frozen B6 active / inactive status
Phase-3 header_by_series.csv
manufacturer family
field strength
2D / 3D acquisition type
series count
>78 / >100 / >200-slice 3D tail membership
```

No model outputs or PV1/PV2 performance are read.

Run:

```bash
cd /media/talafha/Disk_1/CNN_CPC_current
conda activate rsna-knee
git pull --ff-only origin main

export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"
export B6_ROOT="/media/talafha/Disk_1/CNN_CPC/runs/b6_report_labels_v121"

PYTHONPATH=developments/src \
python -m rsna_knee.dataset_domain_intersection_audit \
  --data-root "$DATA_ROOT" \
  --b6-root "$B6_ROOT" \
  --header-csv runs/dataset_header_audit/header_by_series.csv \
  --out-root runs/dataset_domain_intersection_audit
```

Expected artifacts:

```text
runs/dataset_domain_intersection_audit/
├── domain_summary.json
├── study_domain_table.csv
├── cohort_summary.csv
├── cohort_series_categories.csv
└── script_b6_domain_crosstab.csv
```

This phase will establish whether the large script-associated B6 coverage gap is also an MRI scanner/protocol selection gap. That matters before designing a multilingual supervision candidate: recovering previously unused reports may simultaneously recover underrepresented acquisition domains.

## Decision boundary

The next predictive experiment remains **undefined**.

Current data-level decisions are:

```text
globally increase 16 slice positions             NO-GO
create adaptive 3D sampler now                    NO-GO; mechanism plausible but not yet validated
modify frozen B6 v1.2.1                           NO-GO
finish B6 × acquisition-domain intersection       GO
verify compressed-DICOM codec capability          GO before hidden submission
define multilingual supervision family            only after Phase 4 composition audit
```

Any future multilingual extractor must be a separately versioned supervision experiment. B6 v1.2.1, PV1 and PV2 remain frozen historical evidence.