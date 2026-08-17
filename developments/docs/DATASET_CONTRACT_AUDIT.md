# Official dataset contract audit

## Purpose

Architecture development is paused after the successful frozen B34/PV2 mechanism test. Before defining B35 or making a hidden-test submission, the next work item is a descriptive audit of the competition data contract itself.

This audit is intentionally independent of model selection. It does not train a model, change B6, inspect PV1/PV2 target-wise outcomes for architecture design, or promote a checkpoint.

## Phase 1 status: COMPLETE

Recorded in:

```text
developments/docs/DATASET_CONTRACT_AUDIT_PHASE1_RESULT.md
```

Key findings from the exact local release:

```text
training studies                         4407
fully labelled studies                     58
partially labelled studies                  0
report-only studies                      4349
all reports non-empty                    4407

B6-active studies                        3120
B6 usable cells                         14123
Latin-script share of report-only       87.70%
Latin-script share of B6-active         98.75%
Latin-script share of usable B6 cells   99.72%

listed MRI series                       24371
median series/study                         5
maximum series/study                       14
```

The strongest Phase-1 warning is a large script-associated weak-supervision coverage shift: Greek- and Cyrillic-script report buckets account for about 12.3% of report-only studies but only 1.25% of B6-active studies and only 40 of 14,123 usable B6 cells. This is a coverage/selection-bias finding, not a language/site inference.

The supplied training metadata also show perfect redundancy between `Fluid_Sensitive` and `Fat_Suppression`: every listed series has either both flags true or both false. Hidden-test equivalence must not be assumed because the competition contract says the two flags are not necessarily equivalent for every case.

### Gold-label definition

The repository's current `gold_mask()` policy is:

```text
study is gold/development-labelled if ANY of the 12 official target columns is populated
```

Phase 1 resolved the actual release structure: all 58 rows selected by this policy have all twelve official labels populated, and there are no partially labelled studies. Thus `ANY` and `ALL` happen to select the same 58 studies for this exact release.

## Phase 2 status: COMPLETE

Recorded in:

```text
developments/docs/DATASET_CONTRACT_AUDIT_PHASE2_RESULT.md
```

The physical scan found every listed training series directory and no zero-DICOM series:

```text
series scanned                         24371
missing series directories                 0
zero-DICOM series                         0
mean slices/series                     33.61
median slices/series                      30
95th percentile                           45
97th percentile                           88
99th percentile                          160
maximum                                  320
series >78 slices                        763  (3.13%)
series >100 slices                       709  (2.91%)
series >200 slices                        88  (0.36%)
```

The 763 series above 78 slices contain 15.24% of all listed training DICOM files, so the upper tail is small by series count but large by image count.

### Current 16-position sampler relative to Phase 2

The current preprocessing places 16 distributed centers through each series and builds a gap-1 three-slice 2.5D triplet at each center. Therefore one deterministic evaluation view can touch up to 48 distinct source slice indices. Frozen TTA uses center offsets `[-1,0,+1]`; the union of all three views can touch up to 78 distinct source indices.

For the exact current center policy:

```text
one eval view fully covers series with <=48 slices
three-view TTA fully covers series with <=78 slices

series fully covered by one view        95.93%
series fully covered by frozen TTA      96.87%
mean per-series TTA source coverage     98.52%
slice-weighted TTA source coverage      92.03%
```

The Phase-2 decision is therefore:

```text
unconditional global increase above 16 positions    NO-GO
retain current 16-position policy                    YES for now
structurally inspect the >78-slice tail              GO
define B35 from slice counts alone                   NO-GO
```

The long tail includes repeated large counts such as 120, 128, 144, 160, 186 and 320 slices. Eighty-five series contain exactly 320 DICOM files. These should be explained by acquisition/header characteristics before any adaptive-sampling experiment is defined.

## Phase 1/2 implementation

Implemented in:

```text
developments/src/rsna_knee/dataset_contract_audit.py
```

Main outputs:

```text
runs/dataset_contract_audit/
├── summary.json
├── official_label_by_target.csv
├── official_label_count_histogram.csv
├── officially_labeled_studies.csv
├── report_script_buckets.csv
├── b6_coverage_by_report_script.csv
├── series_per_study.csv
├── series_metadata_counts.csv
└── slice_counts_by_series.csv
```

## Phase 3: representative DICOM-header heterogeneity audit

Implemented in:

```text
developments/src/rsna_knee/dataset_header_audit.py
```

Phase 3 reads one representative DICOM header per listed training series using `stop_before_pixels=True`. It does not decode image pixels. It records and aggregates:

```text
manufacturer and scanner model
magnetic field strength
MR acquisition type (when present)
transfer syntax
Rows / Columns
PixelSpacing and in-plane FOV
SliceThickness and SpacingBetweenSlices
NumberOfFrames
photometric/pixel representation
ImageOrientationPatient-derived closest anatomical plane
obliquity relative to patient axes
agreement with supplied Anatomical_Plane
```

The output also stratifies the slice-count tail (`<=48`, `49-78`, `79-100`, `101-200`, `>200`) by supplied plane/flags and available scanner/acquisition metadata. Acquisition categories are descriptive metadata only and must not be interpreted as patient or institutional identity.

Run:

```bash
cd /media/talafha/Disk_1/CNN_CPC_current
conda activate rsna-knee
git pull --ff-only origin main

export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"

PYTHONPATH=developments/src \
python -m rsna_knee.dataset_header_audit \
  --data-root "$DATA_ROOT" \
  --out-root runs/dataset_header_audit
```

Expected artifacts:

```text
runs/dataset_header_audit/
├── header_summary.json
├── header_by_series.csv
├── header_categorical_counts.csv
├── slice_tail_header_profile.csv
└── orientation_vs_supplied_plane.csv
```

## Decision boundary

The next model experiment remains **undefined**. Do not define B35 from PV1/PV2 target outcomes, the Phase-2 long tail, or the Phase-3 scanner categories alone.

After Phase 3, the remaining high-value data questions are:

1. Do >78-, >100- and >200-slice series represent distinct 3D/thin-slice acquisition families that justify an adaptive sampling hypothesis?
2. Are resolution, field strength, orientation or transfer-syntax distributions broad enough to expose a preprocessing robustness gap?
3. Can a separately versioned multilingual report-label candidate recover supervision from the almost-unused Greek- and Cyrillic-script report buckets while remaining accurate on the 58 official labels?
4. How should robustness to `Fluid_Sensitive != Fat_Suppression` be tested, given that no discordant examples occur in the training metadata but the competition contract permits them?

Any future multilingual label extractor must be a new supervision experiment. B6 v1.2.1, PV1 and PV2 remain frozen historical evidence.
