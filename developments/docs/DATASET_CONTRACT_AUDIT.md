# Official dataset contract audit

## Purpose

Architecture development is paused after the successful frozen B34/PV2 mechanism test. Before defining B35 or making a hidden-test submission, the current work is a descriptive audit of the competition data contract and supervision coverage.

The audit does not train a model, alter frozen B6, use PV1/PV2 target-wise outcomes for architecture design, identify institutions, or promote a checkpoint.

## Phase 1 — COMPLETE: labels, reports and supplied series metadata

Recorded in `developments/docs/DATASET_CONTRACT_AUDIT_PHASE1_RESULT.md`.

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

Greek- and Cyrillic-script reports account for about 12.3% of report-only studies but only 40 of 14,123 usable B6 cells. `Fluid_Sensitive` and `Fat_Suppression` are perfectly redundant in the supplied training metadata even though the competition contract permits discordant values elsewhere.

## Phase 2 — COMPLETE: physical DICOM slice counts

Recorded in `developments/docs/DATASET_CONTRACT_AUDIT_PHASE2_RESULT.md`.

```text
median slices/series                      30
95th percentile                           45
99th percentile                          160
maximum                                  320
series >78 slices                        763  (3.13%)
series >100 slices                       709  (2.91%)
series >200 slices                        88  (0.36%)
```

The current 16-center gap-1 2.5D policy can touch up to 48 distinct source slices in one deterministic view and up to 78 across frozen TTA `[-1,0,+1]`. About 96.87% of series are fully coverable by that three-view policy. A global increase above 16 centers is therefore NO-GO from slice counts alone.

## Phase 3 — COMPLETE: DICOM scanner/header heterogeneity

Recorded in `developments/docs/DATASET_CONTRACT_AUDIT_PHASE3_RESULT.md`.

```text
known 2D series                         22329
known 3D series                           836
acquisition type missing                 1206
all >78-slice series are 3D
all >100-slice series are 3D
all >200-slice series are 3D
```

The extreme tail is a thin-slice 3D family rather than ordinary 2D MRI. All 24,371 representative training headers use Explicit VR Little Endian, so compressed DICOM pixel decoding remains a deployment capability that must be tested separately before hidden submission.

## Phase 4 — COMPLETE: B6 supervision × acquisition-domain intersection

Recorded in `developments/docs/DATASET_CONTRACT_AUDIT_PHASE4_RESULT.md`.

The B6-active and B6-inactive report-only populations differ strongly in MRI acquisition composition:

| Metric | B6 active | B6 inactive |
|---|---:|---:|
| studies | 3120 | 1229 |
| studies with any known 3D series | 614 (19.68%) | 41 (3.34%) |
| studies with any >78-slice series | 546 (17.50%) | 41 (3.34%) |
| studies with any >100-slice series | 524 (16.79%) | 37 (3.01%) |
| studies with any >200-slice series | 87 (2.79%) | 0 |

The manufacturer-family mixture also shifts substantially. At series level, B6-inactive studies are 51.77% Siemens and 42.41% Philips, whereas B6-active studies are 38.19% Siemens, 26.96% Philips, 26.58% GE and 7.53% Canon/Toshiba.

Script and acquisition domain are themselves associated in this exact release:

```text
Cyrillic report-only studies  217   Philips-family only; no known 3D
Greek report-only studies     318   Siemens-family only; no known 3D
Latin report-only studies    3814   655 studies with known 3D
```

This is not an institution/site inference. It does show that the weak-label coverage gap is also a supervision-domain selection problem.

Only six official gold studies directly anchor the two non-Latin script groups:

```text
Latin gold       52
Greek gold        3
Cyrillic gold     3
```

Therefore future script-specific performance claims must remain cautious.

## Phase 5 — report-supervision failure-mode inspection

Implemented in:

```text
developments/src/rsna_knee/report_supervision_gap_audit.py
```

This phase does not create a new labeler. It creates a deterministic local-only text sample so the actual report wording behind B6 failures can be inspected before any new supervision family is designed.

The sample contains:

```text
all non-Latin gold cases
Latin gold controls
Latin B6-inactive reports
Greek B6-inactive reports
Cyrillic B6-inactive reports
Latin B6-active controls
Greek B6-active controls
the single Cyrillic B6-active report
```

Selection within non-exhaustive strata is deterministic from a frozen SHA-256 salt. When the Phase-4 study-domain table is supplied, each sampled case also carries descriptive manufacturer/3D/long-series flags.

Run:

```bash
cd /media/talafha/Disk_1/CNN_CPC_current
conda activate rsna-knee
git pull --ff-only origin main

export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"
export B6_ROOT="/media/talafha/Disk_1/CNN_CPC/runs/b6_report_labels_v121"

PYTHONPATH=developments/src \
python -m rsna_knee.report_supervision_gap_audit \
  --data-root "$DATA_ROOT" \
  --b6-root "$B6_ROOT" \
  --domain-study-csv runs/dataset_domain_intersection_audit/study_domain_table.csv \
  --out-root runs/report_supervision_gap_audit \
  --per-stratum 12
```

Outputs:

```text
runs/report_supervision_gap_audit/
├── summary.json
├── sample_manifest.csv
└── report_text_sample.jsonl
```

`report_text_sample.jsonl` contains raw competition report text and is explicitly a **local analysis artifact**. Do not commit it to GitHub.

After inspection, any improved report label extractor must be a new versioned supervision family. A multilingual-only repair is not sufficient by itself because 733 Latin-script report-only studies are also B6-inactive.

## Current decision boundary

```text
globally increase 16 slice positions             NO-GO
create adaptive 3D sampler now                    NO-GO
modify frozen B6 v1.2.1                           NO-GO
define B35                                        NO-GO
inspect actual B6 report failure modes            GO
new separately versioned supervision candidate   only after Phase 5 inspection
verify compressed-DICOM codec capability          GO before hidden submission
```

B6 v1.2.1, PV1 and PV2 remain frozen historical evidence.
