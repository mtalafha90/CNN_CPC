# Official dataset contract audit

## Purpose

Architecture development is paused after the successful frozen B34/PV2 mechanism test. Before defining B35 or making a hidden-test submission, the next work item is a descriptive audit of the competition data contract itself.

This audit is intentionally independent of model selection. It does not train a model, change B6, inspect PV1/PV2 target-wise outcomes for architecture design, or promote a checkpoint.

The official dataset description establishes several facts that this audit must verify against the actual local release:

- one row per study in `train.csv`;
- twelve binary target columns with only a small labelled subset;
- radiology reports may be written in several languages;
- one or more MRI series per study, described by anatomical plane, fluid sensitivity and fat suppression;
- variable slice counts with a long tail;
- heterogeneous scanner/protocol/intensity/resolution characteristics;
- report text is unavailable at test time;
- abnormality prevalence is not guaranteed to match between training and hidden evaluation.

## Phase 1 status: COMPLETE

The first tabular/report/series-metadata pass has been completed and is recorded in:

```text
developments/docs/DATASET_CONTRACT_AUDIT_PHASE1_RESULT.md
```

Key findings from the exact local release are:

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

## Important gold-label definition

The repository's current `gold_mask()` policy is:

```text
study is gold/development-labelled if ANY of the 12 official target columns is populated
```

Phase 1 resolved the actual release structure: all 58 rows selected by this policy have all twelve official labels populated, and there are no partially labelled studies. Thus `ANY` and `ALL` happen to select the same 58 studies for this exact release, although the implementation remains intentionally tolerant of partial labels.

## Phase 1: tabular and report-contract audit

Implemented in:

```text
developments/src/rsna_knee/dataset_contract_audit.py
```

Outputs:

```text
runs/dataset_contract_audit/
├── summary.json
├── official_label_by_target.csv
├── official_label_count_histogram.csv
├── officially_labeled_studies.csv
├── report_script_buckets.csv
├── series_per_study.csv
├── series_metadata_counts.csv
├── b6_coverage_by_report_script.csv      # when --b6-root is supplied
└── slice_counts_by_series.csv            # when --scan-slices is supplied
```

The audit records SHA-256 fingerprints of `train.csv` and `train_series.csv` so later summaries can be tied to the exact local data release.

### Official-label audit

For each target the script reports:

```text
number of populated official labels
positive cells
negative cells
positive prevalence among labelled cells
missing cells
```

At study level it reports the distribution of the number of official labels present (0 through 12) and writes the UIDs of all studies satisfying the repository gold definition. Report text itself is not copied into the audit artifacts.

### Report script audit

Because the official description says reports may be multilingual, the script groups reports by Unicode character system:

```text
Latin
Cyrillic
Arabic
Greek
Hebrew
Devanagari
Hangul
Hiragana
Katakana
CJK
Other
Mixed:<...>
Empty/no-letters
```

These are **script buckets, not language predictions**. A Latin-script report is not automatically English, and no institution/site is inferred from script.

When the frozen B6 root is supplied, the script reports for each script bucket:

```text
report-only studies
B6-active studies
studies with zero usable B6 cells
active-study fraction
usable B6 cells
positive B6 cells
negative B6 cells
usable cells per study
```

This is intended to reveal large coverage differences that could create report-language/script selection bias in the weak-supervision population. It is descriptive only and does not authorize B6 retuning from the same evaluation surfaces.

### Series audit

The supplied `train_series.csv` is audited for:

```text
series rows and unique series
studies with/without listed MRI series
series per study distribution
Anatomical_Plane distribution
Fluid_Sensitive distribution
Fat_Suppression distribution
plane × fluid × fat-suppression combinations
missing supplied metadata
```

With `--scan-slices`, the audit also counts actual `.dcm` files under every listed series directory and reports slice-count quantiles and the long tail above 100 and 200 slices.

## Phase 2: physical DICOM slice-count pass

Run from the repository root:

```bash
cd /media/talafha/Disk_1/CNN_CPC_current
conda activate rsna-knee
git pull --ff-only origin main

export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"
export B6_ROOT="/media/talafha/Disk_1/CNN_CPC/runs/b6_report_labels_v121"

PYTHONPATH=developments/src \
python -m rsna_knee.dataset_contract_audit \
  --data-root "$DATA_ROOT" \
  --b6-root "$B6_ROOT" \
  --out-root runs/dataset_contract_audit \
  --scan-slices
```

This reruns the cheap tabular checks and additionally scans every listed training series directory. The new artifact is:

```text
runs/dataset_contract_audit/slice_counts_by_series.csv
```

and `summary.json` is extended with the physical slice-count distribution.

## Decision boundary

The next model experiment is **not defined yet**. The data audit comes first. In particular, do not define B35 from target-specific PV2 outcomes.

The remaining questions are now:

1. What is the true long-tail distribution of DICOM slices per series relative to the fixed 16-position sampling policy?
2. How heterogeneous are scanner/header, resolution, orientation and intensity characteristics after the slice-count pass?
3. Can a separately versioned multilingual report-label candidate recover supervision from the currently almost-unused Greek- and Cyrillic-script report buckets while remaining accurate on the 58 official labels?
4. How should robustness to `Fluid_Sensitive != Fat_Suppression` be tested, given that no such combinations occur in the training metadata but the competition contract permits them?

Any future multilingual label extractor must be a new supervision experiment. B6 v1.2.1, PV1 and PV2 remain frozen historical evidence.
