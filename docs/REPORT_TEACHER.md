# Fold-safe report teacher benchmark

> Current campaign status is summarized in [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).

The competition test CSV does not provide report text, so reports are **training supervision only**. Final submission inference remains MRI-only.

## Benchmark components

The v0.5 teacher benchmark combined:

1. deterministic multilingual clinical-rule states with fold-safe empirical calibration;
2. word TF-IDF (1-2 grams) + target-specific balanced logistic regression;
3. character TF-IDF (3-5 grams) + target-specific balanced logistic regression.

No external model, external corpus, LLM or pretrained text encoder was used.

## Leakage contract

For outer gold fold `k`:

- outer reports/labels are excluded from rule calibration;
- their report groups are excluded from TF-IDF vocabulary/IDF fitting;
- their labels are excluded from text classifiers;
- the remaining two gold folds are cross-fitted;
- target-specific component weights/calibration are selected only from inner cross-fit evidence;
- the final fold-`k` teacher is then fitted on all non-outer gold.

This makes the benchmark a legitimate test of whether supervised text classification from the tiny gold set is strong enough to serve as an MRI teacher.

## Final result

```text
macro AUC = 0.4924496600
95% CI    = [0.4396044171, 0.5460505497]
```

Per-target AUC:

| Target | AUC |
|---|---:|
| ACL | 0.5723 |
| MCL | 0.3991 |
| Medial Meniscus | 0.5108 |
| Lateral Meniscus | 0.3764 |
| Medial OA | 0.5209 |
| Lateral OA | 0.4449 |
| PF OA | 0.4607 |
| Effusion | 0.4584 |
| Synovitis | 0.4851 |
| Baker's | 0.5453 |
| Contusion | 0.4022 |
| Fracture | 0.7333 |

## Decision

**Rejected as a general Stage-1 teacher.** The pooled result is near chance and the text classifiers see only roughly 38-40 non-outer gold reports per fold.

The exported fold-specific pseudo-label files are retained for audit/research, but they do not globally replace the conservative rule-teacher path.

Fracture was the only clearly strong target in this benchmark. Post-hoc target-specific use is not adopted from these outer results without a new controlled experiment.

## Why B5 still uses reports

Rejecting this teacher does **not** mean reports contain no useful information. It means converting report text into 12 supervised probabilities from only 58 labelled reports was not reliable enough.

B5 takes a different approach:

```text
4,349 report-only competition studies
report -> TF-IDF -> TruncatedSVD semantic embedding
MRI    -> strong SSL ConvNeXt representation
       -> image-report alignment
```

B5 never requires target labels for the report-only studies and excludes all 58 gold cases from representation training. Thus B5 tests whether report semantics can shape the MRI representation without relying on the failed 12-target report classifier.

**B5 is currently running; no B5 AUC is available yet.**

## Reproduce this benchmark

```bash
python -m rsna_knee.report_teacher_cli \
  --train-csv "$DATA_ROOT/train.csv" \
  --out-dir runs/report_teacher \
  --n-bootstrap 2000
```

Outputs:

```text
runs/report_teacher/
  metrics.json
  oof.csv
  fold_assignments.csv
  fold0/teacher.json
  fold0/pseudo_labels.csv
  fold1/teacher.json
  fold1/pseudo_labels.csv
  fold2/teacher.json
  fold2/pseudo_labels.csv
```
