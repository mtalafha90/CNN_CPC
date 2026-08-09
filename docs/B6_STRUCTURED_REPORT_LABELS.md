# B6 — structured multilingual report labels

> **Status — 2026-08-10:** **V1.1 IMPLEMENTED / RE-AUDIT PENDING.** The first real v1.0 corpus audit produced 13,823 high-confidence target cells across 4,349 report-only studies (26.5% of all possible study-target cells). Review of the uncertainty queue exposed systematic structural-target scope errors, so v1.1 narrows evidence to target-local context and changes conflict resolution before B7 is allowed to train.

## Goal

B5 showed that report-aligned representation learning improves the standalone point estimate, but the final gold classifier still learns from only 58 labelled studies. B6 changes that bottleneck by converting each training report into twelve auditable target-level weak labels.

B6 produces, for every target:

```text
positive
negated
uncertain
unmentioned
```

plus a fixed soft probability, confidence weight, reason and evidence snippet.

The 58 gold studies are retained only for later audit. They are excluded from the B6 weak-training export by construction.

## Leakage contract

The B6 exporter:

- uses only competition `train.csv` reports;
- uses no external model;
- uses no external language resource or external data;
- does not fit TF-IDF, a language model, thresholds or calibration on gold labels;
- does not convert report silence into a negative;
- excludes every gold row from `training_targets.csv`;
- preserves uncertain/conflicting mentions at low confidence instead of forcing binary labels.

## V1.0 real-corpus audit

The first run covered all 4,407 training reports, with 58 gold rows kept for audit only and 4,349 report-only rows exported for weak supervision.

At `min_confidence=0.75`, v1.0 produced:

```text
high-confidence usable cells = 13,823
possible report-only cells   = 4,349 x 12 = 52,188
usable fraction overall      = 26.5%
```

Target-wise usable cells:

| Target | Positive | Negative | Usable | Fraction |
|---|---:|---:|---:|---:|
| ACL | 500 | 1,067 | 1,567 | 36.0% |
| MCL | 213 | 1,135 | 1,348 | 31.0% |
| Medial Meniscus | 1,084 | 447 | 1,531 | 35.2% |
| Lateral Meniscus | 542 | 1,046 | 1,588 | 36.5% |
| Medial OA | 484 | 334 | 818 | 18.8% |
| Lateral OA | 401 | 383 | 784 | 18.0% |
| PF OA | 683 | 372 | 1,055 | 24.3% |
| Effusion | 1,333 | 756 | 2,089 | 48.0% |
| Synovitis | 398 | 16 | 414 | 9.5% |
| Baker's | 556 | 476 | 1,032 | 23.7% |
| Contusion | 381 | 465 | 846 | 19.5% |
| Fracture | 203 | 548 | 751 | 17.3% |

This is enough coverage to justify B6, but the v1.0 review queue was dominated by structural targets, especially ACL.

## Why v1.1 was required

The v1.0 review queue showed that many obvious ACL abnormalities were incorrectly assigned `uncertain/conflicting_or_mixed_evidence`. Common patterns included:

```text
ACL: high grade tear
```

appearing near unrelated phrases such as:

```text
patellar tendon: intact
```

or a definite finding repeated with a weaker statement:

```text
complete ACL tear
possible ACL tear
```

V1.0 searched normality and uncertainty across too much of the surrounding clause and treated `definite + uncertain` as a conflict.

## V1.1 changes

V1.1 is a corpus-driven parser correction, not gold-label tuning.

Changes:

1. **Target-local evidence window.** Normality, uncertainty and structural pathology are evaluated near the target mention rather than across the full broad clause.
2. **Arrow delimiters.** `>` is treated as a report-item boundary in addition to periods and semicolons.
3. **Pathology dominates unrelated normality.** Phrases such as `ACL grade 1 sprain with intact fibers` remain positive.
4. **Direct negated structural findings.** Forms such as `ACL: no tear` are detected explicitly.
5. **Indication/query uncertainty.** `assess for`, `evaluate for`, `concern for`, `history of`, `r/o`, `rule out`, `query` and `quid` are low-confidence unless a definite diagnostic statement is also present.
6. **Uncertain duplicates no longer cancel definite evidence.** `complete ACL tear` plus `possible ACL tear` resolves to the definite positive.
7. **Only opposing definite states conflict.** A report must contain both definite positive and definite negative evidence to enter `conflicting_definite_evidence`.
8. **Duplicate observations are deduplicated.** Repeated findings/impression text does not inflate conflict logic.
9. **Audit reason counts.** `audit.json` now reports parser-reason counts for mentioned cells.

Regression tests cover all of these structural-target cases.

## Multilingual rule layer

B6 reuses the established report-normalization and compartment-aware OA logic, then adds an auditable structured evidence layer with accent-insensitive aliases and target-local context handling.

The vocabulary covers common forms across English plus several languages represented in the existing competition-oriented parser, including Spanish/Portuguese, French, German, Dutch, Italian, Turkish and South-Slavic forms.

This remains an auditable rule system. It is not claimed to be the final multilingual solution before repeated real-corpus review.

## Fixed soft-label contract

| State | Probability | Confidence |
|---|---:|---:|
| positive | `0.97` | `0.90` |
| negated | `0.03` | `0.90` |
| uncertain | `0.50` | `0.25` |
| unmentioned | `0.50` | `0.00` |
| conflicting definite evidence | `0.50` | `0.20` |

The later MRI loss must use the confidence column. Unmentioned cells have zero weight.

## Run B6 v1.1

Pull the latest code and reinstall the editable package:

```bash
cd /media/talafha/Disk_1/CNN_CPC
git pull
python -m pip install -e .
pytest -q tests/test_b6_report_labels.py
```

Keep the v1.0 output for comparison and write v1.1 to a new directory:

```bash
rsna-knee-b6 \
  --train-csv "$DATA_ROOT/train.csv" \
  --out-root runs/b6_report_labels_v11 \
  --min-confidence 0.75 \
  --max-review 1000
```

## Outputs

```text
runs/b6_report_labels_v11/
├── structured_labels.csv
├── training_targets.csv
├── review_queue.csv
├── audit.json
└── policy.json
```

`training_targets.csv` contains report-only studies only; all gold rows are excluded.

## V1.1 decision gate

Do not start B7 yet. Compare v1.1 with v1.0 first.

We want:

- a large reduction in false ACL/MCL/meniscus conflicts;
- high-confidence structural positives to increase when the old uncertainty was spurious;
- no collapse of reliable negatives;
- review queue examples that are genuinely ambiguous rather than obvious tears/sprains;
- stable OA, effusion, Baker's, contusion and fracture distributions unless a real scope correction affects them.

Inspect:

```bash
cat runs/b6_report_labels_v11/audit.json

python - <<'PY'
import pandas as pd
p = pd.read_csv('runs/b6_report_labels_v11/review_queue.csv')
print(p['target'].value_counts())
print(p[['StudyInstanceUID','target','confidence','reason','evidence']].head(100).to_string(index=False))
PY
```

Once v1.1 passes this corpus gate, the next B6 step is a **small gold audit used only to estimate weak-label precision**, not to fit B6 thresholds. Only after that will B7 consume the B6 training targets.
