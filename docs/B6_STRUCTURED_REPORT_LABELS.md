# B6 — structured multilingual report labels

> **Status — 2026-08-10:** **IMPLEMENTED / CORPUS AUDIT PENDING.** B6 is the first post-B5 step toward using all 4,349 report-only studies as target-level supervision. The initial implementation is competition-only and does not fit or calibrate on the 58 gold labels.

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

The initial B6 exporter:

- uses only competition `train.csv` reports;
- uses no external model;
- uses no external language resource or external data;
- does not fit TF-IDF, a language model, thresholds or calibration on gold labels;
- does not convert report silence into a negative;
- excludes every gold row from `training_targets.csv`;
- preserves uncertain/conflicting mentions at low confidence instead of forcing binary labels.

This makes the first B6 corpus audit independent of the 58 labels.

## Multilingual rule layer

B6 reuses the established report-normalization and compartment-aware OA logic, then adds a stricter structured evidence layer with accent-insensitive aliases and clause-local context handling.

The vocabulary covers common forms across English plus several languages represented in the existing competition-oriented parser, including Spanish/Portuguese, French, German, Dutch, Italian, Turkish and South-Slavic forms.

B6 explicitly handles:

- target/anatomy aliases;
- structural abnormality language for ACL/MCL/menisci;
- normal/intact language;
- negation;
- uncertainty such as possible/suspected/cannot-exclude forms;
- contradictory evidence;
- compartment-aware OA findings.

This is deliberately an auditable rule system. It is not claimed to be the final multilingual solution before the real corpus audit is inspected.

## Fixed soft-label contract

The first B6 pass uses fixed values rather than gold-fitted calibration:

| State | Probability | Confidence |
|---|---:|---:|
| positive | `0.97` | `0.90` |
| negated | `0.03` | `0.90` |
| uncertain | `0.50` | `0.25` |
| unmentioned | `0.50` | `0.00` |
| conflicting evidence | `0.50` | `0.20` |

The later MRI loss must use the confidence column. In particular, unmentioned cells have zero weight.

## Run B6

After pulling the latest repository and reinstalling the editable package:

```bash
cd /media/talafha/Disk_1/CNN_CPC
git pull
python -m pip install -e .

rsna-knee-b6 \
  --train-csv "$DATA_ROOT/train.csv" \
  --out-root runs/b6_report_labels \
  --min-confidence 0.75 \
  --max-review 1000
```

Equivalent module invocation:

```bash
python -m rsna_knee.b6_report_labels \
  --train-csv "$DATA_ROOT/train.csv" \
  --out-root runs/b6_report_labels
```

## Outputs

```text
runs/b6_report_labels/
├── structured_labels.csv
├── training_targets.csv
├── review_queue.csv
├── audit.json
└── policy.json
```

### `structured_labels.csv`

Contains all training studies for audit, including gold rows. For each target it stores:

```text
<Target>                 fixed soft probability
<Target>__confidence     weak-label training weight
<Target>__state          positive/negated/uncertain/unmentioned
<Target>__mentioned      explicit report evidence flag
<Target>__reason         parser decision reason
<Target>__evidence       normalized local evidence
```

### `training_targets.csv`

This is the B6 weak-supervision artifact intended for B7. It contains **report-only studies only**; gold rows are excluded.

### `review_queue.csv`

Prioritizes mentioned-but-uncertain and conflicting cells. This is the main manual audit surface for the first B6 pass.

### `audit.json`

Reports target-wise counts for positive, negated, uncertain and unmentioned cells, plus high-confidence usable coverage.

## First decision gate

Do not start B7 from B6 labels until the first corpus audit is inspected.

For each target we want to inspect:

1. high-confidence positive count;
2. high-confidence negative count;
3. fraction of report-only studies receiving a usable label;
4. uncertain/conflict count;
5. representative review-queue errors;
6. obvious language-specific misses.

The goal is **precision first**, not maximum coverage. A smaller set of reliable report labels is more useful than forcing all 4,349 × 12 cells to become binary.

## Commands to inspect the first run

```bash
cat runs/b6_report_labels/audit.json
cat runs/b6_report_labels/policy.json

python - <<'PY'
import pandas as pd
p = pd.read_csv('runs/b6_report_labels/review_queue.csv')
print(p['target'].value_counts())
print(p[['StudyInstanceUID','target','reason','evidence']].head(50).to_string(index=False))
PY
```

Also run the B6 tests:

```bash
pytest -q tests/test_b6_report_labels.py
```

## What comes after the audit

B6 should be improved only from concrete corpus failures: missed language forms, false negation scope, uncertainty scope, compartment confusion or target-definition ambiguity. Once the structured labels pass the audit, B7 will train a B5-initialized MRI student using these confidence-weighted labels across the report-only corpus.
