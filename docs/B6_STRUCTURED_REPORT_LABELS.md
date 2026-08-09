# B6 — structured multilingual report labels

> **Status — 2026-08-10:** **V1.2.1 CORPUS-FROZEN / GOLD AUDIT NEXT.** B6 now produces 14,123 high-confidence target cells across 4,349 report-only studies, covering 27.06% of the 52,188 possible study-target cells. Parser changes are closed before the 58-gold audit so that the gold labels are not used to tune B6.

## Goal

B5 improved the MRI representation but still left the downstream classifier dependent on only 58 fully labelled studies. B6 converts competition reports into twelve target-level weak labels that can later supervise B7.

For each study/target B6 exports:

```text
positive
negated
uncertain
unmentioned
```

plus a fixed soft probability, confidence weight, decision reason and local evidence snippet.

The 58 gold studies are never included in `training_targets.csv`.

## Leakage and tuning contract

B6:

- uses only competition `train.csv` reports;
- uses no external model or external data;
- does not fit report thresholds or calibration on gold labels;
- never maps unmentioned findings to negative;
- excludes all gold rows from the B6 weak-training artifact;
- assigns low/zero training weight to uncertain/unmentioned cells;
- was refined only from report-text review before the gold audit;
- is **frozen at v1.2.1 before inspecting gold-label agreement**.

After the gold audit, parser changes are not allowed. The audit is diagnostic only and must not become another round of report-rule tuning.

## Fixed soft-label contract

| State | Probability | Confidence |
|---|---:|---:|
| positive | `0.97` | `0.90` |
| negated | `0.03` | `0.90` |
| uncertain | `0.50` | `0.25` |
| unmentioned | `0.50` | `0.00` |
| conflicting definite evidence | `0.50` | `0.20` |

B7 must use the confidence column rather than treating every cell equally.

## Corpus progression

### V1.0

```text
usable cells = 13,823 / 52,188 = 26.5%
```

The review queue exposed excessive structural uncertainty caused by broad context scope and treating definite + uncertain repetitions as conflicts.

### V1.1

V1.1 introduced target-local context, arrow delimiters, direct negated structural findings, non-diagnostic indication handling, duplicate suppression and definite-only conflict logic.

```text
usable cells = 14,072
```

### V1.2 / V1.2.1

Further non-gold report review showed three important patterns:

1. abnormality may coexist with `intact fibers`;
2. a directly negated tear does not negate a different abnormality such as mucoid degeneration;
3. a focal abnormality can coexist with a normal component of the same target.

V1.2 therefore made explicit structural abnormality dominate generic local normality and suppressed non-diagnostic history/indication text. V1.2.1 added component-aware aggregation so examples such as the following resolve correctly:

```text
superficial MCL tear + deep MCL intact -> positive
posterior-horn meniscus tear + remainder intact -> positive
proximal ACL tear + tibial insertion intact -> positive
ACL intact + complete ACL tear -> conflict/uncertain
```

## V1.2.1 real-corpus audit

The frozen corpus run contains all 4,407 training reports, with 58 gold rows retained only inside `structured_labels.csv` for the later audit and 4,349 report-only rows in `training_targets.csv`.

At `min_confidence=0.75`:

```text
high-confidence usable cells = 14,123
possible report-only cells   = 52,188
usable fraction overall      = 27.06%
```

Target-wise usable supervision:

| Target | Positive | Negative | Usable | Fraction |
|---|---:|---:|---:|---:|
| ACL | 572 | 1,089 | 1,661 | 38.19% |
| MCL | 271 | 1,089 | 1,360 | 31.27% |
| Medial Meniscus | 1,126 | 536 | 1,662 | 38.22% |
| Lateral Meniscus | 448 | 1,182 | 1,630 | 37.48% |
| Medial OA | 484 | 334 | 818 | 18.81% |
| Lateral OA | 402 | 382 | 784 | 18.03% |
| PF OA | 682 | 372 | 1,054 | 24.24% |
| Effusion | 1,338 | 757 | 2,095 | 48.17% |
| Synovitis | 399 | 17 | 416 | 9.57% |
| Baker's | 557 | 476 | 1,033 | 23.75% |
| Contusion | 389 | 466 | 855 | 19.66% |
| Fracture | 203 | 552 | 755 | 17.36% |

Compared with v1.1, v1.2.1 adds 51 usable cells. The important improvement is qualitative: structural definite conflicts dropped sharply while obvious focal abnormalities with normal uninvolved components are retained as positive.

## Remaining review queue

The v1.2.1 review queue still contains 107 `conflicting_definite_evidence` cells:

```text
Contusion           32
Medial Meniscus     20
Lateral Meniscus    12
ACL                 11
Fracture            10
Effusion             9
MCL                  8
Baker's              4
Synovitis            1
```

Many are genuine report disagreements or temporal/semantic distinctions such as:

```text
ACL normal / later low-grade ACL injury
MCL intact / later grade-I sprain
meniscus normal / later tear in impression
no acute tear / chronic or degenerative tear
```

At this stage, continuing to add corpus rules risks converting B6 into an increasingly hand-engineered labeler. The parser is therefore frozen before any gold comparison.

## Frozen v1.2.1 outputs

```text
runs/b6_report_labels_v121/
├── structured_labels.csv
├── training_targets.csv
├── review_queue.csv
├── audit.json
└── policy.json
```

`training_targets.csv` contains only the 4,349 report-only studies and is the candidate B7 supervision artifact.

## Gold audit — next gate

A dedicated command evaluates the **already-frozen** B6 labels on the 58 gold studies. It does not fit, calibrate, search thresholds or change the parser.

Install the latest editable package and run tests:

```bash
cd /media/talafha/Disk_1/CNN_CPC
git pull
python -m pip install -e .
pytest -q tests/test_b6_report_labels.py tests/test_b6_gold_audit.py
```

Then run:

```bash
rsna-knee-b6-audit \
  --train-csv "$DATA_ROOT/train.csv" \
  --structured runs/b6_report_labels_v121/structured_labels.csv \
  --out-root runs/b6_report_labels_v121/gold_audit \
  --min-confidence 0.75
```

Outputs:

```text
runs/b6_report_labels_v121/gold_audit/
├── gold_audit.json
├── gold_usable_cells.csv
└── gold_mismatches.csv
```

For each target, `gold_audit.json` reports:

- number of gold labels defined;
- number and coverage of high-confidence B6 cells;
- TP / TN / FP / FN;
- positive precision;
- recall/sensitivity;
- specificity;
- negative predictive value;
- accuracy;
- balanced accuracy.

It also reports pooled usable-cell metrics and macro averages across targets.

## Interpretation rule

The gold audit is a **measurement**, not a new tuning loop.

After the audit:

- do not modify B6 lexical rules from gold false positives/false negatives;
- do not optimize `min_confidence` on the 58 labels;
- do not select a new parser version from gold performance;
- use the mismatch file only to understand the reliability limits of B6;
- then design B7 with the frozen v1.2.1 labels and their confidence weights.

This preserves a clean scientific narrative: B6 was built from competition report text, frozen, and only then checked against the small gold set.
