# B6 — structured multilingual report labels

> **Status — 2026-08-10:** **COMPLETE / FROZEN at v1.2.1.** B6 is the frozen weak-label source for B7-v1, B7.1, and the currently training B8 experiment. No additional parser or confidence-threshold tuning should be performed using the 58 gold studies.

## Goal

B6 converts each competition training report into auditable target-level weak labels for all 12 abnormalities:

```text
positive
negated
uncertain
unmentioned
```

Each cell also stores a fixed soft probability, confidence, reason, and evidence snippet. The 58 gold studies are retained only for audit and are excluded from `training_targets.csv`.

## Frozen leakage contract

B6 v1.2.1:

- uses only competition `train.csv` reports;
- uses no external model, external language resource, or external data;
- does not fit calibration on gold;
- does not convert report silence to a negative;
- excludes every gold row from the weak-training export;
- keeps uncertain/unmentioned cells at low or zero confidence;
- uses the audited confidence threshold `0.75`.

## Corpus audit — v1.2.1

The final corpus run covered all 4,407 reports, with 58 gold rows audit-only and 4,349 report-only rows available to the weak-label exporter.

At confidence `>=0.75`, v1.2.1 produced **14,123 usable target cells** across the report-only pool.

| Target | Positive | Negative | Usable | Fraction |
|---|---:|---:|---:|---:|
| ACL | 572 | 1,089 | 1,661 | 38.2% |
| MCL | 271 | 1,089 | 1,360 | 31.3% |
| Medial Meniscus | 1,126 | 536 | 1,662 | 38.2% |
| Lateral Meniscus | 448 | 1,182 | 1,630 | 37.5% |
| Medial OA | 484 | 334 | 818 | 18.8% |
| Lateral OA | 402 | 382 | 784 | 18.0% |
| PF OA | 682 | 372 | 1,054 | 24.2% |
| Effusion | 1,338 | 757 | 2,095 | 48.2% |
| Synovitis | 399 | 17 | 416 | 9.6% |
| Baker's | 557 | 476 | 1,033 | 23.8% |
| Contusion | 389 | 466 | 855 | 19.7% |
| Fracture | 203 | 552 | 755 | 17.4% |

The final review queue contained 107 definite-conflict cells. Remaining conflicts are now mostly real semantic/report disagreements rather than broad parser-scope errors, so further corpus-rule expansion was stopped.

## Parser evolution

### v1.0

The initial implementation established multilingual aliases, target states, negation, uncertainty, and an audit queue. It produced 13,823 usable report-only cells but over-produced structural uncertainty.

### v1.1

The first real-corpus review motivated target-local context, arrow delimiters, direct negated structural findings, non-diagnostic indication handling, deduplication, and the rule that uncertain duplicates do not cancel definite evidence.

### v1.2 / v1.2.1

The final corrections added:

- pathology dominance over generic nearby normality;
- preservation of abnormalities such as degeneration even when tear is explicitly absent;
- detection of `loss of normal fibers` as abnormality;
- suppression of clinical indication/history as diagnostic evidence;
- component-aware structural aggregation, so a focal abnormality can coexist with normal uninvolved components.

Examples resolved correctly include:

```text
superficial MCL tear + deep MCL intact -> positive
posterior-horn meniscal tear + remainder intact -> positive
proximal ACL tear + tibial insertion intact -> positive
ACL intact + complete ACL tear -> conflict
```

## Gold audit — frozen v1.2.1

The final parser was audited once against the 58 gold studies without fitting, threshold search, or post-audit parser changes.

Across 251 high-confidence usable gold cells:

```text
TP = 116
TN = 80
FP = 52
FN = 3
```

Pooled metrics:

```text
positive precision = 0.690476
sensitivity        = 0.974790
specificity        = 0.606061
NPV                = 0.963855
accuracy           = 0.780876
balanced accuracy  = 0.790425
coverage           = 0.360632
```

The main scientific conclusion is asymmetric: **B6 explicit negatives are much more reliable than B6 explicit positives.** This motivates the fixed global weak-supervision policy used by B7-v1, B7.1, and B8:

```text
B6 positive -> target 0.85, weight 0.50
B6 negated  -> target 0.05, weight 1.00
uncertain/unmentioned -> ignored
```

Because the global policy was chosen after inspecting the B6 gold audit, subsequent B7/B7.1/B8 performance on the same 58 studies is development performance rather than pristine independent validation.

## Frozen artifacts

```text
runs/b6_report_labels_v121/
├── structured_labels.csv
├── training_targets.csv
├── review_queue.csv
├── audit.json
├── policy.json
└── gold_audit/
    ├── gold_audit.json
    ├── gold_usable_cells.csv
    └── gold_mismatches.csv
```

## Downstream experiment record

The frozen B6 export has now supported:

```text
B7-v1  macro AUC = 0.5397724412
B7.1   macro AUC = 0.5644802945  [current leader]
B8     pending                     [training in progress]
```

B7.1 improved the point estimate after increasing weak-training coverage from about 1.28 nominal corpus passes to four full passes. B8 keeps this same B6 supervision and full coverage while changing only the MRI spatial representation/attention path.

## Final decision

**B6 = PASS as asymmetric weak supervision and is frozen.**

Do not create another B6 parser revision based on the existing gold audit or later B7/B8 target-level results. Current downstream status is documented in `docs/B7_WEAK_SUPERVISION.md`, `docs/B7_1_FULL_COVERAGE.md`, `docs/B8_SPATIAL_ANATOMY.md`, and `docs/EXPERIMENT_STATUS.md`.
