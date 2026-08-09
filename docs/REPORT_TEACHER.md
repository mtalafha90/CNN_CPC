# Fold-safe report teacher benchmark

The competition test CSV contains only `StudyInstanceUID`, so report text is a **training teacher only**. Final submission inference remains MRI-only.

Version 0.5.0 introduced a competition-data-only teacher ensemble combining:

1. the deterministic multilingual clinical-rule state parser with fold-safe empirical calibration;
2. word TF-IDF (1-2 grams) with target-specific balanced logistic regression;
3. character TF-IDF (3-5 grams) with target-specific balanced logistic regression.

No external model, external corpus, LLM, or pretrained text encoder is used by this benchmark.

## Leakage contract

For outer gold fold `k`:

- gold reports and labels from fold `k` are excluded from rule calibration;
- their report groups are excluded from TF-IDF vocabulary/IDF fitting;
- their labels are excluded from every text classifier;
- the remaining two gold folds are cross-fitted against each other;
- target-specific component weights are selected from those inner cross-fit AUCs only;
- ensemble probability calibration and confidence are fitted only from the inner cross-fit predictions;
- only then is the final fold-`k` teacher fitted on all non-outer gold and used to predict the outer gold and report-only studies.

Thus the benchmark is a valid test of whether this text approach deserves promotion into MRI training.

## Observed real-data result

The complete 58-gold-study OOF benchmark produced:

```text
macro AUC = 0.4924496599988921
95% CI    = [0.4396044171132367, 0.5460505496859447]
```

Per-target AUC:

```text
ACL                 0.5723
MCL                 0.3991
Medial Meniscus     0.5108
Lateral Meniscus    0.3764
Medial OA           0.5209
Lateral OA          0.4449
PF OA               0.4607
Effusion            0.4584
Synovitis           0.4851
Baker's             0.5453
Contusion           0.4022
Fracture            0.7333
```

## Decision

**Rejected as a general Stage-1 teacher.**

The ensemble is statistically near chance and does not provide the large supervision-quality improvement required to rescue the MRI student. In particular, a text classifier trained from only roughly 38-40 non-outer gold reports per fold is too data-starved for a twelve-target, multilingual report problem.

The exported `runs/report_teacher/fold*/pseudo_labels.csv` files are retained as research artifacts, but they must **not** replace the production rule teacher globally. Fracture is the only clearly strong target in this benchmark; target-specific use can be reconsidered later inside a controlled ensemble.

The engineering priority after this result moves to a much stronger image representation learned from the 4,349 non-gold MRI studies using competition-data-only self-supervised learning. See `docs/SSL_STRONG.md`.

## Reproduce the benchmark

```bash
python -m rsna_knee.report_teacher_cli \
  --train-csv "$DATA_ROOT/train.csv" \
  --out-dir runs/report_teacher \
  --n-bootstrap 2000
```

Outputs remain:

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
