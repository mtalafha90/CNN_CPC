# Fold-safe report teacher benchmark

> **Status — 2026-08-12:** **COMPLETED / REJECTED AS A GENERAL 12-TARGET TEACHER.** The benchmark remains historical. Reports later proved useful through B5 representation alignment and B6 structured weak states. B13 is the current reused-gold champion; B15 now motivates a direct audit of B6 report states.

The competition test CSV does not provide report text, so reports are **training supervision only**. Final submission inference remains MRI-only.

## Benchmark

The fold-safe teacher combined:

1. deterministic multilingual clinical-rule states with fold-safe empirical calibration;
2. word TF-IDF + target-specific balanced logistic regression;
3. character TF-IDF + target-specific balanced logistic regression.

No external model, corpus, LLM or pretrained text encoder was used.

For every outer gold fold, the outer reports/labels were excluded from rule calibration, vocabulary fitting, text-classifier fitting and component selection.

## Result

```text
macro AUC = 0.4924496600
95% CI   = [0.4396044171,0.5460505497]
```

Decision: **rejected as a general Stage-1 12-target teacher**. The tiny gold-labelled report sample was insufficient for reliable supervised text probabilities across all targets.

Fracture was the strongest target point estimate, but target-specific post-hoc adoption was not allowed.

## Why reports remained useful

Rejecting the fold-safe teacher did not imply that reports contain no useful information.

B5 instead used all 4,349 report-only competition studies for semantic image-report alignment without requiring 12-target labels. B5 reached `0.5243650851` under the unchanged frozen-feature probe and became the representation source for B7-family experiments.

B6 then converted reports into auditable target states:

```text
positive
negated
uncertain
unmentioned
```

with 14,123 usable high-confidence cells across 3,120 active report-only studies.

## Current B15-era interpretation

B15 strongly improved ranking of the frozen B6 teacher labels:

```text
B13-v2 control weak-v2  0.5652498118
B15 weak-v2             0.7319060415
paired median           +0.1675245839
95% CI                  [+0.1124433208,+0.2165156305]
```

Yet B15's expert-gold macro AUC was `0.6209002783`, below B13 `0.6293565948`.

This makes the information content of the report states themselves a high-priority diagnostic. The next step is not to revive the tiny supervised text teacher, but to measure how `positive`, `negated`, `uncertain`, and `unmentioned` states relate to expert truth by target.

Do not assume unmentioned findings are negatives.

## Reproduction

```bash
python -m rsna_knee.report_teacher_cli \
  --train-csv "$DATA_ROOT/train.csv" \
  --out-dir runs/report_teacher \
  --n-bootstrap 2000
```

Current status: [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md). B6 record: [`B6_STRUCTURED_REPORT_LABELS.md`](B6_STRUCTURED_REPORT_LABELS.md).