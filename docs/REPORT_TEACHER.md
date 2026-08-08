# Strong fold-safe report teacher

The competition test CSV contains only `StudyInstanceUID`, so report text is a **training teacher only**. Final submission inference remains MRI-only.

Version 0.5.0 adds a competition-data-only teacher ensemble that is deliberately evaluated before it is allowed to replace the conservative rule teacher used by Stage 1.

## Components

For each of the 12 targets the teacher combines:

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

Thus `fold{k}/pseudo_labels.csv` is safe for a future image-model outer fold `k`.

## Run

After pulling `main`:

```bash
cd /media/talafha/Disk_1/CNN_CPC
conda activate rsna-knee
python -m pip install -e .

pytest -q tests/test_report_teacher.py

python -m rsna_knee.report_teacher_cli \
  --train-csv "$DATA_ROOT/train.csv" \
  --out-dir runs/report_teacher \
  --n-bootstrap 2000
```

The console prints the ensemble OOF result and writes the complete benchmark to `runs/report_teacher/metrics.json`.

## Outputs

```text
runs/report_teacher/
  metrics.json
  oof.csv
  fold_assignments.csv
  fold0/
    teacher.json
    pseudo_labels.csv
  fold1/
    teacher.json
    pseudo_labels.csv
  fold2/
    teacher.json
    pseudo_labels.csv
```

`oof.csv` contains one strictly out-of-fold prediction for each gold study, plus the three component scores and the final teacher confidence. Each fold-specific pseudo-label file contains all studies with:

- the 12 teacher probabilities;
- `<target>__confidence` for every target;
- `is_gold`;
- `is_outer_gold`;
- `teacher_fold`.

Official gold labels are **not overwritten** in these exports; image training will continue to override pseudo supervision with official finite gold cells.

## Confidence

Confidence is target-specific, not a universal report-state weight. It combines:

- the target ensemble's inner cross-fit AUC above chance; and
- how far the calibrated prediction lies from `0.5`.

An inner AUC of `0.5` yields zero pseudo-label confidence. A high-AUC target can produce high-confidence pseudo-labels only for decisive predictions. This is intended to unlock strong report-derived supervision without blindly lowering the global trusted threshold.

## Decision gate

Do **not** route these pseudo-labels into MRI training merely because they exist. First inspect:

```bash
python - <<'PY'
import json
from pathlib import Path
p = json.loads(Path('runs/report_teacher/metrics.json').read_text())
print('macro OOF:', p['oof']['macro_auc'])
print('95% CI   :', p['oof']['ci_lower'], p['oof']['ci_upper'])
print('\nPer-target ensemble OOF:')
for target, auc in p['oof']['per_target_auc'].items():
    print(f'{target:20s} {auc}')
print('\nComponent macro OOF:')
for name, result in p['component_oof'].items():
    print(f"{name:10s} {result['macro_auc']}")
PY
```

The next engineering step is to integrate the fold-specific teacher into Stage 1 only after its OOF performance and confidence distribution have been reviewed. The old rule-teacher path remains the production default until that gate is passed.
