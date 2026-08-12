# B18 — Fisher-style expert-guided epoch selection

> **Status — 2026-08-13:** IMPLEMENTED / PREDECLARED / NOT YET RUN. Package `0.28.0`.

B18 is a checkpoint-selection experiment built directly on B17. It does **not**
change the MRI encoder, downstream architecture, B6 supervision, augmentation,
resolution, slice count, or optimizer. Its only intervention is to use the
repeatedly reused 58-study expert set to select one global epoch among five
short frozen-encoder B6-only training passes.

## Why B18

B13--B17 should now be treated as a statistically unresolved development tier,
not a sequence of demonstrated improvements. B17 remains the reference
checkpoint because it has the largest frozen point estimate:

```text
B13  0.6293565948
B16  0.6349770242
B17  0.6425890153
```

but B17-B16 remains unresolved:

```text
raw delta          +0.0076119910
paired median      +0.0074330332
95% paired CI      [-0.0188853047,+0.0332991195]
P(B17>B16)          0.7110
```

B18 tests a narrower hypothesis inspired by short-training / expert-guided
checkpoint selection: perhaps epoch 5 is not the best transfer point when the
training labels are sparse noisy report-derived B6 supervision.

## Frozen B18 question

```text
Among five otherwise identical B17 training epochs, does selecting one epoch
using the global 12-target macro AUC on the repeatedly reused 58-study expert
set provide a better checkpoint for independent test transfer than always
keeping epoch 5?
```

## Unchanged B17 gradient surface

```text
training studies                    3120
usable B6 cells                    14123
positive cells                      6871
negative cells                      7252
eligible real MRI series           17475
batches / epoch                     1560
batch size                             2
```

B6 targets remain:

```text
positive -> target 0.85, weight 0.50
negated  -> target 0.05, weight 1.00
uncertain/unmentioned -> ignored
minimum confidence -> 0.75
```

No additional generic label smoothing is applied because B6 positive/negative
targets are already soft (`0.85/0.05`).

## Unchanged model

```text
initializer                 completed B16 report-aligned encoder
encoder                     fully frozen
encoder LR                  0
encoder optimizer member    false
encoder training mode       eval
resolution                  224 x 224
positions / series          16
series aggregation          learned one-token-per-series
series-pool heads           8
study Transformer           2 layers / 8 heads
pathology-query layers      1
dropout                     0.25
head LR                     1e-4
minimum LR                  1e-6
weight decay                1e-4
grad clip                   1.0
```

The encoder SHA-256 must remain identical after every epoch.

## Expert selection surface

The same 58 fully labelled studies are used after each epoch:

```text
expert studies               58
expert target cells          696
expert MRI series            336
TTA                           [-1,0,1]
selection metric              global 12-target macro ROC AUC
candidate epochs              1,2,3,4,5
tie break                     earliest epoch
```

Important: expert labels enter **no gradient**. Evaluation runs under
`torch.no_grad()`.

Only the single global macro AUC is logged for each epoch. Per-target AUCs are
intentionally not written to the selection history, and there is no bootstrap
or target-specific checkpoint choice.

## Selection rule

```text
train epoch 1 on B6 only -> global expert macro AUC
train epoch 2 on B6 only -> global expert macro AUC
train epoch 3 on B6 only -> global expert macro AUC
train epoch 4 on B6 only -> global expert macro AUC
train epoch 5 on B6 only -> global expert macro AUC

select = epoch with maximum global macro AUC
exact numerical tie -> earliest epoch
```

All five candidate epochs are trained before selection. This is therefore
expert-guided checkpoint selection rather than literal wall-clock early stopping.

## Critical interpretation

Because the 58 studies select the checkpoint:

```text
DO NOT report the selected 58-study score as validation evidence.
DO NOT compare selected B18 gold AUC against B17 and call the difference a gain.
DO NOT select a different epoch per target.
DO NOT tune smoothing, LR, architecture, resolution or TTA from the epoch curve.
DO NOT bootstrap the five candidate scores to manufacture significance.
```

The selected score is a **checkpoint-selection statistic only**. B18 must be
judged on Kaggle hidden test or another genuinely independent dataset.

## Outputs

```text
runs/b18_fisher_selection/
  policy.json
  supervision_plan.json
  history.json
  selection_history.json
  selection.json
  candidates/
    epoch_1.pt
    epoch_2.pt
    epoch_3.pt
    epoch_4.pt
    epoch_5.pt
  b18_model.pt                  # globally selected checkpoint
```

## Run

```bash
cd /media/talafha/Disk_1/CNN_CPC
conda activate rsna-knee
git pull --ff-only origin main
python -m pip install -e .

export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"

rsna-knee-b18 \
  --config configs/b18_fisher_selection.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root runs/b6_report_labels_v121 \
  --series-policy runs/b12_variable_series/audit/series_policy.json \
  --report-ssl-checkpoint runs/b16_full_report/report_ssl/b16_report_encoder.pt \
  --out-root runs/b18_fisher_selection
```

Every training epoch must still show exactly:

```text
batches                       1560 / 1560
studies                       3120 / 3120
active B6 cells              14123 / 14123
positive cells                6871 / 6871
negative cells                7252 / 7252
series                       17475 / 17475
encoder LR                    0
encoder frozen                true
encoder training mode         false
encoder gradients             false
encoder SHA                   unchanged
full coverage                 true
full series coverage          true
budget limited                false
```

After each epoch B18 additionally prints one selection line containing only:

```text
b18_expert_selection_epoch
 global_macro_auc
 selection_only_not_validation = true
```

After epoch 5, `selection.json` records the frozen global decision and
`b18_model.pt` contains the selected epoch.

## Hidden-test submission

After successful selection:

```bash
rsna-knee-b18-submit \
  --config configs/b18_fisher_selection.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b18_fisher_selection/b18_model.pt \
  --out runs/b18_fisher_selection/submission_smoke.csv
```

The local three-study test remains a schema/inference smoke test only. The
selected B18 model's meaningful performance estimate must come from independent
competition evaluation.
