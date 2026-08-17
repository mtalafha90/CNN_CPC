# Phase 9 — matched B34 original-B6 vs Phase-8 supervision experiment

## Status

**FROZEN BEFORE MRI RESULTS.**

Phase 8 produced a valid global merged supervision artifact. Phase 9 tests whether that supervision change improves a fixed MRI model when architecture and MRI exposure are held constant.

This is the first downstream MRI experiment authorized to consume the Phase-8 merged supervision artifact.

## Scientific question

```text
Does replacing original frozen B6 v1.2.1 supervision with the globally merged
Phase-8 supervision improve the same fixed B34 knee-MRI model when every MRI-side
choice and every report-only study/series exposure is held constant?
```

## Why the old PV2 training split cannot be reused

PV2 training used 1,997 studies drawn only from the original 3,120 B6-active population. Phase 8 leaves every original B6-active row unchanged. Therefore a control/candidate experiment restricted to the PV2 training split would present identical supervision to both arms and would not test Phase 8.

Phase 9 instead exposes both arms to the complete 4,349 report-only population.

## Matched MRI exposure

Both arms iterate over:

```text
report-only studies                    4349
eligible real MRI series              24035
batch size                                 2
batches per full epoch                 2175
fixed training endpoint                  E2
```

The 24,035-series count is the already audited complete non-gold MRI/report-alignment surface used by B16.

The original B6-inactive studies are **not removed from the control dataloader**. They remain present with zero supervised BCE weight. This is required because removing them from only one arm would change MRI acquisition-domain exposure as well as supervision.

Thus:

```text
CONTROL
4349 studies / 24035 series exposed
3120 studies carry usable B6 cells
1229 studies carry zero supervised BCE weight

CANDIDATE
4349 studies / 24035 series exposed
4173 studies carry usable merged cells
176 studies carry zero supervised BCE weight
```

## Frozen architecture

Both arms use B34 exactly:

```text
B16 frozen report-aligned ConvNeXt-Tiny encoder
+ B20 deterministic centered 90% crop
+ B12/B13 all-real-series study representation
+ B34 complementary query
+ B34 train-only zero-initialized local-context scaffold
+ exact local-context bypass under model.eval()
```

B34 is used because its mechanism was frozen before PV2 and passed the predeclared PV2 scaffold test. Phase 9 does not define B35 and does not alter B34.

## Frozen supervision arms

### CONTROL

```text
source                 frozen B6 v1.2.1
active studies         3120
usable cells          14123
positive cells         6871
negative cells         7252
```

### CANDIDATE

```text
source                 frozen Phase-8 merged supervision
training_targets SHA   c59d78c74743112f09946fd18b64d7726947e6f75b83aabd1f585389a89d045a
active studies         4173
usable cells          18024
positive cells         9590
negative cells         8434
```

Phase 9 aborts if the Phase-8 CSV fingerprint changes.

## Frozen B7 cell semantics

Both arms use the same rule:

```text
positive -> target 0.85, base weight 0.50
negated  -> target 0.05, base weight 1.00
uncertain/unmentioned -> zero weight
minimum definite confidence -> 0.75
```

The existing target-balanced weak BCE rule is retained. The target multiplier is defined as:

```text
mean target weight mass / target weight mass
```

and is recomputed mechanically from each arm's frozen supervision table. This is considered part of the supervision treatment, not an MRI-side hyperparameter change. The formula is identical in both arms and no target-specific manual override is permitted.

## Everything else is matched

```text
architecture                              identical B34
encoder                                   exact same frozen B16 checkpoint
crop                                      exact same B20 90% crop
study population                          same 4349 report-only studies
series exposure                           same 24035 eligible MRI series
slice policy                              same
augmentation                              same
batch size                                same = 2
optimizer                                 same AdamW
head LR                                   same frozen config
weight decay                              same frozen config
scheduler                                 same 5-epoch cosine horizon
training endpoint                         same fixed E2
construction seed                         same +40,000,000 offset
loader seed                               same +40,100,000 offset
post-construction RNG seed                same +40,200,000 offset
gold gradients                            zero in both arms
checkpoint selection from validation      none
```

## Diagnostic evaluation

After both E2 checkpoints exist, evaluate both once on the repeatedly reused 58-study expert surface.

Primary diagnostic:

```text
macro ROC AUC across the 12 expert-labelled targets
```

Paired bootstrap difference:

```text
candidate macro AUC - control macro AUC
```

Positive favors Phase-8 supervision.

This surface is **diagnostic only** because it has been repeatedly reused during model development. There is no new promotion threshold derived from it.

The strongest independent signal remains the hidden competition evaluation or new external expert-labelled data.

## Required local paths

```bash
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"
export B6_ROOT="/media/talafha/Disk_1/CNN_CPC/runs/b6_report_labels_v121"
export PHASE8_ROOT="/media/talafha/Disk_1/CNN_CPC/runs/translation_rescue_supervision_v1"
export SERIES_POLICY="/media/talafha/Disk_1/CNN_CPC/runs/b12_variable_series/audit/series_policy.json"
export B16_ENCODER="/media/talafha/Disk_1/CNN_CPC/runs/b16_full_report/report_ssl/b16_report_encoder.pt"
```

## Run control

```bash
PYTHONPATH=developments/src \
python -m rsna_knee.phase9_matched_supervision_training \
  --arm control \
  --config config/current_model.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root "$B6_ROOT" \
  --phase8-root "$PHASE8_ROOT" \
  --series-policy "$SERIES_POLICY" \
  --report-ssl-checkpoint "$B16_ENCODER" \
  --out-root runs/phase9_matched_supervision
```

## Run candidate

```bash
PYTHONPATH=developments/src \
python -m rsna_knee.phase9_matched_supervision_training \
  --arm candidate \
  --config config/current_model.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root "$B6_ROOT" \
  --phase8-root "$PHASE8_ROOT" \
  --series-policy "$SERIES_POLICY" \
  --report-ssl-checkpoint "$B16_ENCODER" \
  --out-root runs/phase9_matched_supervision
```

Do not stop one arm based on the other arm's training loss. Both must complete the fixed E2 endpoint.

## Paired reused-gold diagnostic

```bash
PYTHONPATH=developments/src \
python -m rsna_knee.phase9_matched_supervision_eval \
  --config config/current_model.yaml \
  --data-root "$DATA_ROOT" \
  --control-checkpoint runs/phase9_matched_supervision/control/model.pt \
  --candidate-checkpoint runs/phase9_matched_supervision/candidate/model.pt \
  --out-root runs/phase9_matched_supervision/eval \
  --n-bootstrap 5000
```

## Expected outputs

```text
runs/phase9_matched_supervision/
├── control/
│   ├── model.pt
│   ├── training_audit.json
│   └── history.json
├── candidate/
│   ├── model.pt
│   ├── training_audit.json
│   └── history.json
└── eval/
    ├── control_gold_predictions.csv
    ├── candidate_gold_predictions.csv
    ├── paired_gold_predictions.csv
    └── comparison.json
```

## Decision boundary

```text
run matched control and candidate                         GO
same 4349-study MRI exposure in both arms                 REQUIRED
same 24035-series exposure in both arms                   REQUIRED
remove zero-weight B6-inactive studies from control       NO-GO
change architecture between arms                          NO-GO
change crop/sampling/optimizer/seeds between arms          NO-GO
use gold for gradients or checkpoint selection             NO-GO
filter Phase-8 targets/scripts after seeing Phase-9       NO-GO
retune B34 from Phase-9 results                            NO-GO
promote from reused-gold diagnostic alone                  NO-GO
independent hidden competition comparison                  GO after both checkpoints are frozen
```
