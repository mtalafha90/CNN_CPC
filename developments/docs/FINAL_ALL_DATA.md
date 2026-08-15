# Final all-data production model

> **Status — 2026-08-13:** IMPLEMENTED / DEFERRED / NOT TRAINED. Package lineage `0.27.0`; active development package is `0.28.0` with B18.

This remains a final-production option, not a development experiment. It is deliberately **deferred while B18 expert-guided epoch selection is being tested**. Do not run this fit yet: once the 58 expert studies enter gradients, they can no longer serve even as the deliberately consumed B18 checkpoint-selection surface.

## Why this model exists

B17 has the highest reused-gold point estimate in the unresolved B13--B17 development tier:

```text
B17 macro AUC        0.6425890153
B16 macro AUC        0.6349770242
raw B17-B16          +0.0076119910
paired 95% CI        [-0.0188853047,+0.0332991195]
P(B17>B16)            0.7110
```

These results do not establish B17 as statistically superior. If development is later closed, a final competition fit can consume the 58 expert labels rather than leave useful labelled information unused.

## How all 4,407 studies are used

```text
Total train.csv studies                         4407

B16 report-aligned representation:
  all non-gold MRI/report pairs                 4349
  ├─ B6-active later used for downstream        3120
  └─ B6-inactive representation-only            1229

Final downstream:
  B6-active non-gold                            3120
  expert-gold                                     58
                                                 ----
  supervised downstream                         3178
```

Therefore every training study has a learning role:

- all 4,349 non-gold studies affected the report-aligned MRI encoder;
- the 3,120 B6-active studies also provide pathology supervision;
- the 1,229 B6-inactive studies are not assigned invented labels;
- all 58 expert studies would provide true 0/1 pathology labels in this final fit.

## Final supervision

B6 non-gold cells remain unchanged:

```text
positive -> target 0.85, weight 0.50
negated  -> target 0.05, weight 1.00
uncertain/unmentioned -> ignored
minimum confidence -> 0.75
```

Expert cells:

```text
all 58 studies x 12 targets = 696 cells
true labels = 0/1
base weight = 1.0
```

No target-specific expert upweighting, label smoothing, ELR/SCE, pseudo-label completion, or uncertain/unmentioned conversion is added.

## Final downstream surface

```text
training studies           3178
B6 supervision cells      14123
expert supervision cells    696
total supervised cells    14819
B6 series                 17475
expert series               336
total series              17811
batch size                     2
batches / epoch             1589
epochs                         5
```

The B6-active subset must still reproduce the frozen series SHA:

```text
5c4bb1c52294e45f9e83274c5c07d198dc54811c49b96111b7c8439bd7bcd376
```

## Model / optimization

The model follows the B17 recipe:

```text
initializer: completed B16 report-aligned encoder
encoder: frozen
encoder LR: 0
encoder optimizer membership: false
encoder mode during training: eval
encoder SHA: must remain unchanged
hierarchy/head LR: 1e-4
minimum LR: 1e-6
weight decay: 1e-4
grad clip: 1.0
epochs: 5 exact full passes
TTA: [-1,0,1]
```

## Critical validation consequence

If this final fit is eventually run, the 58 expert studies enter gradients. Therefore:

```text
DO NOT compute a final-model AUC on the 58-study gold set.
DO NOT compare final-model gold AUC against B17/B18.
DO NOT choose an epoch using gold after the final fit begins.
DO NOT tune gold weight after hidden-test feedback without creating a new final recipe.
```

Its first performance estimate would have to be independent competition evaluation.

## Run — intentionally deferred

Do **not** run this command while B18 is active. It is retained here for the later final-production phase:

```bash
cd /media/talafha/Disk_1/CNN_CPC
conda activate rsna-knee
git pull --ff-only origin main
python -m pip install -e .

export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"

rsna-knee-final \
  --config configs/final_all_data.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root runs/b6_report_labels_v121 \
  --series-policy runs/b12_variable_series/audit/series_policy.json \
  --report-ssl-checkpoint runs/b16_full_report/report_ssl/b16_report_encoder.pt \
  --out-root runs/final_all_data
```

Expected checkpoint if eventually run:

```text
runs/final_all_data/final_model.pt
```

Every epoch must report:

```text
batches                         1589 / 1589
study_draws                     3178 / 3178
active_supervision_cells       14819 / 14819
series_instances               17811 / 17811
encoder_lr                      0
encoder_frozen                  true
encoder_training_mode           false
encoder_gradients_detected      false
encoder_sha256                  identical every epoch
full_coverage                   true
full_series_coverage            true
budget_limited                  false
```

## Local submission smoke test

After five complete epochs, if this final fit is eventually authorized:

```bash
rsna-knee-final-submit \
  --config configs/final_all_data.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/final_all_data/final_model.pt \
  --out runs/final_all_data/submission_smoke.csv
```

The local three-study test remains only an inference/schema smoke test.
