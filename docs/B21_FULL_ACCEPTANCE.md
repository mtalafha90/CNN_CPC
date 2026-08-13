# B21 full-data acceptance protocol

> **Status — 2026-08-14:** WEAK-V2 DEVELOPMENT GATE PASSED / FULL-DATA REFIT IMPLEMENTED / GOLD ACCEPTANCE NOT YET RUN. B20 remains the active working model.

## Frozen weak-v2 result

The leakage-safe matched comparison completed on the frozen 623-study weak-v2 holdout:

```text
B20-v2 control macro AUC        0.7298727911
B21 pre-resize macro AUC        0.7410090411
raw B21 - control              +0.0111362500
paired median                  +0.0109814529
paired 95% CI        [+0.0001624070,+0.0226346590]
P(B21 > control)                0.9758888435
valid bootstrap reps           4894 / 5000
```

This is teacher-agreement evidence, not expert truth. It is sufficient only to freeze the B21-v1 preprocessing decision for the next predeclared step.

## Full-data refit

The full-data acceptance candidate uses:

```text
initializer                    historical B16 report-aligned encoder
encoder                        frozen
B6-active training studies     3120
usable B6 cells               14123
positive / negative            6871 / 7252
eligible MRI series           17475
crop fraction                  0.90
crop stage                     native array before resize
output resolution              224 x 224
training endpoint              fixed epoch 2
cosine scheduler horizon       5 epochs
expert checkpoint selection    disabled
gold labels in gradients       0
```

The trainer requires the already-completed favorable weak-v2 `comparison.json` before it will start.

Canonical output:

```text
runs/b21_full_acceptance/b21_full_model.pt
```

## One-look gold acceptance

The gold evaluator compares the frozen full-data B21 candidate against historical B20 using the same 58 expert studies and matching TTA `[-1,0,1]`.

Predeclared promotion rule:

```text
PROMOTE B21 as working model
iff B21 global 12-target gold macro AUC > 0.667159355531343
```

The historical B20 replay is also performed in the same evaluator. The replay must remain within `0.005` of the canonical B20 score or the acceptance decision aborts.

A stronger scientific superiority statement requires:

```text
paired B21 - B20 bootstrap 95% CI lower bound > 0
```

Target-level AUCs are descriptive only and are forbidden for target mixing, retuning, crop adjustment, or promotion decisions.

The evaluator refuses to run if its output directory already exists. This is a one-look governance guard, not a claim that the reused 58-study gold surface is independent validation.

## Important limitation

Historical B20 was developed using the same 58-study gold surface, including checkpoint selection and broader modelling decisions. Therefore even this one-look B21 acceptance comparison is **not pristine independent validation**. It is a frozen governance comparison only. Hidden competition evaluation remains the independent predictive signal.

## Commands

Full-data B21 refit:

```bash
rsna-knee-b21-full \
  --config configs/b21_full_acceptance.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root runs/b6_report_labels_v121 \
  --series-policy runs/b12_variable_series/audit/series_policy.json \
  --report-ssl-checkpoint runs/b16_full_report/report_ssl/b16_report_encoder.pt \
  --weak-v2-comparison runs/b21_preresize_crop/weak_v2_comparison/comparison.json \
  --out-root runs/b21_full_acceptance
```

After verifying the full-data checkpoint, run the one-look acceptance exactly once:

```bash
rsna-knee-b21-gold-acceptance \
  --config configs/b21_full_acceptance.yaml \
  --data-root "$DATA_ROOT" \
  --b20-checkpoint runs/b20_crop_focus/b20_model.pt \
  --b21-checkpoint runs/b21_full_acceptance/b21_full_model.pt \
  --out-root runs/b21_full_acceptance/gold_acceptance
```

Do not run a second weak-v2 optimization round or modify B21 based on target-wise weak-v2 results before this frozen acceptance step.
