# Active working model

> **Decision — 2026-08-14:** B20 remains the active working model. B21-v1 passed the frozen weak-v2 development gate and is now frozen for one full-data refit plus one predeclared gold acceptance comparison. B21 does not replace B20 before that comparison is reviewed.

## Active model

```text
model                  B20_crop_only_joint_focus
checkpoint             runs/b20_crop_focus/b20_model.pt
canonical epoch        2
implemented geometry   native MRI -> resize 224 -> center crop 90% -> resize 224
cosine/vignette mask   no
encoder                frozen historical B16 report-aligned encoder
canonical gold score   0.667159355531343
```

Historical B20 is preserved unchanged.

## B21 frozen preprocessing decision

B21 changes the spatial ordering to:

```text
native MRI -> center crop 90% -> percentile normalization -> single resize 224
```

The normalization support therefore differs slightly from B20 and is part of the declared B21-v1 intervention. Crop-fraction sweeps remain forbidden under this implementation.

## Completed weak-v2 development gate

A leakage-safe B16 representation was first rebuilt from B15 MRI SSL while excluding both the 623 weak-v2 holdout studies and all 58 gold studies. Matched B20-v2 and B21 arms then trained on the same 2,497 weak-v2 training studies with the same safe encoder, same seeded hierarchy, same optimizer/augmentation, fixed E2 endpoint, and historical five-epoch cosine-scheduler horizon.

Completed result:

```text
B20-v2 control macro AUC        0.7298727911
B21 pre-resize macro AUC        0.7410090411
raw B21 - control              +0.0111362500
paired median                  +0.0109814529
paired 95% CI        [+0.0001624070,+0.0226346590]
P(B21 > control)                0.9758888435
```

This measures agreement with frozen B6 report supervision, not expert truth. The result freezes the B21-v1 preprocessing decision; it does not promote B21 automatically.

## Full-data acceptance stage

The next and only permitted B21-v1 step is:

```text
historical B16 encoder
        -> frozen
full 3,120-study B6 surface
17,475 eligible MRI series
pre-resize crop fraction 0.90
fixed epoch 2 endpoint
scheduler horizon 5
no gold selection during training
        -> B21 full-data candidate
```

Canonical candidate path after training:

```text
runs/b21_full_acceptance/b21_full_model.pt
```

Canonical protocol: [`B21_FULL_ACCEPTANCE.md`](B21_FULL_ACCEPTANCE.md).

## Predeclared one-look acceptance rule

The acceptance evaluator compares the frozen B21 full-data candidate against historical B20 on the same 58 expert studies with TTA `[-1,0,1]`.

```text
working-model promotion:
B21 global 12-target gold macro AUC > 0.667159355531343

stronger scientific superiority statement:
paired B21-B20 bootstrap 95% CI lower bound > 0
```

The evaluator also replays B20 and aborts if the replay differs from the canonical B20 score by more than `0.005`.

Target-level AUCs are descriptive only. They cannot be used for target mixing, crop changes, retuning, or a second B21-v1 development round.

The 58 expert studies were already reused during historical B20 development. Therefore the acceptance comparison remains a governance/development comparison, not pristine independent validation.

## Historical B20/B18 audit context

```text
B20 cross-fitted epoch selections       [2,2,2]
B20 cross-fitted OOF macro AUC          0.6671593555313430
B20 measured epoch-selection optimism   0.0

B18 cross-fitted epoch selections       [2,2,2]
B18 replay OOF macro AUC                0.6655517376076434
B18 measured epoch-selection optimism   0.0
```

The B20-vs-B18 difference is too small to establish predictive superiority on the reused gold surface.

## Model roles

```text
B17  frozen fixed-epoch reference
B18  frozen full-FOV comparator
B19  rejected spatial formulation
B20  ACTIVE WORKING MODEL
B21  weak-v2-passed frozen candidate; full-data acceptance pending
```

## Governance

- Do not modify historical B20.
- Do not reopen the B21-v1 crop fraction, normalization order, loss, architecture, aggregation, or resolution before the one-look acceptance.
- Do not perform another weak-v2 tuning round for B21-v1.
- Train the full-data candidate using the historical B16 encoder so the final B20-vs-B21 comparison preserves the historical representation source.
- Run the gold acceptance evaluator once after the full-data checkpoint is verified.
- Do not automatically rewrite the working-model decision from the evaluator; review the frozen global result first.
- Hidden competition evaluation remains the independent predictive-performance signal.
