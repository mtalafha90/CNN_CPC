# Strong competition-data MRI SSL

> **Status — 2026-08-12:** **COMPLETED / RETAINED HISTORICAL REPRESENTATION BASELINE.** Strong SSL supported B1/B4 and initialized B5. B13 is now the reused-gold champion; B15 later tested a different ImageNet-initialized knee-MRI SSL protocol and did not improve global gold AUC.

## Motivation

Reference results before strong SSL:

```text
B0 random-init macro AUC  0.4762536432
report teacher macro AUC  0.49245
```

The strong schedule increased competition-only non-gold MRI coverage while preserving the rule that no gold study participates in SSL pretraining.

## Frozen schedule

```text
ssl_epochs                8
max batches/epoch      1000
batch size                3
ssl n_slices               9
positions/stream           2
projection dim           256
temperature              0.15
metadata weight          0.25
LR                    0.0002
minimum LR           0.000001
external pretrained     false
```

## Completed run

```text
completed epochs          8
completed batches      8000
study draws           24000
approx corpus passes   5.52
active 2.5D examples 238274
loss ~3.434 -> ~2.862
```

Checkpoint:

```text
runs/ssl_strong/ssl_encoder.pt
```

## B1 probe result

```text
B1 macro AUC       0.5030284974
95% CI            [0.4474281231,0.5566718294]
paired median vs B0 +0.02646
95% paired CI     [-0.04464,+0.09870]
P(B1 > B0)         0.771
```

The point estimate supported useful in-domain MRI representation learning, though the 58-study uncertainty was wide.

## Downstream lineage

Strong SSL supported B1/B4 and initialized B5. B5 then added report-semantic representation alignment and initialized B7-family weak-supervision models.

Later B13 changed to an ImageNet ConvNeXt protocol and reached `0.6293565948`, the current reused-gold champion.

B15 tested a separate MRI-domain adaptation path:

```text
ImageNet -> knee-MRI same-study contrastive SSL -> B13 hierarchy
```

B15 raised frozen weak-v2 teacher agreement from `0.5652498118` to `0.7319060415`, with paired median `+0.1675245839`, but its reused-gold result was `0.6209002783`, below B13.

Thus in-domain SSL remains useful, but B15 demonstrates that stronger weak-teacher compatibility is not sufficient by itself for stronger expert-gold ranking.

## Current direction

The next evidence-driven step is a B6 report-state audit rather than another strong-SSL schedule sweep. Do not tune SSL duration, target-specific representations, or B15-like SSL hyperparameters from reused gold.

Current status: [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).