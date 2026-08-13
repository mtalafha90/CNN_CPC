# B18 — Nested epoch-selection audit

> **Status — 2026-08-13:** COMPLETED. No retraining. Epoch 2 is stable under three-fold cross-fitted epoch selection.

This audit reuses the five saved B18 candidate checkpoints and changes only the checkpoint-selection evaluation rule. It therefore isolates checkpoint-selection optimism without changing the trained models.

## Primary two-fold cross-fitted audit

For each held-out outer fold, the other two folds selected one global epoch using the same 12-target macro ROC AUC and earliest-epoch tie break as B18.

```text
outer fold 0 -> epoch 2
outer fold 1 -> epoch 2
outer fold 2 -> epoch 2

cross-fitted OOF macro AUC          0.6655517376076434
all 12 target AUCs defined          true
estimated epoch-selection optimism  0.0
```

Thus the B18 epoch-2 checkpoint is stable under the primary cross-fitted checkpoint-selection audit.

## Fixed-endpoint comparison

```text
post-hoc replay epoch-2 macro AUC    0.6655517376076434
fixed epoch-5 macro AUC              0.6425890152580378
replay uplift vs epoch 5            +0.0229627223496056
```

Epoch 5 is the B17 fixed-epoch endpoint, so the B17 -> B18 checkpoint-timing gain survives this specific selection-bias audit.

## Strict historical-manifest sensitivity analysis

Using only one approximately one-third inner fold to select the epoch produced:

```text
selected epochs                     [2,5,2]
strict OOF macro AUC                0.6475369755138950
estimated selection optimism        0.0180147620937484
```

The strict estimator is retained as a small-selection-set sensitivity diagnostic. With only about 19 studies used to rank five epochs, epoch selection is substantially noisier. Because gold labels never enter B18 gradient training, the two-fold cross-fitted estimator is the primary checkpoint-selection analysis.

## Historical selection statistic versus deterministic replay

The original B18 training-time record stored epoch 2 at:

```text
historical training-time statistic  0.6654496134246369
post-hoc checkpoint replay          0.6655517376076434
absolute difference                +0.0001021241830065
```

This tiny discrepancy does not alter the selected epoch or any scientific conclusion. The historical number remains the canonical record of what was observed during the original training run; the replay number is used for the nested audit because it comes from rescoring the saved checkpoint under the current deterministic audit path.

The audit inference path explicitly uses `model.eval()`, fixed `[-1,0,1]` TTA, and deterministic evaluation preprocessing, so this is not the earlier Grad-CAM visualization dropout bookkeeping issue. The most plausible interpretation is a very small numerical replay difference affecting one or more near-tied prediction ranks. The original run did not save the complete per-study prediction matrix, so the exact historical rank change cannot be reconstructed retrospectively.

## Interpretation boundary

This audit estimates **checkpoint-selection optimism only**. The same 58 expert-labelled studies have influenced the broader development campaign, so neither the historical nor cross-fitted value is pristine independent validation.

Correct conclusion:

> B18 epoch 2 is a robust checkpoint choice under three-fold cross-fitted epoch selection, and the B17 -> B18 fixed-epoch-to-selected-epoch improvement is not explained by checkpoint-selection noise in this audit.

Incorrect conclusions:

```text
Do not call 0.66555 independent validation.
Do not claim external generalization from the 58-study surface.
Do not reopen B18 for retuning.
```

## Project role after this audit

```text
B17  fixed-epoch reference
B18  frozen full-FOV comparator
B19  rejected spatial formulation
B20  active working model / primary knee-focused candidate
```

Future modelling, localization diagnostics, and controlled improvements should proceed from B20 unless an explicit comparator experiment requires B18.
