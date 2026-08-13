# Active working model

> **Decision — 2026-08-13:** B20 is the active working model for all further model development and interpretability work.

## Active model

```text
model                  B20_crop_only_joint_focus
checkpoint             runs/b20_crop_focus/b20_model.pt
canonical epoch        2
spatial policy         centered 90% crop -> resize 224x224
cosine/vignette mask   no
encoder                frozen B16 report-aligned encoder
```

B20 is the preferred knee-focused formulation because it retains the clean 90% anatomical crop while avoiding B19's synthetic vignette/border shortcut.

## Validation state

B20 nested epoch-selection audit:

```text
all-58 selected macro AUC           0.6671593555313430
cross-fitted selected epochs        [2,2,2]
cross-fitted OOF macro AUC          0.6671593555313430
measured epoch-selection optimism   0.0
strict selected epochs              [2,5,2]
strict OOF macro AUC                0.6351640998170208
fixed epoch-5 macro AUC             0.6577823350159498
```

B18 full-FOV comparator nested audit:

```text
historical selected statistic       0.6654496134246369
post-hoc replay epoch-2 macro AUC   0.6655517376076434
cross-fitted selected epochs        [2,2,2]
cross-fitted OOF macro AUC          0.6655517376076434
measured epoch-selection optimism   0.0
strict selected epochs              [2,5,2]
strict OOF macro AUC                0.6475369755138950
fixed epoch-5 / B17 endpoint        0.6425890152580378
```

The B20-vs-B18 cross-fitted difference is approximately `+0.0016076`, which is too small to claim predictive superiority on this repeatedly reused 58-study development surface.

## Model roles

```text
B17  frozen fixed-epoch reference
B18  frozen full-FOV comparator
B19  rejected spatial formulation
B20  ACTIVE WORKING MODEL
```

## Governance

- Continue new modelling work from B20 unless a deliberately controlled comparator experiment requires B18.
- Do not reopen B17/B18/B19 for outcome-driven retuning.
- Treat the 58 expert studies as a reused development/checkpoint-selection surface, not independent validation.
- The nested audits measure checkpoint-selection optimism only.
- Do not claim B20 is globally more accurate than B18 from the small reused-gold difference.
- Prioritize B20 localization/shortcut diagnostics, target-wise error analysis, and controlled B20 improvements before defining a new model family.
