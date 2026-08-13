# B20 nested epoch-selection audit

Status: completed 2026-08-13.

This audit reused B20 candidate epochs 1 through 5. No training was repeated.

Results:

```text
all-58 selected epoch              2
all-58 macro AUC                   0.667159355531343
fixed epoch-5 macro AUC            0.6577823350159498
uplift versus epoch 5             +0.00937702051539313

cross-fit selected epochs          [2, 2, 2]
cross-fit OOF macro AUC            0.667159355531343
cross-fit selection optimism       0.0

strict selected epochs             [2, 5, 2]
strict OOF macro AUC               0.6351640998170208
strict selection optimism          0.03199525571432216
```

The primary cross-fitted audit selected epoch 2 for every outer fold, so epoch 2 remains the canonical B20 checkpoint. The strict one-inner-fold result is retained as a small-selection-set sensitivity analysis.

These values estimate checkpoint-selection optimism only. The 58 expert studies are reused development data, so this is not pristine independent validation.

Current decision: B20 is the primary knee-focused candidate; keep epoch 2.
