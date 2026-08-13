# CNN_CPC — RSNA Knee Abnormality Detection

`CNN_CPC` is a PyTorch research pipeline for the **2026 RSNA Knee Abnormality Detection** challenge. The released training surface contains 4,407 studies: 58 fully expert-labelled studies and 4,349 report-only/non-gold studies, with multiple MRI series per knee and 12 study-level targets evaluated by macro ROC AUC.

## Current project state — 2026-08-13

> **B20 is the active working model.** All further model-development, localization, and shortcut-diagnostic work should proceed from B20 unless an explicitly controlled comparison requires B18.

| Model | Role | Spatial input | Canonical epoch | Current evidence |
|---|---|---|---:|---:|
| **B17** | fixed-epoch reference | full FOV | 5 | `0.6425890153` |
| **B18** | frozen full-FOV comparator | full FOV | 2 | cross-fit replay `0.6655517376` |
| **B19** | rejected spatial ablation | 90% crop + cosine vignette | 3 | rejected: artificial border shortcut |
| **B20** | **ACTIVE WORKING MODEL** | 90% crop only | 2 | cross-fit `0.6671593555` |

The 58 expert-labelled studies are a repeatedly reused **development/checkpoint-selection surface, not independent validation**. The nested audits below estimate checkpoint-selection optimism only.

### B18 nested epoch-selection audit

The five saved B18 checkpoints were rescored without retraining. In the primary two-fold cross-fitted audit, every outer fold independently selected epoch 2:

```text
selected epochs                     [2,2,2]
cross-fitted OOF macro AUC          0.6655517376076434
estimated epoch-selection optimism  0.0
fixed epoch-5 / B17 endpoint        0.6425890152580378
```

The strict one-inner-fold sensitivity analysis selected `[2,5,2]` and produced `0.6475369755138950`, showing the expected instability when only about one third of the 58-study surface is used to rank five epochs.

The original B18 training-time statistic was `0.6654496134246369`; deterministic post-hoc replay produced `0.6655517376076434`, a tiny difference of `+0.0001021241830065`. The historical number remains the original-run record; the replay number is used for the nested audit. This difference does not change the selected epoch or interpretation.

### B20 nested epoch-selection audit

```text
selected epochs                     [2,2,2]
cross-fitted OOF macro AUC          0.6671593555313430
estimated epoch-selection optimism  0.0
fixed epoch-5 macro AUC             0.6577823350159498
```

The strict one-inner-fold sensitivity analysis selected `[2,5,2]` and produced `0.6351640998170208`.

### Current interpretation

```text
B17 -> B18: checkpoint-timing gain survives the primary cross-fitted audit
B18 -> B20: small numerical gain only (~+0.00161 on cross-fit replay)
B19: rejected
B20: active working model / primary knee-focused formulation
```

The B20-vs-B18 difference is too small and comes from the same reused 58-study development surface, so it is **not evidence of predictive superiority**. B20 is retained as the working model because it is the clean knee-focused formulation: centered 90% crop, no vignette, no synthetic border shortcut.

## Canonical records

- [`docs/WORKING_MODEL.md`](docs/WORKING_MODEL.md) — active-model decision and governance.
- [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md) — experiment ledger.
- [`docs/B18_FISHER_SELECTION.md`](docs/B18_FISHER_SELECTION.md) — original B18 protocol/result.
- [`docs/B18_NESTED_EPOCH_AUDIT.md`](docs/B18_NESTED_EPOCH_AUDIT.md) — completed B18 checkpoint-selection audit.
- [`docs/B19_JOINT_FOCUS.md`](docs/B19_JOINT_FOCUS.md) — rejected B19 vignette formulation.
- [`docs/B20_CROP_ONLY_FOCUS.md`](docs/B20_CROP_ONLY_FOCUS.md) — canonical B20 model record.
- [`docs/B20_NESTED_EPOCH_AUDIT.md`](docs/B20_NESTED_EPOCH_AUDIT.md) — completed B20 checkpoint-selection audit.
- [`docs/VISUALIZATION_GUIDE.md`](docs/VISUALIZATION_GUIDE.md) — visualization commands and interpretation rules.
- [`docs/VALIDATION.md`](docs/VALIDATION.md) — validation governance.

## Setup

```bash
cd /media/talafha/Disk_1/CNN_CPC
conda activate rsna-knee
git pull --ff-only origin main
python -m pip install -e .

export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"
```

## Active B20 commands

### Submission/inference

```bash
rsna-knee-b20-submit \
  --config configs/b20_crop_focus.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b20_crop_focus/b20_model.pt \
  --out runs/b20_crop_focus/test_predictions.csv
```

### Standard Grad-CAM

```bash
rsna-knee-b20-visualize \
  --config configs/b20_crop_focus.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b20_crop_focus/b20_model.pt \
  --target effusion \
  --cam-layer 28x28 \
  --cam-threshold 0.65
```

### Robust explanation diagnostic

```bash
rsna-knee-b20-explain \
  --config configs/b20_crop_focus.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b20_crop_focus/b20_model.pt \
  --uid <EXPERT_STUDY_UID> \
  --target effusion \
  --cam-layer 28x28 \
  --absolute-threshold 0.65 \
  --cam-percentile 80 \
  --cam-mass 0.80
```

### B20 nested audit

```bash
rsna-knee-b20-nested-audit \
  --config configs/b20_crop_focus.yaml \
  --data-root "$DATA_ROOT" \
  --candidates-root runs/b20_crop_focus/candidates \
  --out-root runs/b20_crop_focus/nested_epoch_audit
```

### B18 comparator nested audit

```bash
rsna-knee-b18-nested-audit \
  --config configs/b18_fisher_selection.yaml \
  --data-root "$DATA_ROOT" \
  --candidates-root runs/b18_fisher_selection/candidates \
  --out-root runs/b18_fisher_selection/nested_epoch_audit
```

## Frozen supervision and encoder

```text
B6-active studies       3120
usable B6 cells        14123
positive cells          6871
negative cells          7252
eligible MRI series    17475
```

B18--B20 use the frozen B16 report-aligned encoder. Encoder SHA-256:

```text
b328667cf9dfa9b909ef181c1bcc8975ec42bcd8b9eddad08f908875b73fae96
```

## Governance

```text
B17: frozen fixed-epoch reference
B18: frozen full-FOV comparator; do not retune
B19: rejected spatial formulation
B20: ACTIVE WORKING MODEL; epoch 2 retained
B18/B20 cross-fit epoch selections: [2,2,2]
B18/B20 measured checkpoint-selection optimism in primary cross-fit audit: 0.0
B18 vs B20 predictive superiority: unresolved
expert labels: development/checkpoint-selection surface only; never used in B18--B20 gradients
selected expert scores: not independent validation evidence
FINAL all-data fit: deferred pending independent evaluation decision
```
