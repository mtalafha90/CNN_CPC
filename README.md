# CNN_CPC — RSNA Knee Abnormality Detection

`CNN_CPC` is a PyTorch research pipeline for the **2026 RSNA Knee Abnormality Detection** challenge. The released training surface contains 4,407 studies: 58 fully expert-labelled studies and 4,349 report-only/non-gold studies, with multiple MRI series per knee and 12 study-level targets evaluated by macro ROC AUC.

## Current project state — 2026-08-14

> **B20 is the active working model.** B21-v1 corrected the crop ordering and passed a leakage-safe weak-v2 development gate, but failed the predeclared full-data expert acceptance comparison. B22 then showed that extending the B21 formulation from E2 to E5 does not rescue expert performance.

| Model | Role | Spatial input / experiment | Canonical result | Status |
|---|---|---|---:|---|
| **B17** | fixed-epoch reference | full FOV | E5 `0.6425890153` | frozen |
| **B18** | full-FOV comparator | full FOV | replay E2 `0.6655517376` | frozen; nested audit complete |
| **B19** | spatial ablation | 90% crop + cosine vignette | E3 `0.6581308356` | rejected: artificial border shortcut |
| **B20** | **ACTIVE WORKING MODEL** | resize 224 -> 90% crop -> resize 224 | E2 `0.6671593555` | **active** |
| **B21-v1** | crop-order correction | native 90% crop -> normalization -> resize 224 | weak-v2 `0.7410090411`; gold `0.6573196516` | weak-v2 passed; gold acceptance failed |
| **B22** | B21 training-duration audit | same B21 preprocessing, E1-E5 | best E2 `0.6574269018` | closed; longer training did not rescue |

The 58 expert-labelled studies are a repeatedly reused **development/model-selection surface, not independent validation**. Hidden competition evaluation remains the independent predictive-performance signal.

### B18 nested epoch-selection audit

```text
selected epochs                     [2,2,2]
cross-fitted OOF macro AUC          0.6655517376076434
estimated epoch-selection optimism  0.0
fixed epoch-5 / B17 endpoint        0.6425890152580378
```

Strict one-inner-fold sensitivity analysis:

```text
selected epochs                     [2,5,2]
OOF macro AUC                       0.6475369755138950
estimated selection optimism        0.0180147620937484
```

### B20 nested epoch-selection audit

```text
selected epochs                     [2,2,2]
cross-fitted OOF macro AUC          0.6671593555313430
estimated epoch-selection optimism  0.0
fixed epoch-5 macro AUC             0.6577823350159498
```

Strict one-inner-fold sensitivity analysis:

```text
selected epochs                     [2,5,2]
OOF macro AUC                       0.6351640998170208
estimated selection optimism        0.0319952557143222
```

The B20-vs-B18 cross-fitted difference is only about `+0.00161`, so it is not evidence of predictive superiority on the reused 58-study surface. B20 is retained because it is the clean historical knee-focused formulation without B19's artificial vignette boundary.

## B21/B22 result: crop order and training duration

Historical B20 actually executes:

```text
native MRI -> percentile normalization -> resize 224 -> center crop 90% -> resize 224
```

B21 tested:

```text
native MRI -> center crop 90% -> percentile normalization -> resize 224
```

A leakage-safe weak-v2 matched comparison favored B21:

```text
B20-v2 control macro AUC        0.7298727911
B21 pre-resize macro AUC        0.7410090411
raw B21 - control              +0.0111362500
paired 95% CI        [+0.0001624070,+0.0226346590]
```

But the frozen full-data expert acceptance comparison did not:

```text
B20 replay macro AUC            0.6674066371
B21 macro AUC                   0.6573196516
B21 - B20 replay               -0.0100869854
paired 95% CI        [-0.0328814731,+0.0117052345]
P(B21 > B20)                    0.1812
```

B21 was therefore not promoted.

B22 then tested whether B21 had simply been stopped too early:

```text
Epoch   training loss   expert macro AUC
E1      0.7388751291    0.6135270850
E2      0.6381611442    0.6574269018  <- best
E3      0.6087977977    0.6387456622
E4      0.5890809184    0.6136783995
E5      0.5680555741    0.6282683534
```

The training objective continues to improve after E2 while expert ranking deteriorates. Longer downstream training therefore does **not** rescue the pre-resize crop under the frozen B16/B6 regime.

### Current interpretation

```text
B17/B18/B20/B22: E2 repeatedly emerges as the strongest downstream region
B21: weak-v2 improvement did not transfer to expert gold
B22: more optimization lowers weak-training loss but worsens expert ranking after E2
```

The current optimization bottleneck is therefore the **weak-label / development-selection signal**, not more epochs. Future work should first audit or improve the model-selection surface before investing in another large architecture or preprocessing sweep.

## Canonical records

- [`docs/WORKING_MODEL.md`](docs/WORKING_MODEL.md) — active model and current scientific position.
- [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md) — full experiment ledger through B22.
- [`docs/B18_NESTED_EPOCH_AUDIT.md`](docs/B18_NESTED_EPOCH_AUDIT.md) — completed B18 checkpoint-selection audit.
- [`docs/B19_JOINT_FOCUS.md`](docs/B19_JOINT_FOCUS.md) — rejected B19 vignette formulation.
- [`docs/B20_CROP_ONLY_FOCUS.md`](docs/B20_CROP_ONLY_FOCUS.md) — canonical B20 model record.
- [`docs/B20_NESTED_EPOCH_AUDIT.md`](docs/B20_NESTED_EPOCH_AUDIT.md) — completed B20 checkpoint-selection audit.
- [`docs/B21_PRERESIZE_CROP.md`](docs/B21_PRERESIZE_CROP.md) — B21 crop-order development and final disposition.
- [`docs/B21_FULL_ACCEPTANCE.md`](docs/B21_FULL_ACCEPTANCE.md) — completed one-look B21 acceptance result.
- [`docs/B22_DURATION_AUDIT.md`](docs/B22_DURATION_AUDIT.md) — completed E1-E5 duration audit.
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

## Frozen supervision and encoder

```text
B6-active studies       3120
usable B6 cells        14123
positive cells          6871
negative cells          7252
eligible MRI series    17475
```

Historical B16 encoder SHA used by the B18-B22 full-data runs:

```text
b328667cf9dfa9b909ef181c1bcc8975ec42bcd8b9eddad08f908875b73fae96
```

## Governance

```text
B17: frozen fixed-epoch reference
B18: frozen full-FOV comparator; nested audit complete
B19: rejected spatial formulation
B20: ACTIVE WORKING MODEL; preserve checkpoint/preprocessing exactly
B21: closed; weak-v2 passed but gold acceptance failed
B22: closed exploratory duration audit; E2 best, no longer-training rescue
58-study expert surface: reused development surface, not independent validation
weak-v2: teacher-agreement diagnostic, not a validated expert-truth surrogate
no target-specific epoch choice or B20/B21/B22 target mixing
FINAL all-data expert-label fit: deferred
```
