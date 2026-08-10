# CNN_CPC — RSNA Knee Abnormality Detection

`CNN_CPC` is a PyTorch research pipeline for the **2026 RSNA Knee Abnormality Detection** challenge. The released training surface contains 4,407 studies, 58 fully labelled gold studies, 4,349 report-only studies, multiple MRI series per knee, and 12 study-level targets evaluated with macro ROC AUC.

> **Current snapshot — 2026-08-10:** **B7.1 full-corpus weak supervision remains the best standalone development model**, macro AUC `0.5644802945`, 95% bootstrap CI `[0.5052432984, 0.6229422178]`. B8 spatial-anatomy learning scored `0.5300962807` and was rejected. A label-free series-metadata audit then exposed 552 cross-contrast substitutions in the historical six-stream router. **B9 strict semantic routing is implemented and predeclared before training/evaluation.**

Canonical measured status: [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md).  
B9 protocol: [`docs/B9_STRICT_ROUTING.md`](docs/B9_STRICT_ROUTING.md).

## Current software state

```text
package version       0.14.0
current leader        B7.1 full-corpus weak supervision
leader macro AUC      0.5644802945
active experiment     B9 strict semantic routing
external pretraining  disabled
final inference       MRI-only
```

## Six MRI streams

```text
sagittal_fluid       sagittal_structural
coronal_fluid        coronal_structural
axial_fluid          axial_structural
```

B9 makes the semantic contract exact:

```text
*_fluid       -> Fluid_Sensitive == True only
*_structural  -> Fluid_Sensitive == False only
missing class -> missing stream / presence mask False
```

The historical B7.1 selector remains untouched for reproducibility.

## Why B9 exists

The historical dual selector tried to fill two stream slots whenever a plane had at least two series. When both acquisitions belonged to the same contrast class, one could be assigned to the opposite semantic slot.

Full training metadata audit:

| Stream | Historical selected | Strict selected | Wrong-slot assignments removed |
|---|---:|---:|---:|
| sagittal_fluid | 4,401 | 4,150 | 251 |
| sagittal_structural | 4,294 | 4,266 | 28 |
| coronal_fluid | 4,250 | 4,248 | 2 |
| coronal_structural | 3,440 | 3,406 | 34 |
| axial_fluid | 4,407 | 4,407 | 0 |
| axial_structural | 1,094 | 857 | 237 |
| **Total** | **21,886** | **21,334** | **552** |

That is **552 / 21,886 = 2.52%** of historically selected streams. The three provided test studies contain one analogous false assignment among 14 historically selected streams; strict routing yields 13 semantically valid streams instead.

This audit uses only series metadata, not target labels.

## Experiment ladder

| ID | Method | Macro AUC | Decision |
|---|---|---:|---|
| B0 | random initialization | `0.4762536432` | baseline |
| report teacher | fold-safe rules + TF-IDF | `0.49245` | rejected |
| B1 | strong competition-only MRI SSL | `0.5030284974` | retained reference |
| B2 | 0.1x encoder LR | `0.4993244663` | rejected |
| B3 | pathology-aware MIL | `0.4944652486` | rejected |
| B4 | frozen SSL + target-wise PCA/LR | `0.5137567459` | retained image-only ablation |
| B5 | image-report SSL + unchanged B4 probe | `0.5243650851` | retained representation baseline |
| B6 | structured multilingual report labels | n/a | frozen weak-label source |
| B7-v1 | B5-init pathology-query model + B6 labels | `0.5397724412` | coverage ablation |
| **B7.1** | **B7 with full 3,120-study epoch coverage** | **`0.5644802945`** | **current leader** |
| B5+B7.1 rank | fixed 50:50 rank ensemble | `0.5540141184` | rejected |
| B8 | 2x2 spatial-token anatomy model | `0.5300962807` | rejected; branch closed |
| **B9** | **B7.1 recipe + strict semantic routing** | pending | **implemented / predeclared** |

## B9 frozen contract

Only routing changes versus B7.1. The following remain fixed:

```text
B5 competition-only encoder initialization
B6 v1.2.1 weak labels
positive target/weight 0.85 / 0.50
negative target/weight 0.05 / 1.00
16 slices per stream
batch size 2
4 epochs
1560 batches per epoch
encoder LR 1e-5
head LR 1e-4
same augmentation
TTA [-1, 0, 1]
5000 bootstrap replicates
no gold gradients
no gold early stopping
```

## Install / update

```bash
cd /media/talafha/Disk_1/CNN_CPC
git pull --ff-only origin main
conda activate rsna-knee
python -m pip install -e .

python - <<'PY'
import rsna_knee
print(rsna_knee.__version__)
PY
```

Expected:

```text
0.14.0
```

## Test B9

```bash
pytest -q \
  tests/test_b6_report_labels.py \
  tests/test_b6_gold_audit.py \
  tests/test_b7_weak_supervision.py \
  tests/test_b9_strict_routing.py
```

## Train B9

```bash
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"

rsna-knee-b9 \
  --config configs/b9_strict_routing.yaml \
  --data-root "$DATA_ROOT" \
  --b5-checkpoint runs/b5_report_ssl/b5_encoder.pt \
  --b6-root runs/b6_report_labels_v121 \
  --out-root runs/b9_strict_routing
```

Outputs:

```text
runs/b9_strict_routing/
├── b9_model.pt
├── history.json
├── policy.json
├── routing_audit.json
└── supervision_plan.json
```

Before gold evaluation:

```bash
cat runs/b9_strict_routing/routing_audit.json
cat runs/b9_strict_routing/history.json
cat runs/b9_strict_routing/supervision_plan.json
```

The routing audit must certify `strict_semantic_mismatches: 0`.

## B9 gold development evaluation

Run only after the training artifacts are inspected:

```bash
rsna-knee-b9-eval \
  --config configs/b9_strict_routing.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b9_strict_routing/b9_model.pt \
  --out-root runs/b9_strict_routing/gold_eval
```

Primary benchmark:

```text
B7.1 macro AUC = 0.5644802945
```

Primary comparison: paired B7.1 -> B9 study-level bootstrap with 5,000 replicates.

## Validation caution

The same 58 gold studies have informed repeated method decisions. Current scores are **development/model-selection estimates**, not pristine hidden-test estimates. Do not tune target-specific routing, weak-label weights, per-target model winners, or ensemble weights from these 58 labels and then present the result as independent validation.

`docs/competition.md` remains a preserved competition-summary document and is intentionally not rewritten by experiment updates.
