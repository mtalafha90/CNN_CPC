# CNN_CPC — RSNA Knee Abnormality Detection

`CNN_CPC` is a PyTorch research pipeline for the **2026 RSNA Knee Abnormality Detection** challenge. The released training surface contains 4,407 studies, 58 fully labelled gold studies, 4,349 report-only studies, multiple MRI series per knee, and 12 study-level targets evaluated with macro ROC AUC.

> **Current snapshot — 2026-08-11:** **B13 remains the development champion**, macro AUC `0.6293565948`, 95% CI `[0.5789896351,0.6775867717]`. Development has been reopened for one controlled high-upside experiment: **B14 keeps B13's ImageNet ConvNeXt protocol but removes the one-token-per-series compression and retains every `K x 16` slice token through the study Transformer.** B14 is implemented, predeclared and training ready.

Canonical status: [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md).  
B13 result/protocol: [`docs/B13_IMAGENET_INIT.md`](docs/B13_IMAGENET_INIT.md).  
B14 frozen protocol: [`docs/B14_IMAGENET_FULL_TOKENS.md`](docs/B14_IMAGENET_FULL_TOKENS.md).

## Current software state

```text
package version          0.22.0
previous benchmark       B7.1 = 0.5644802945
previous best            B12  = 0.5660915179
development champion     B13  = 0.6293565948
active experiment        B14 ImageNet + full K x 16 slice-token memory
primary B14 comparison   B14 vs B13, aligned 5000-replicate bootstrap
final inference           MRI-only
```

## Experiment ladder

| ID | Method | Macro AUC | Decision |
|---|---|---:|---|
| B0 | random initialization | `0.4762536432` | baseline |
| report teacher | fold-safe rules + TF-IDF | `0.49245` | rejected |
| B1 | strong competition-only MRI SSL | `0.5030284974` | retained reference |
| B2 | lower encoder LR | `0.4993244663` | rejected |
| B3 | pathology-aware low-capacity MIL | `0.4944652486` | rejected |
| B4 | frozen SSL + target-wise PCA/LR | `0.5137567459` | retained ablation |
| B5 | image-report SSL + unchanged B4 probe | `0.5243650851` | representation baseline |
| B6 | structured report labels | n/a | frozen weak-label source |
| B7-v1 | B5-init pathology-query model + B6 labels | `0.5397724412` | coverage ablation |
| **B7.1** | **B7 with full 3,120-study epoch coverage** | **`0.5644802945`** | previous benchmark |
| B5+B7.1 rank | fixed 50:50 rank ensemble | `0.5540141184` | rejected |
| B8 | spatial-token anatomy model | `0.5300962807` | rejected |
| B9 | strict semantic routing | `0.5334962669` | rejected |
| B10 | physical-scale normalization | `0.5523982721` | rejected globally |
| B11-v1 | absolute teacher pseudo-label completion | n/a | stopped at viability gate |
| B11.1 | target-wise teacher tails | `0.5506902702` | rejected globally |
| **B12** | **all real MRI series with full slice-token memory + B5 init** | **`0.5660915179`** | retained / tied with B7.1 |
| B12.1 | hierarchical one-token-per-series + B5 init | not run | implemented / skipped |
| **B13** | **hierarchical one-token-per-series + ImageNet ConvNeXt protocol** | **`0.6293565948`** | **RETAINED / DEVELOPMENT CHAMPION** |
| **B14** | **full `K x 16` slice-token memory + same ImageNet protocol as B13** | pending | **IMPLEMENTED / ACTIVE** |

## B13 retained result

```text
B13 macro AUC      0.6293565948
95% CI            [0.5789896351,0.6775867717]

B13 vs B12
median delta       +0.0638674720
95% paired CI      [+0.0127183837,+0.1144643292]
P(B13 > B12)        0.9920

B13 vs B7.1
median delta       +0.0652260946
95% paired CI      [+0.0039768779,+0.1266069220]
P(B13 > B7.1)       0.9808
```

The 58 labelled studies have been repeatedly reused, so these remain development/model-selection estimates rather than independent validation.

## Why B14

B13 performs pathology-specific cross-attention only **after** each real MRI acquisition has been compressed to one learned token:

```text
B13
16 slice tokens -> 1 generic series token
K series tokens -> study Transformer -> pathology queries
```

B14 removes that compression:

```text
B14
K real series x 16 slice tokens
    -> study Transformer
    -> pathology-query cross-attention
```

This is the already-proven B12 full-token architecture combined with B13's stronger ImageNet encoder protocol. Everything else is frozen to B13.

## Frozen B14 controls

```text
same torchvision ConvNeXt-Tiny IMAGENET1K_V1
same ImageNet mean/std normalization
same 3120 training studies
same 14123 B6 cells: 6871 positive / 7252 negative
same 17475-series mapping
same series SHA-256
5c4bb1c52294e45f9e83274c5c07d198dc54811c49b96111b7c8439bd7bcd376
same 16 sampled positions / series
same 224x224 resize
same metadata embeddings
same batch size 2
same encoder LR 1e-5 / head LR 1e-4
same augmentation
same 4 epochs
same TTA [-1,0,1]
same 5000 bootstrap replicates
zero gold gradients / zero gold early stopping
```

## Install / test

```bash
cd /media/talafha/Disk_1/CNN_CPC
git pull --ff-only origin main
conda activate rsna-knee
python -m pip install -e .
python -c "import rsna_knee; print(rsna_knee.__version__)"
```

Expected: `0.22.0`.

```bash
python -m compileall -q src tests
pytest -q \
  tests/test_b13_imagenet_init.py \
  tests/test_b14_full_slice_tokens.py \
  tests/test_b12_variable_series.py
```

## Train B14

```bash
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"

rsna-knee-b14 \
  --config configs/b14_imagenet_full_tokens.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root runs/b6_report_labels_v121 \
  --series-policy runs/b12_variable_series/audit/series_policy.json \
  --out-root runs/b14_imagenet_full_tokens
```

Every full epoch must retain `1560` batches, `3120` study draws, `14123` active cells, `17475` series, `max K=14`, `full_coverage=true`, `full_series_coverage=true`, and `budget_limited=false`.

## Evaluate and compare

```bash
rsna-knee-b14-eval \
  --config configs/b14_imagenet_full_tokens.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b14_imagenet_full_tokens/b14_model.pt \
  --out-root runs/b14_imagenet_full_tokens/gold_eval

python -m rsna_knee.cli evaluate \
  --train-csv "$DATA_ROOT/train.csv" \
  --oof runs/b13_imagenet/gold_eval/gold_predictions.csv \
  --compare-oof runs/b14_imagenet_full_tokens/gold_eval/gold_predictions.csv \
  --n-bootstrap 5000 \
  --out runs/b14_imagenet_full_tokens/gold_eval/b13_vs_b14.json
```

No target-wise B13/B14 mixtures, epoch extensions, slice-count tuning or ensemble-weight search are allowed from the 58-study B14 result.
