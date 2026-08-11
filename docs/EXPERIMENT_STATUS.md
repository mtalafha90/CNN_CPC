# Experiment status

**Snapshot:** 2026-08-11  
**Package:** `0.22.0`  
**Gold development set:** 58 fully labelled studies  
**Primary metric:** macro ROC AUC across 12 targets

The 58-study surface has supported repeated sequential development decisions and is therefore a development/model-selection set rather than pristine independent validation.

## Current headline

- **Development champion:** **B13**, macro AUC `0.6293565948`, 95% CI `[0.5789896351,0.6775867717]`.
- Versus B12: median `+0.0638674720`, 95% paired CI `[+0.0127183837,+0.1144643292]`, `P(B13>B12)=0.9920`.
- Versus B7.1: median `+0.0652260946`, 95% paired CI `[+0.0039768779,+0.1266069220]`, `P(B13>B7.1)=0.9808`.
- **Active experiment: B14 ImageNet full slice-token aggregation.**
- B14 preserves B13's ImageNet encoder protocol and all training/evaluation controls but removes the learned one-token-per-series compression.
- Primary B14 comparison is predeclared as B14 versus B13 with the same aligned 5,000-replicate bootstrap.

## Experiment ladder

| ID | Method | Macro AUC | Status |
|---|---|---:|---|
| B0 | random-init Stage-1 model | `0.4762536432` | baseline |
| Report teacher | fold-safe rules + TF-IDF teacher | `0.49245` | rejected |
| B1 | competition-only MRI SSL + Stage-1 | `0.5030284974` | retained reference |
| B2 | B1 with lower encoder LR | `0.4993244663` | rejected |
| B3 | pathology-aware low-capacity MIL | `0.4944652486` | rejected |
| B4 | frozen SSL + target-wise PCA/LR | `0.5137567459` | retained ablation |
| B5 | image-report SSL + unchanged B4 probe | `0.5243650851` | representation baseline |
| B6 | structured report labels | n/a | frozen weak-label source |
| B7-v1 | B5-init pathology-query MRI + B6 labels | `0.5397724412` | coverage ablation |
| **B7.1** | **full 3,120-study B7 coverage** | **`0.5644802945`** | previous benchmark |
| B5+B7.1 rank | fixed global 50:50 rank ensemble | `0.5540141184` | rejected |
| B8 | spatial-token anatomy model | `0.5300962807` | rejected |
| B9 | strict exact-contrast routing | `0.5334962669` | rejected |
| B10 | physical-scale normalization | `0.5523982721` | rejected globally |
| B11-v1 | absolute teacher pseudo-label completion | n/a | stopped at viability gate |
| B11.1 | target-wise teacher tails | `0.5506902702` | rejected globally |
| **B12** | **all real MRI series + full slice-token memory + B5 init** | **`0.5660915179`** | retained / tied with B7.1 |
| B12.1 | one learned token per series + B5 init | not run | implemented / skipped |
| **B13** | **one learned token per series + ImageNet ConvNeXt protocol** | **`0.6293565948`** | **RETAINED / DEVELOPMENT CHAMPION** |
| **B14** | **full `K x 16` slice-token memory + same ImageNet protocol** | pending | **IMPLEMENTED / ACTIVE** |

## Frozen B6 supervision surface

```text
report-only studies       4349
active B6 studies         3120
inactive B6 studies       1229
usable B6 cells          14123
positive cells            6871
negative cells            7252
```

Frozen policy:

```text
positive -> target 0.85, base weight 0.50
negated  -> target 0.05, base weight 1.00
uncertain/unmentioned -> ignored
minimum confidence -> 0.75
```

## Frozen all-series surface

```text
training studies        3120
eligible real series   17475
historical dual unique 15468
extra series            2007
max series / study        14
series SHA-256
5c4bb1c52294e45f9e83274c5c07d198dc54811c49b96111b7c8439bd7bcd376
```

## B13 retained result

B13 uses torchvision ConvNeXt-Tiny `IMAGENET1K_V1` plus standard ImageNet mean/std normalization and hierarchical one-token-per-series aggregation.

Training completed four exact full-coverage epochs:

```text
epoch 1 loss  0.7450505349
epoch 2 loss  0.6865059846
epoch 3 loss  0.6524747430
epoch 4 loss  0.6132239342
```

Frozen gold result:

```text
B13 macro AUC      0.6293565948
95% CI            [0.5789896351,0.6775867717]
```

Per-target B13 AUCs, descriptive only:

```text
ACL                0.4742647059
MCL                0.5555555556
Medial Meniscus    0.6093750000
Lateral Meniscus   0.6795031056
Medial OA          0.6279069767
Lateral OA         0.6189555126
PF OA              0.6177606178
Effusion           0.7677018634
Synovitis          0.7108721625
Baker's            0.7481884058
Contusion          0.5533063428
Fracture           0.5888888889
```

No target-specific model mixing is permitted from these values.

## B14 — active controlled experiment

### Scientific question

Does B13's generic one-token-per-series compression discard useful focal slice-level pathology information?

```text
B13
16 slice tokens / real series -> 1 learned series token
K series tokens -> study Transformer -> pathology queries

B14
K real series x 16 slice tokens -> study Transformer -> pathology queries
```

The B14 architecture is the already-tested B12 full-token model combined with the B13 ImageNet encoder protocol.

### Frozen B14 controls versus B13

```text
same ImageNet ConvNeXt-Tiny IMAGENET1K_V1
same ImageNet mean/std normalization
same 3120 studies
same 14123 cells: 6871 positive / 7252 negative
same 17475-series mapping and SHA-256
same 16 sampled positions / series
same 224x224 resize
same plane/fluid/fat metadata embeddings
same batch size 2
same seed / DataLoader seed offsets
same encoder LR 1e-5
same head LR 1e-4
same weight decay
same augmentation
same 4 epochs
same TTA [-1,0,1]
same 5000 bootstrap replicates
zero gold gradients / zero gold early stopping
```

Single change:

```text
B13 learned one-token-per-series compression
    -> removed
B14 retains all K x 16 slice tokens in study memory
```

### Commands

```bash
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"

rsna-knee-b14 \
  --config configs/b14_imagenet_full_tokens.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root runs/b6_report_labels_v121 \
  --series-policy runs/b12_variable_series/audit/series_policy.json \
  --out-root runs/b14_imagenet_full_tokens
```

After four complete epochs:

```bash
rsna-knee-b14-eval \
  --config configs/b14_imagenet_full_tokens.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b14_imagenet_full_tokens/b14_model.pt \
  --out-root runs/b14_imagenet_full_tokens/gold_eval
```

Primary paired comparison:

```bash
python -m rsna_knee.cli evaluate \
  --train-csv "$DATA_ROOT/train.csv" \
  --oof runs/b13_imagenet/gold_eval/gold_predictions.csv \
  --compare-oof runs/b14_imagenet_full_tokens/gold_eval/gold_predictions.csv \
  --n-bootstrap 5000 \
  --out runs/b14_imagenet_full_tokens/gold_eval/b13_vs_b14.json
```

## B14 decision rule

- If B14 is clearly better globally, retain B14.
- If statistically tied, do not build target-wise hybrids; treat both as viable and prioritize an independent Kaggle signal.
- If clearly worse, reject B14 and retain B13.

Do not tune slice count, epochs, learning rates, target-specific winners, thresholds or ensemble weights from the B14 gold result.

Actual hidden-test / leaderboard performance remains unknown until a real competition submission is evaluated.
