# B4 frozen SSL + classical pathology classifiers

B4 freezes the strong competition-only SSL ConvNeXt encoder and uses the 58 gold labels only in low-capacity target-specific PCA + logistic-regression classifiers.

> Current campaign status is summarized in [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).

## Motivation

Before B4, the completed full-model candidates were:

| Model | Macro AUC |
|---|---:|
| B0 random | `0.4762536432` |
| B1 strong SSL | `0.5030284974` |
| B2 differential LR | `0.4993244663` |
| B3 pathology-aware MIL | `0.4944652486` |
| B1+B3 fixed rank | `0.5048038179` |

B4 tests whether the strong SSL representation already contains useful pathology signal that is obscured by high-variance end-to-end fine-tuning on only 58 trusted studies.

## Representation

Checkpoint:

```text
runs/ssl_strong/ssl_encoder.pt
```

Requirements:

- source metadata equals `competition_training_data`;
- no external pretrained image weights;
- encoder is fully frozen;
- deterministic gold feature extraction;
- six dual MRI streams;
- mean, standard deviation and max pooling of 768-dimensional slice embeddings.

The verified gold cache is:

```text
study_uids = (58,)
features   = (58, 6, 2304)
present    = (58, 6)
finite     = true
```

Missing-stream presence flags are explicitly appended to downstream design matrices.

## Nested classifier protocol

For each outer fold and target:

1. hold the outer fold untouched;
2. use the predefined inner fold for target-specific policy selection;
3. fit PCA + logistic regression on the remaining gold selection-training fold;
4. choose among the fixed grid by inner target AUC;
5. refit the selected recipe on all non-outer gold studies;
6. predict the outer fold once.

Feature modes:

- `all`: all six streams;
- `prior`: fixed anatomy/sequence subsets declared before B4 OOF.

Grid:

```text
feature mode:   all, prior
PCA components: 4, 8, 12, 16
logistic C:     0.1, 1.0
```

## Reproduction

```bash
rsna-knee-b4 extract \
  --config configs/train_local_ssl_strong.yaml \
  --split train \
  --scope gold \
  --out runs/b4_frozen_ssl/gold_features.npz

rsna-knee-b4 nested \
  --config configs/train_local_ssl_strong.yaml \
  --features runs/b4_frozen_ssl/gold_features.npz \
  --out-root runs/b4_frozen_ssl \
  --n-bootstrap 5000
```

## Final result

```text
pooled macro AUC = 0.5137567459
95% CI           = [0.4619827141, 0.5642366629]
```

Per-target AUC:

| Target | AUC |
|---|---:|
| ACL | 0.5858 |
| MCL | 0.4807 |
| Medial Meniscus | 0.5421 |
| Lateral Meniscus | 0.6050 |
| Medial OA | 0.5504 |
| Lateral OA | 0.3985 |
| PF OA | 0.6384 |
| Effusion | 0.4447 |
| Synovitis | 0.4456 |
| Baker's | 0.3750 |
| Contusion | 0.5587 |
| Fracture | 0.5403 |

Compared with B1 (A=B1, B=B4):

```text
paired median difference = +0.0102107449
95% CI                   = [-0.0514266147, +0.0709432872]
P(B4 > B1)               = 0.6378
```

## Policy instability

The 36 target/fold selections were highly dispersed:

```text
feature mode: prior 20, all 16
PCA:          4 -> 10, 8 -> 11, 12 -> 6, 16 -> 9
C:            0.1 -> 19, 1.0 -> 17
```

B4.1-B4.3 tested increasingly shared/stabilized selection policies. All reduced pooled OOF performance.

## Decision

**B4 is the current best clean standalone point estimate (`0.5138`).** The paired uncertainty versus B1 is wide, so it is not claimed as a statistically proven improvement.

The B4 selector branch is now frozen: no further policy/grid redesign should be driven by the same 58 outer labels. B5 instead changes the representation while keeping this B4 probe fixed.
