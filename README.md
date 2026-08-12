# CNN_CPC — RSNA Knee Abnormality Detection

`CNN_CPC` is a PyTorch research pipeline for the **2026 RSNA Knee Abnormality Detection** challenge. The released training surface contains 4,407 studies, 58 fully labelled gold studies, 4,349 report-only studies, multiple MRI series per knee, and 12 study-level targets evaluated with macro ROC AUC.

> **Current snapshot — 2026-08-13:** **B17 is the current reused-gold development champion by the predeclared global point-estimate rule** at macro AUC `0.6425890153`. Its advantage over B16 is `+0.0076119910`, with paired median `+0.0074330332`, 95% paired CI `[-0.0188853047,+0.0332991195]`, and `P(B17>B16)=0.7110`. The gain is positive but statistically unresolved. B17 freezes the completed B16 report-aligned encoder and trains only the unchanged B13/B16 hierarchy/head for five exact full B6 passes.

Canonical records:

- [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md) — current experiment ledger.
- [`docs/B17_FROZEN_ENCODER.md`](docs/B17_FROZEN_ENCODER.md) — completed B17 protocol and result.
- [`docs/B16_FULL_REPORT_ALIGNMENT.md`](docs/B16_FULL_REPORT_ALIGNMENT.md) — B16 protocol and result.
- [`docs/B15_MRI_SSL.md`](docs/B15_MRI_SSL.md) — B15 protocol and results.
- [`docs/B6_B15_GOLD_DIAGNOSTIC.md`](docs/B6_B15_GOLD_DIAGNOSTIC.md) — B6 state/noise-alignment diagnostic.
- [`docs/VALIDATION.md`](docs/VALIDATION.md) — validation governance.

## Current software state

```text
package version          0.26.0
primary metric           12-target macro ROC AUC
development champion     B17 = 0.6425890153 by frozen point-estimate rule
B16 reference            0.6349770242 / statistically unresolved with B17
B13 reference            0.6293565948
B15 reused gold          0.6209002783
B6 full-state baseline   0.7024597743 / descriptive information reference, not ceiling
weak-v2                  teacher agreement only / not a B16/B17 gate
final inference          MRI-only
next independent signal  hidden Kaggle evaluation
```

## Experiment ladder

| ID | Method | Macro AUC / evaluation | Decision |
|---|---|---:|---|
| B0 | random initialization | `0.4762536432` gold | baseline |
| report teacher | fold-safe rules + TF-IDF | `0.49245` gold | rejected |
| B1 | competition-only MRI SSL | `0.5030284974` gold | historical reference |
| B4 | frozen SSL + PCA/LR | `0.5137567459` gold | retained ablation |
| B5 | image-report SSL + B4 probe | `0.5243650851` gold | representation baseline |
| B6 | structured report labels | n/a | frozen weak-label source |
| B7.1 | full B6 weak-supervised pathology model | `0.5644802945` gold | historical benchmark |
| B12 | all real MRI series | `0.5660915179` gold | historical reference |
| **B13** | **ImageNet ConvNeXt + hierarchical one-token-per-series** | **`0.6293565948` gold** | historical champion/reference |
| B14 | full `K x 16` slice-token memory + B13 protocol | `0.6197914249` gold | rejected globally |
| B15 | ImageNet -> knee-MRI SSL -> B13 hierarchy | weak-v2 `0.7319060415`; gold `0.6209002783` | teacher gain, no gold gain |
| **B16** | **B15 encoder -> full-report semantic alignment -> full B13/B6 surface** | **`0.6349770242` gold** | historical champion/reference; unresolved with B17 |
| **B17** | **freeze completed B16 report-aligned encoder; train hierarchy/head only for 5 fixed full B6 epochs** | **`0.6425890153` gold** | **current champion by frozen point-estimate rule; superiority unresolved** |

## B17 completed result

B17 used the completed report-aligned encoder checkpoint:

```text
runs/b16_full_report/report_ssl/b16_report_encoder.pt
```

and froze it completely during downstream training. The encoder had zero trainable parameters, zero optimizer membership, zero gradients, remained in evaluation mode, and retained exactly the same SHA-256 fingerprint through all five epochs:

```text
b328667cf9dfa9b909ef181c1bcc8975ec42bcd8b9eddad08f908875b73fae96
```

Every epoch covered exactly:

```text
3120 studies
14123 usable B6 cells
6871 positive / 7252 negative
17475 real MRI series
1560 batches
full coverage true
full series coverage true
budget limited false
```

Training losses:

```text
0.7371836930
0.6336947483
0.6087776578
0.5862506992
0.5667051629
```

One-look reused-gold result:

```text
B17 macro AUC      0.6425890153
95% CI            [0.5935606351,0.6887356582]
B16 macro AUC      0.6349770242
raw B17-B16       +0.0076119910
paired median     +0.0074330332
95% paired CI     [-0.0188853047,+0.0332991195]
P(B17 > B16)       0.7110
```

The predeclared rule uses the global point estimate, so B17 becomes the development champion. The paired evidence does **not** establish true superiority over B16.

B17 gives modest support to preserving the report-aligned representation rather than continuing to update the encoder directly against sparse/noisy B6 supervision. Because B17 also used five fixed downstream epochs versus four for B16, the difference cannot be attributed solely to freezing.

## B6/B15 diagnostic result

The post-B15 diagnostic found:

```text
coverage-conditioned B6 AUC   0.7736374158 on 251 / 696 cells
full-surface state baseline   0.7024597743
high-confidence B6 cells      251
B6 correct / wrong            196 / 55
```

On the 55 B6-wrong cells, B15 did not systematically move toward B6 errors; 63.6% moved toward expert truth. The state baseline is treated as a **supervision-information reference**, not a numerical MRI ceiling.

## Governance

B16 and B17 are closed to post-gold tuning. Do not add epoch 6, choose label smoothing or ELR/SCE from the B17 gold result, tune head LR from gold, or construct target-specific B16/B17 mixtures.

The 58-study gold surface remains a repeatedly reused development/model-selection set. The most credible independent performance signal remains the **hidden Kaggle evaluation**.
