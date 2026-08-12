# CNN_CPC — RSNA Knee Abnormality Detection

`CNN_CPC` is a PyTorch research pipeline for the **2026 RSNA Knee Abnormality Detection** challenge. The released training surface contains 4,407 studies, 58 fully labelled gold studies, 4,349 report-only studies, multiple MRI series per knee, and 12 study-level targets evaluated with macro ROC AUC.

> **Current snapshot — 2026-08-12:** **B16 is the current reused-gold development champion by the predeclared global point-estimate rule** at macro AUC `0.6349770242`. Its advantage over B13 is only `+0.0056204295` and is statistically unresolved: paired 95% CI `[-0.0395927864,+0.0519351407]`, `P(B16>B13)=0.5828`. B16 is therefore retained by policy, not claimed as demonstrated superior. The 58-study gold set remains a repeatedly reused development/model-selection surface, not independent validation.

Canonical records:

- [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md) — current experiment ledger.
- [`docs/B16_FULL_REPORT_ALIGNMENT.md`](docs/B16_FULL_REPORT_ALIGNMENT.md) — B16 protocol, training integrity, and final result.
- [`docs/B15_MRI_SSL.md`](docs/B15_MRI_SSL.md) — B15 protocol and results.
- [`docs/B6_B15_GOLD_DIAGNOSTIC.md`](docs/B6_B15_GOLD_DIAGNOSTIC.md) — B6 state/noise-alignment diagnostic.
- [`docs/WEAK_HOLDOUT_V2.md`](docs/WEAK_HOLDOUT_V2.md) — frozen weak-v2 contract.
- [`docs/VALIDATION.md`](docs/VALIDATION.md) — validation governance.

## Current software state

```text
package version          0.25.0
primary metric           12-target macro ROC AUC
development champion     B16 = 0.6349770242 by frozen point-estimate rule
B13 reference            0.6293565948 / statistically unresolved with B16
B14 gold                 0.6197914249 / rejected globally
B15 weak-v2              0.7319060415 / teacher-agreement gate passed
B15 reused gold          0.6209002783 / no global improvement
B6 full-state baseline   0.7024597743 / descriptive, not ceiling
weak-v2                  teacher agreement only / not a B16 gate
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
| **B13** | **ImageNet ConvNeXt + hierarchical one-token-per-series** | **`0.6293565948` gold** | **retained historical champion/reference** |
| B14 | full `K x 16` slice-token memory + B13 protocol | `0.6197914249` gold | rejected globally |
| B15 | ImageNet -> knee-MRI SSL -> B13 hierarchy | weak-v2 `0.7319060415`; gold `0.6209002783` | teacher gain, no gold gain |
| **B16** | **B15 encoder -> full-report semantic alignment -> full B13/B6 surface** | **`0.6349770242` gold** | **current champion by frozen point-estimate rule; superiority unresolved** |

## B16 completed result

B16 report alignment used all 4,349 non-gold reports and 24,035 eligible real MRI series. Four complete report-alignment passes reduced total loss from `3.8958491301` to `2.5218941658`; every epoch covered exactly 4,349 studies, 24,035 series, and 48,070 2.5D examples with no budget truncation.

B16 then returned to the exact full B13 downstream surface:

```text
3120 studies
14123 usable B6 cells
6871 positive / 7252 negative
17475 real MRI series
1560 batches per epoch
4 complete epochs
```

Final reused-gold result:

```text
B16 macro AUC      0.6349770242
95% CI            [0.5854729266,0.6830266155]
B13 macro AUC      0.6293565948
raw B16-B13       +0.0056204295
paired median     +0.0050711608
95% paired CI     [-0.0395927864,+0.0519351407]
P(B16 > B13)       0.5828
```

The predeclared rule was global point estimate only, so B16 is retained as the development champion. The paired result does **not** establish true superiority over B13.

## B6/B15 diagnostic result

The post-B15 diagnostic found:

```text
coverage-conditioned B6 AUC   0.7736374158 on 251 / 696 cells
full-surface state baseline   0.7024597743
high-confidence B6 cells      251
B6 correct / wrong            196 / 55
```

On the 55 B6-wrong cells, B15 did not systematically move toward B6 errors; 63.6% moved toward expert truth. The pooled state audit showed informative but pathology-dependent middle states, so B16 used full report semantics rather than gold-derived `uncertain`/`unmentioned` pseudo-targets.

## Governance

Do not tune B16 after the gold look, extend epochs, change report temperature/queue/loss weight, create target-wise B13/B16 hybrids, or derive new soft labels from the repeatedly reused 58-study gold set.

The most credible next performance signal is the **hidden Kaggle evaluation**. Any further B17 experiment must be separately predeclared rather than tuned as a continuation of B16.
