# CNN_CPC — RSNA Knee Abnormality Detection

`CNN_CPC` is a PyTorch research pipeline for the **2026 RSNA Knee Abnormality Detection** challenge. The released training surface contains 4,407 studies, 58 fully expert-labelled studies, 4,349 report-only/non-gold studies, multiple MRI series per knee, and 12 study-level targets evaluated with macro ROC AUC.

> **Current snapshot — 2026-08-13:** **B17 remains the reused-gold development champion** at macro AUC `0.6425890153` by the predeclared global point-estimate rule. A separate **FINAL all-data production model** is now implemented. It consumes the 58 expert studies in training and therefore must not be evaluated on the reused 58-study gold surface afterward; its first performance estimate must be the hidden Kaggle evaluation.

Canonical records:

- [`docs/FINAL_ALL_DATA.md`](docs/FINAL_ALL_DATA.md) — final-production all-data protocol and run commands.
- [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md) — development experiment ledger.
- [`docs/B17_FROZEN_ENCODER.md`](docs/B17_FROZEN_ENCODER.md) — completed B17 protocol and result.
- [`docs/B16_FULL_REPORT_ALIGNMENT.md`](docs/B16_FULL_REPORT_ALIGNMENT.md) — B16 protocol and result.
- [`docs/B15_MRI_SSL.md`](docs/B15_MRI_SSL.md) — B15 protocol and results.
- [`docs/B6_B15_GOLD_DIAGNOSTIC.md`](docs/B6_B15_GOLD_DIAGNOSTIC.md) — B6 state/noise-alignment diagnostic.
- [`docs/VALIDATION.md`](docs/VALIDATION.md) — validation governance.

## Current software state

```text
package version          0.27.0
primary metric           12-target macro ROC AUC
development champion     B17 = 0.6425890153 by frozen point-estimate rule
B16 reference            0.6349770242 / statistically unresolved with B17
B13 reference            0.6293565948
B6 full-state baseline   0.7024597743 / information reference, not ceiling
final production         implemented / not yet trained
final inference          MRI-only
next performance signal  hidden Kaggle evaluation
```

## Development ladder

| ID | Method | Macro AUC / evaluation | Decision |
|---|---|---:|---|
| B0 | random initialization | `0.4762536432` gold | baseline |
| B1 | competition-only MRI SSL | `0.5030284974` gold | historical reference |
| B4 | frozen SSL + PCA/LR | `0.5137567459` gold | retained ablation |
| B5 | image-report SSL + B4 probe | `0.5243650851` gold | representation baseline |
| B7.1 | full B6 weak-supervised pathology model | `0.5644802945` gold | historical benchmark |
| B12 | all real MRI series | `0.5660915179` gold | historical reference |
| B13 | ImageNet ConvNeXt + hierarchical one-token-per-series | `0.6293565948` gold | historical champion/reference |
| B14 | full `K x 16` slice-token memory + B13 protocol | `0.6197914249` gold | rejected globally |
| B15 | ImageNet -> knee-MRI SSL -> B13 hierarchy | weak-v2 `0.7319060415`; gold `0.6209002783` | teacher gain, no gold gain |
| B16 | B15 encoder -> full-report semantic alignment -> full B13/B6 surface | `0.6349770242` gold | historical champion/reference |
| **B17** | **freeze completed B16 report-aligned encoder; train hierarchy/head only for 5 fixed full B6 epochs** | **`0.6425890153` gold** | **development champion; superiority unresolved** |
| **FINAL** | **B17 recipe + all 58 expert labels in downstream training** | **no gold evaluation permitted** | **final production / hidden-test only** |

## B17 result

```text
B17 macro AUC      0.6425890153
95% CI            [0.5935606351,0.6887356582]
B16 macro AUC      0.6349770242
raw B17-B16       +0.0076119910
paired median     +0.0074330332
95% paired CI     [-0.0188853047,+0.0332991195]
P(B17 > B16)       0.7110
```

B17 is retained by the frozen point-estimate rule; the paired evidence does not establish true superiority over B16.

## Final all-data production model

All 4,407 studies now have a training role across the pipeline:

```text
B16 representation learning
  all non-gold MRI/report studies               4349
  ├─ B6-active downstream studies               3120
  └─ B6-inactive representation-only studies    1229

FINAL downstream
  B6-active non-gold                            3120
  expert-gold                                     58
                                                 ----
  supervised downstream                         3178
```

Final downstream surface:

```text
B6 cells                  14123
expert cells                696
total supervised cells    14819
B6 series                 17475
expert series               336
total series              17811
batches / epoch            1589
epochs                        5
encoder                    frozen B16 report-aligned encoder
```

Expert labels use true `0/1` targets at base weight `1.0`. B6 supervision remains unchanged. The 1,229 B6-inactive studies do not receive invented pathology labels.

Run:

```bash
rsna-knee-final \
  --config configs/final_all_data.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root runs/b6_report_labels_v121 \
  --series-policy runs/b12_variable_series/audit/series_policy.json \
  --report-ssl-checkpoint runs/b16_full_report/report_ssl/b16_report_encoder.pt \
  --out-root runs/final_all_data
```

After five exact full epochs, generate the submission with:

```bash
rsna-knee-final-submit \
  --config configs/final_all_data.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/final_all_data/final_model.pt \
  --out runs/final_all_data/submission_smoke.csv
```

## Governance

B16 and B17 are closed to post-gold tuning. The FINAL model consumes all 58 expert labels in gradients, so **do not calculate or report a final-model AUC on those 58 studies** and do not use them for early stopping or checkpoint selection. The final model goes directly to independent hidden Kaggle evaluation.
