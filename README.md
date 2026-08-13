# CNN_CPC — RSNA Knee Abnormality Detection

`CNN_CPC` is a PyTorch research pipeline for the **2026 RSNA Knee Abnormality Detection** challenge. The released training surface contains 4,407 studies, 58 fully expert-labelled studies, 4,349 report-only/non-gold studies, multiple MRI series per knee, and 12 study-level targets evaluated with macro ROC AUC.

> **Current snapshot — 2026-08-13:** B13--B17 are treated as a **statistically unresolved development tier**. B17 (`0.6425890153`) remains the historical fixed-epoch reference checkpoint. **B18 is completed**: it reproduced the frozen B17 five-epoch B6-only trajectory, used the repeatedly reused 58-study expert set only for one global epoch choice, and selected **epoch 2** with selection statistic `0.6654496134`. That selected score is checkpoint-selection evidence only, not independent validation. The selected checkpoint passed the local three-study inference/schema smoke test. The next meaningful signal is independent competition evaluation.

Canonical records:

- [`docs/B18_FISHER_SELECTION.md`](docs/B18_FISHER_SELECTION.md) — completed B18 expert-guided epoch-selection protocol and result.
- [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md) — development experiment ledger.
- [`docs/B17_FROZEN_ENCODER.md`](docs/B17_FROZEN_ENCODER.md) — completed B17 protocol and result.
- [`docs/B16_FULL_REPORT_ALIGNMENT.md`](docs/B16_FULL_REPORT_ALIGNMENT.md) — B16 protocol and result.
- [`docs/B15_MRI_SSL.md`](docs/B15_MRI_SSL.md) — B15 protocol and results.
- [`docs/B6_B15_GOLD_DIAGNOSTIC.md`](docs/B6_B15_GOLD_DIAGNOSTIC.md) — B6 state/noise-alignment diagnostic.
- [`docs/FINAL_ALL_DATA.md`](docs/FINAL_ALL_DATA.md) — implemented but deferred final-production all-data protocol.
- [`docs/VALIDATION.md`](docs/VALIDATION.md) — validation governance.

## Current software state

```text
package version          0.28.0
primary metric           12-target macro ROC AUC
historical reference     B17 epoch 5 = 0.6425890153 reused-gold point estimate
B13--B17                 statistically unresolved development tier
B18 selected checkpoint  epoch 2
B18 selection statistic  0.6654496134 / selection-only, not validation
B16 reference            0.6349770242
B13 reference            0.6293565948
B6 full-state baseline   0.7024597743 / supervision-information reference, not MRI ceiling
final all-data fit       implemented / deferred / not trained
next independent signal  competition evaluation on data not used for B18 selection
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
| B13 | ImageNet ConvNeXt + hierarchical one-token-per-series | `0.6293565948` gold | unresolved high-performing tier |
| B14 | full `K x 16` slice-token memory + B13 protocol | `0.6197914249` gold | rejected globally |
| B15 | ImageNet -> knee-MRI SSL -> B13 hierarchy | weak-v2 `0.7319060415`; gold `0.6209002783` | teacher gain, no global gold gain |
| B16 | B15 encoder -> full-report semantic alignment -> full B13/B6 surface | `0.6349770242` gold | unresolved high-performing tier |
| **B17** | **freeze completed B16 report-aligned encoder; train hierarchy/head only for 5 fixed full B6 epochs** | **`0.6425890153` gold** | **historical fixed-epoch reference; superiority unresolved** |
| **B18** | **same B17 training; select one of epochs 1--5 using global expert macro AUC only** | **epoch 2 selected; `0.6654496134` selection statistic only** | **completed; independent evaluation pending** |
| FINAL | B17-style recipe + all 58 expert labels in gradients | no gold evaluation permitted | implemented / deferred |

## B18 completed result

The exact predeclared five-epoch selection history was:

```text
epoch 1  loss 0.7371836930  selection AUC 0.6187157061
epoch 2  loss 0.6336947483  selection AUC 0.6654496134  <- selected
epoch 3  loss 0.6087776578  selection AUC 0.6511148368
epoch 4  loss 0.5862506992  selection AUC 0.6394162186
epoch 5  loss 0.5667051629  selection AUC 0.6425890153
```

All five epochs completed the exact frozen B17 surface:

```text
training studies          3120
B6 cells                 14123
positive / negative      6871 / 7252
MRI series               17475
encoder                   frozen B16 report-aligned encoder
encoder LR                0
candidate epochs          5
resolution                224
positions / series        16
TTA                       [-1,0,1]
additional smoothing      0
robust loss               none
```

The encoder SHA-256 remained unchanged:

```text
b328667cf9dfa9b909ef181c1bcc8975ec42bcd8b9eddad08f908875b73fae96
```

Epoch 5 reproduced the prior B17 result (`0.6425890153`) to numerical precision, while the predeclared global selection rule chose epoch 2. Because the same 58 expert studies selected that checkpoint, `0.6654496134` is **not independent validation evidence** and must not be reported as a validated gain over B17.

Selected checkpoint:

```text
runs/b18_fisher_selection/b18_model.pt
```

## Local inference smoke test

The selected epoch-2 checkpoint passed the local three-study schema/inference smoke test:

```text
test rows                    3
test series                 15
series / study               5 / 5 / 5
TTA                          [-1,0,1]
sample columns match         true
sample UID order match       true
metadata repairs             0
```

Command:

```bash
rsna-knee-b18-submit \
  --config configs/b18_fisher_selection.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b18_fisher_selection/b18_model.pt \
  --out runs/b18_fisher_selection/submission_smoke.csv
```

The local three-study output is only a smoke surface. The submission manifest uses the neutral label `B18_submission_inference`; independent competition evaluation must use data not consumed in B18 checkpoint selection.

## Governance

```text
B16/B17: closed to post-gold retuning
B13--B17: treat as statistically unresolved tier
B18: completed; epoch 2 frozen as selected checkpoint
B18: expert labels may select one GLOBAL epoch only
B18: expert labels never enter gradients
B18: selected expert macro is not validation evidence
B18: no target-specific epoch selection or target mixing
B18: no generic smoothing/robust-loss/architecture/resolution/TTA retuning
FINAL all-data fit: deferred pending independent competition evaluation
```

The selected B18 checkpoint must now be judged on a genuinely independent competition evaluation surface.
