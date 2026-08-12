# CNN_CPC — RSNA Knee Abnormality Detection

`CNN_CPC` is a PyTorch research pipeline for the **2026 RSNA Knee Abnormality Detection** challenge. The released training surface contains 4,407 studies, 58 fully expert-labelled studies, 4,349 report-only/non-gold studies, multiple MRI series per knee, and 12 study-level targets evaluated with macro ROC AUC.

> **Current snapshot — 2026-08-13:** B13--B17 are treated as a **statistically unresolved development tier**. B17 (`0.6425890153`) remains the reference checkpoint because it has the largest reused-gold point estimate, not because superiority has been established. **B18 is now implemented and predeclared**: it repeats the exact frozen B17 five-epoch B6-only recipe and uses the repeatedly reused 58-study expert set only to select one global epoch by 12-target macro AUC. The final all-data fit is implemented but deliberately deferred while B18 is active.

Canonical records:

- [`docs/B18_FISHER_SELECTION.md`](docs/B18_FISHER_SELECTION.md) — active B18 expert-guided epoch-selection protocol.
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
reference checkpoint     B17 = 0.6425890153 reused-gold point estimate
B13--B17                 statistically unresolved development tier
B16 reference            0.6349770242
B13 reference            0.6293565948
B6 full-state baseline   0.7024597743 / supervision-information reference, not MRI ceiling
B18                      implemented / predeclared / not yet run
final all-data fit       implemented / deferred / not trained
final inference          MRI-only unless hidden competition schema proves otherwise
next independent signal  Kaggle hidden evaluation
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
| **B17** | **freeze completed B16 report-aligned encoder; train hierarchy/head only for 5 fixed full B6 epochs** | **`0.6425890153` gold** | **reference checkpoint; superiority unresolved** |
| **B18** | **same B17 training; select one of epochs 1--5 using global expert macro AUC only** | **selection statistic only; no validation claim** | **implemented / predeclared** |
| FINAL | B17-style recipe + all 58 expert labels in gradients | no gold evaluation permitted | implemented / deferred |

## Why B18

The measured B6 state-only information reference is:

```text
B6 state-only macro AUC   0.7024597743
B17 MRI macro AUC         0.6425890153
numerical difference      0.0598707590
```

That difference is **not** a guaranteed extractable MRI gap because the B6 state baseline uses radiologist-report-derived information. It nevertheless motivates testing whether the short frozen B6 training trajectory contains a better-transfer checkpoint before changing model scale, architecture, or supervision.

B18 keeps the gradient path exactly B17:

```text
training studies          3120
B6 cells                 14123
positive / negative      6871 / 7252
MRI series               17475
encoder                   frozen B16 report-aligned encoder
encoder LR                0
head LR                   1e-4
candidate epochs          5
resolution                224
positions / series        16
TTA                       [-1,0,1]
additional smoothing      0
robust loss               none
```

After each epoch, the 58 expert studies are evaluated under `torch.no_grad()` and **only the global 12-target macro AUC** is logged. Per-target AUCs are intentionally not used or logged for selection. The maximum global macro selects the checkpoint; a numerical tie chooses the earliest epoch.

Because the 58 studies now select the checkpoint, the selected score is **not validation evidence**. B18 must be judged on an independent competition evaluation.

## Run B18

```bash
cd /media/talafha/Disk_1/CNN_CPC
conda activate rsna-knee
git pull --ff-only origin main
python -m pip install -e .

export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"

rsna-knee-b18 \
  --config configs/b18_fisher_selection.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root runs/b6_report_labels_v121 \
  --series-policy runs/b12_variable_series/audit/series_policy.json \
  --report-ssl-checkpoint runs/b16_full_report/report_ssl/b16_report_encoder.pt \
  --out-root runs/b18_fisher_selection
```

Selected checkpoint:

```text
runs/b18_fisher_selection/b18_model.pt
```

Local inference smoke test after successful selection:

```bash
rsna-knee-b18-submit \
  --config configs/b18_fisher_selection.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b18_fisher_selection/b18_model.pt \
  --out runs/b18_fisher_selection/submission_smoke.csv
```

## B17 reference result

```text
B17 macro AUC      0.6425890153
95% CI            [0.5935606351,0.6887356582]
B16 macro AUC      0.6349770242
raw B17-B16       +0.0076119910
paired median     +0.0074330332
95% paired CI     [-0.0188853047,+0.0332991195]
P(B17 > B16)       0.7110
```

This result does not establish true B17 superiority; it defines the current reference checkpoint only.

## Governance

```text
B16/B17: closed to post-gold retuning
B13--B17: treat as statistically unresolved tier
B18: expert labels may select one GLOBAL epoch only
B18: expert labels never enter gradients
B18: do not report selected expert macro as validation evidence
B18: no target-specific epoch selection or target mixing
B18: no generic smoothing/robust-loss/architecture/resolution changes
FINAL all-data fit: deferred until development is closed
```

The selected B18 checkpoint must ultimately be evaluated on Kaggle hidden test or another genuinely independent surface.
