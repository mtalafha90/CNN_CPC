# Phase 9 v2 — matched B34 supervision experiment with frozen PV2 holdout

## Status

**FROZEN BEFORE ANY PHASE-9 ENDPOINT OR VALIDATION RESULT.**

Phase 9 v1 kept all 4,349 report-only studies in both training arms. That preserved matched MRI exposure, but it also consumed the 499-study PV2 validation surface and left the repeatedly reused 58-study expert set as the only planned readout.

Before any Phase-9 E2 checkpoint or Phase-9 validation result was inspected, the protocol was revised. Phase 9 v1 is retained as an auditable historical protocol but is **superseded for execution** by Phase 9 v2.

## Scientific question

```text
Does adding the frozen Phase-8 translated supervision improve a fixed B34
training trajectory when the same MRI studies, series, architecture, encoder,
preprocessing, optimizer, stochastic path and endpoint are used in both arms?
```

## Why v2 holds out PV2 validation

The frozen PV2 validation set contains 499 studies selected by UID-only hashing from the old PV1-training population. Phase 8 leaves all original B6-active rows unchanged, so these 499 studies have identical original-B6 supervision in the control and candidate tables.

Using them as a Phase-9 v2 holdout therefore gives a larger no-Phase9-gradient paired surface whose labels are outside the Phase-8 treatment itself.

PV2 remains **weak-label and historically exposed**. It is not independent clinical validation and cannot by itself promote a model. Its role here is narrower: a fixed global readout for the supervision-treatment experiment.

Frozen PV2 fingerprints:

```text
PV2 split SHA-256
b53331ce314b2d2ccc68aea1737427c01bd0d916997e78fbefe88fec5cc95855

parent PV1 split SHA-256
a0032307abb1ab99724eb39fac25332ce131c575f64d823083bb37f5ec20d1e6
```

## Frozen training/holdout populations

```text
complete report-only population             4349 studies / 24035 series
PV2 holdout                                  499 studies /  2775 series
Phase-9 v2 training population              3850 studies / 21260 series
batch size                                      2
batches per full epoch                       1925
fixed endpoint                                 E2
```

The same 499 UIDs are removed from both arms before gradients.

All 1,229 originally B6-inactive report-only studies remain in both training loaders. They retain zero supervised weight in the control unless Phase 8 rescued them in the candidate. This preserves the acquisition-domain control that motivated Phase 9.

Because the 499 holdout studies all come from the original B6-active population:

```text
CONTROL active training studies      3120 - 499 = 2621
CANDIDATE active training studies    4173 - 499 = 3674
```

Exact cell counts are validated at runtime from the frozen PV2 manifest and the frozen supervision artifacts. No manual target adjustment is allowed.

## Frozen architecture and MRI-side contract

Both arms use exactly:

```text
B34 training-only local-context scaffold
same frozen B16 report-aligned ConvNeXt-Tiny encoder
same B20 centered 90% crop
same B12/B13 all-real-series policy
same 16-position 2.5D sampling
same augmentation
same batch size = 2
same AdamW optimizer
same head LR / weight decay
same five-epoch cosine scheduler horizon
same fixed E2 endpoint
same construction seed offset          +40,000,000
same loader seed offset                +40,100,000
same post-construction RNG seed offset +40,200,000
zero PV2-holdout gradients
zero expert-gold gradients
no checkpoint selection from validation
```

The target-balance multiplier formula remains the frozen B7 rule and is recomputed mechanically from each arm's retained training supervision. This is part of the supervision treatment and is not manually tuned.

## Supervision arms

### CONTROL

Frozen B6 v1.2.1.

### CANDIDATE

Frozen Phase-8 global merge:

```text
training_targets.csv SHA-256
c59d78c74743112f09946fd18b64d7726947e6f75b83aabd1f585389a89d045a
```

Before training, v2 verifies that every target/weight on the 499-study holdout is exactly equal to original B6 in **both** arms. Any mismatch aborts the experiment.

## Primary evaluation

After both fixed-E2 checkpoints are frozen, evaluate once on the 499-study PV2 holdout using **original frozen B6 supervision only**.

Primary metric:

```text
macro of per-target B6-weighted soft-label BCE
lower is better
```

Paired bootstrap:

```text
candidate macro weighted BCE - control macro weighted BCE
```

Negative favors Phase-8 supervision.

Secondary metric:

```text
macro ROC AUC over original B6 positive/negated states where both classes exist
```

The metric code is the same frozen machinery used by PV1/PV2. No new threshold or target-specific rule is introduced.

## Interpretation boundary

A Phase-9 v2 improvement means that adding the translated supervision improved the fixed B34 training trajectory as measured against held-out original-B6 weak labels.

It does **not** establish that the translated labels are clinically correct, does not directly validate the rescued non-Latin population, and does not provide independent clinical performance evidence. PV2 has historical exposure and B34 itself was previously mechanistically evaluated on this surface.

Therefore:

```text
use Phase-9 v2 for global supervision-treatment evidence       YES
use target-wise Phase-9 v2 results to filter rescue cells       NO
retune B34 from Phase-9 v2                                      NO
promote model from PV2 holdout alone                            NO
use hidden competition / new external expert data for promotion YES
```

## Required paths

```bash
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"
export B6_ROOT="/media/talafha/Disk_1/CNN_CPC/runs/b6_report_labels_v121"
export PHASE8_ROOT="/media/talafha/Disk_1/CNN_CPC/runs/translation_rescue_supervision_v1"
export SERIES_POLICY="/media/talafha/Disk_1/CNN_CPC/runs/b12_variable_series/audit/series_policy.json"
export B16_ENCODER="/media/talafha/Disk_1/CNN_CPC/runs/b16_full_report/report_ssl/b16_report_encoder.pt"
export PV1_MANIFEST="runs/prospective_weak_v1/split_manifest.json"
export PV2_MANIFEST="runs/prospective_weak_v2/split_manifest.json"
```

## Train control

```bash
PYTHONPATH=developments/src \
python -m rsna_knee.phase9_matched_supervision_v2_training \
  --arm control \
  --config config/current_model.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root "$B6_ROOT" \
  --phase8-root "$PHASE8_ROOT" \
  --parent-pv1-manifest "$PV1_MANIFEST" \
  --pv2-manifest "$PV2_MANIFEST" \
  --series-policy "$SERIES_POLICY" \
  --report-ssl-checkpoint "$B16_ENCODER" \
  --out-root runs/phase9_matched_supervision_v2
```

## Train candidate

```bash
PYTHONPATH=developments/src \
python -m rsna_knee.phase9_matched_supervision_v2_training \
  --arm candidate \
  --config config/current_model.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root "$B6_ROOT" \
  --phase8-root "$PHASE8_ROOT" \
  --parent-pv1-manifest "$PV1_MANIFEST" \
  --pv2-manifest "$PV2_MANIFEST" \
  --series-policy "$SERIES_POLICY" \
  --report-ssl-checkpoint "$B16_ENCODER" \
  --out-root runs/phase9_matched_supervision_v2
```

Both arms must complete fixed E2. Do not stop or select based on training loss.

## Evaluate once

```bash
PYTHONPATH=developments/src \
python -m rsna_knee.phase9_matched_supervision_v2_eval \
  --config config/current_model.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root "$B6_ROOT" \
  --parent-pv1-manifest "$PV1_MANIFEST" \
  --pv2-manifest "$PV2_MANIFEST" \
  --control-checkpoint runs/phase9_matched_supervision_v2/control/model.pt \
  --candidate-checkpoint runs/phase9_matched_supervision_v2/candidate/model.pt \
  --out-root runs/phase9_matched_supervision_v2/eval \
  --n-bootstrap 5000
```

## Expected outputs

```text
runs/phase9_matched_supervision_v2/
├── control/
│   ├── model.pt
│   ├── training_audit.json
│   └── history.json
├── candidate/
│   ├── model.pt
│   ├── training_audit.json
│   └── history.json
└── eval/
    ├── control_pv2_predictions.csv
    ├── candidate_pv2_predictions.csv
    ├── paired_pv2_predictions.csv
    └── comparison.json
```

## Decision boundary

```text
execute Phase-9 v1 all-4349-gradient protocol                 NO-GO / superseded
execute Phase-9 v2 matched 3850-study training                GO
hold exact PV2 499 out of both arms                           REQUIRED
use original B6 labels only on PV2 evaluation                 REQUIRED
change architecture/crop/sampling/optimizer/seeds             NO-GO
use PV2 or gold for checkpoint selection                      NO-GO
filter translated cells after seeing Phase-9 results          NO-GO
promote from Phase-9 v2 alone                                 NO-GO
hidden competition / new external expert confirmation         GO after checkpoints freeze
```
