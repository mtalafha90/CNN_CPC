# Prospective weak-validation v1

> **Frozen before B34.** This framework replaces the repeatedly reused 58-study expert set as the primary architecture-selection surface for subsequent **downstream/frozen-encoder** development. It is a weak-label validation framework, not independent clinical validation.

## Why this exists

The 58-study expert development set has been inspected repeatedly across the B20/B27/B28/B29/B30/B31/B32/B33 lineage. Continuing to design architectures against that same surface would increasingly optimize to the evaluation set.

Prospective weak-validation v1 therefore freezes a fresh StudyInstanceUID-level partition of the 3,120 active B6 weak-supervision studies before B34.

## Frozen split

```text
source active studies   3120
training studies        2496  (80%)
validation studies       624  (20%)
```

Assignment is based only on StudyInstanceUID:

```text
key(uid) = SHA256("CNN_CPC|prospective-weak-v1|2026-08-16" + NUL + uid)
```

All 3,120 active UIDs are sorted by this key. The first 624 are assigned to validation and the remaining 2,496 to training. B6 states, confidences, expert labels, previous model scores, and model predictions do not influence membership.

The generated manifest records the exact UID lists and SHA-256 fingerprints. Once generated, the split must not be changed after model outcomes are inspected.

### Scope limitations

This policy certifies **study-level** separation only. It does not claim patient-identity grouping because the frozen supervision surface is keyed by `StudyInstanceUID`. If a trustworthy patient identifier becomes available later, that would require a separately named validation policy rather than silently changing this split.

There is also an important representation-pretraining limitation: the historical B16 encoder was aligned on **all 4,349 non-gold MRI/report pairs**. The 624 PV1 validation studies are therefore not unseen by the historical encoder. For that reason PV1 is valid only for comparing **downstream architecture changes while the exact same B16 encoder remains frozen and shared**. PV1 must not be used to select a new encoder, a new representation-pretraining method, or any change that retrains the encoder using this validation population.

## Matched controls

Before B34, retrain these three controls from the same historical B16/B20 initialization on the frozen 80% subset:

```text
PV1-B20   historical B20 architecture, fixed E2
PV1-B31   frozen B31 local-context complementary model, fixed E2
PV1-B33   frozen B33 uniform complementary mean, fixed E2
```

All three use:

```text
same B6 supervision policy
same exact frozen B16 encoder
same 90% post-resize crop
same variable-series policy
same optimizer and LR
same augmentation
same construction seed
same loader seed
same five-epoch scheduler horizon
same fixed E2 endpoint
no validation checkpoint selection
no expert labels
```

The 624 validation studies are never loaded during the matched-control supervised training stage.

## Primary selection metric

Primary metric on the untouched matched-control validation partition:

```text
macro of per-target B6-weighted soft-label BCE
```

For each target, BCE is averaged using the frozen B6 cell weights; the final metric is the unweighted macro average across all 12 targets. **Lower is better.**

This avoids unstable class-count behavior for rare weak-label targets while keeping every target equally represented at the macro level.

Secondary metric:

```text
macro AUC over hard B6 positive/negated states
```

AUC is reported only for targets where both classes occur in the validation partition. It is descriptive and secondary.

Paired study-level bootstrap differences of the primary loss are reported for B31-vs-B20, B33-vs-B20 and B33-vs-B31.

## Interpretation boundary

This surface is fresh with respect to the **supervised downstream B20--B33 architecture decisions**, but its labels are generated from B6 reports and its fixed B16 encoder was historically pretrained using all non-gold reports. PV1 therefore provides a prospective **frozen-encoder downstream architecture-selection** signal only. It is not independent expert/clinical validation and is not a valid encoder-selection surface.

A future B34 may be selected against this frozen surface only if it retains the exact shared frozen B16 encoder. The split itself, primary metric, fixed E2 endpoint, and matched-control protocol must not be changed in response to observed outcomes.

## Evaluation memory-safety policy

The first PV1 evaluation attempt was terminated by `systemd-oomd` after the terminal scope reached a 56.6 GiB memory peak. The failure was operational, not a model-selection outcome, and no final comparison was produced.

PV1 evaluation v1.1 therefore freezes a lower-memory implementation while preserving the exact validation population, checkpoints, `[-1,0,1]` TTA, prediction semantics, and metrics. The evaluator now:

```text
loads one checkpoint at a time
predicts the full 624-study validation surface
stores predictions on CPU/disk
releases the model/checkpoint payload
gc.collect() + torch.cuda.empty_cache()
then loads the next checkpoint
```

The evaluation loader is also fixed at:

```text
batch_size                1
num_workers               1
prefetch_factor           1
persistent_workers        false
series_cache_mb_per_worker 0
```

These settings are resource-management controls only and are not a new experimental degree of freedom. Per-model predictions are written immediately as `b20_predictions.csv`, `b31_predictions.csv`, and `b33_predictions.csv`, with matching metadata JSON files, so progress remains visible if an external process kill occurs again.

## Commands

Create the frozen manifest:

```bash
PYTHONPATH=developments/src python -m rsna_knee.prospective_weak_v1 \
  --config config/current_model.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root "$B6_ROOT" \
  --out runs/prospective_weak_v1/split_manifest.json
```

Train matched controls:

```bash
for MODEL in b20 b31 b33; do
  PYTHONPATH=developments/src python -m rsna_knee.prospective_weak_v1_training \
    --model "$MODEL" \
    --config config/current_model.yaml \
    --data-root "$DATA_ROOT" \
    --split-manifest runs/prospective_weak_v1/split_manifest.json \
    --b6-root "$B6_ROOT" \
    --series-policy "$SERIES_POLICY" \
    --report-ssl-checkpoint "$B16_ENCODER" \
    --out-root runs/prospective_weak_v1
done
```

Evaluate all three together with the frozen low-memory implementation:

```bash
PYTHONPATH=developments/src python -m rsna_knee.prospective_weak_v1_eval \
  --config config/current_model.yaml \
  --data-root "$DATA_ROOT" \
  --split-manifest runs/prospective_weak_v1/split_manifest.json \
  --b6-root "$B6_ROOT" \
  --b20-checkpoint runs/prospective_weak_v1/b20/model.pt \
  --b31-checkpoint runs/prospective_weak_v1/b31/model.pt \
  --b33-checkpoint runs/prospective_weak_v1/b33/model.pt \
  --out-root runs/prospective_weak_v1/eval \
  --n-bootstrap 5000
```

Expected final files:

```text
runs/prospective_weak_v1/
├── split_manifest.json
├── b20/model.pt
├── b20/training_audit.json
├── b31/model.pt
├── b31/training_audit.json
├── b33/model.pt
├── b33/training_audit.json
└── eval/
    ├── b20_predictions.csv
    ├── b20_prediction_meta.json
    ├── b31_predictions.csv
    ├── b31_prediction_meta.json
    ├── b33_predictions.csv
    ├── b33_prediction_meta.json
    ├── paired_predictions.csv
    └── comparison.json
```
