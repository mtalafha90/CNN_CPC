# Prospective weak-validation v1

> **Frozen before B34.** This framework replaced the repeatedly reused 58-study expert set as the primary architecture-selection surface for subsequent **downstream/frozen-encoder** development. It is a weak-label validation framework, not independent clinical validation. The original B20/B31/B33 PV1 comparison is now complete and immutable.

## Why this exists

The 58-study expert development set had been inspected repeatedly across the B20/B27/B28/B29/B30/B31/B32/B33 lineage. Continuing to design architectures against that same surface would increasingly optimize to the evaluation set.

Prospective weak-validation v1 therefore froze a fresh StudyInstanceUID-level partition of the 3,120 active B6 weak-supervision studies before B34.

## Frozen split

```text
source active studies   3120
training studies        2496  (80%)
validation studies       624  (20%)
validation series       3544
```

Assignment is based only on StudyInstanceUID:

```text
key(uid) = SHA256("CNN_CPC|prospective-weak-v1|2026-08-16" + NUL + uid)
```

All 3,120 active UIDs are sorted by this key. The first 624 are assigned to validation and the remaining 2,496 to training. B6 states, confidences, expert labels, previous model scores, and model predictions do not influence membership.

Frozen split SHA-256:

```text
a0032307abb1ab99724eb39fac25332ce131c575f64d823083bb37f5ec20d1e6
```

The generated manifest records the exact UID lists and SHA-256 fingerprints. The split must not be changed after model outcomes are inspected.

### Scope limitations

This policy certifies **study-level** separation only. It does not claim patient-identity grouping because the frozen supervision surface is keyed by `StudyInstanceUID`. If a trustworthy patient identifier becomes available later, that would require a separately named validation policy rather than silently changing this split.

There is also an important representation-pretraining limitation: the historical B16 encoder was aligned on **all 4,349 non-gold MRI/report pairs**. The 624 PV1 validation studies are therefore not unseen by the historical encoder. For that reason PV1 is valid only for comparing **downstream architecture changes while the exact same B16 encoder remains frozen and shared**. PV1 must not be used to select a new encoder, a new representation-pretraining method, or any change that retrains the encoder using this validation population.

## Original matched controls

The three controls frozen before the first PV1 result were:

```text
PV1-B20   historical B20 architecture, fixed E2
PV1-B31   frozen B31 local-context complementary model, fixed E2
PV1-B33   frozen B33 uniform complementary mean, fixed E2
```

All three used:

```text
same B6 supervision policy
same exact frozen B16 encoder
same 90% post-resize crop
same variable-series policy
same optimizer and LR
same augmentation
same construction seed
same loader seed
same post-construction training seed
same five-epoch scheduler horizon
same fixed E2 endpoint
no validation checkpoint selection
no expert labels
```

The 624 validation studies were never loaded during the matched-control supervised training stage.

## Primary selection metric

Primary metric on the untouched matched-control validation partition:

```text
macro of per-target B6-weighted soft-label BCE
```

For each target, BCE is averaged using the frozen B6 cell weights; the final metric is the unweighted macro average across all 12 targets. **Lower is better.**

Secondary metric:

```text
macro AUC over hard B6 positive/negated states
```

AUC is reported only for targets where both classes occur in the validation partition. It is descriptive and secondary.

Paired study-level bootstrap differences of the primary loss were frozen for B31-vs-B20, B33-vs-B20 and B33-vs-B31.

## Completed original PV1 result

Evaluation version `1.1.0` completed successfully after the low-memory correction. No expert labels were read.

```text
Model   macro weighted soft BCE   secondary macro AUC
B20             0.6155808446          0.5727579473
B31             0.5743065510          0.7567308761
B33             0.5849690647          0.7565223439
```

Primary paired bootstrap:

```text
B31 - B20
median difference            -0.0411411835
95% CI                       [-0.0518219731, -0.0295859090]
P(B31 better)                 1.0000

B33 - B20
median difference            -0.0306878238
95% CI                       [-0.0473458761, -0.0130344232]
P(B33 better)                 0.9992

B33 - B31
median difference            +0.0108619917
95% CI                       [+0.0022462249, +0.0195118995]
P(B33 better)                 0.0050
```

Because lower loss is better, the original frozen PV1 primary ranking is:

```text
B31 > B33 > B20
```

This ranking is now immutable as the original prospective downstream architecture-selection result.

B31 and B33 have almost identical secondary macro AUC despite the significant primary-loss separation. This means the B31-vs-B33 difference is expressed much more strongly in the frozen soft-label loss than in ranking performance.

## Interpretation boundary

The completed result establishes that B31 is the **PV1-selected downstream development architecture** under the exact shared frozen B16 encoder and B6 weak-label contract. It does **not** establish independent expert/clinical superiority. B20 therefore remains the active historical model pending hidden competition or external expert-labelled evidence.

The split itself, primary metric, fixed E2 endpoint, and original B20/B31/B33 result must not be changed in response to observed outcomes.

## Evaluation memory-safety policy

The first PV1 evaluation attempt was terminated by `systemd-oomd` after the terminal scope reached a 56.6 GiB memory peak. The failure was operational, not a model-selection outcome, and no final comparison was produced.

PV1 evaluation v1.1 froze a lower-memory implementation while preserving the exact validation population, checkpoints, `[-1,0,1]` TTA, prediction semantics, and metrics. The evaluator:

```text
loads one checkpoint at a time
predicts the full 624-study validation surface
stores predictions on CPU/disk
releases the model/checkpoint payload
gc.collect() + torch.cuda.empty_cache()
then loads the next checkpoint
```

The evaluation loader is fixed at:

```text
batch_size                  1
num_workers                 1
prefetch_factor             1
persistent_workers          false
series_cache_mb_per_worker  0
```

These settings are resource-management controls only and are not a new experimental degree of freedom.

## Original commands

Create the frozen manifest:

```bash
PYTHONPATH=developments/src python -m rsna_knee.prospective_weak_v1 \
  --config config/current_model.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root "$B6_ROOT" \
  --out runs/prospective_weak_v1/split_manifest.json
```

Train original matched controls:

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

Original evaluation:

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

Canonical original files:

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

## Post-result B29 mechanistic addendum

After the original PV1 result was observed, one additional global mechanism-decomposition experiment was frozen using the already-existing B29 architecture. B29 predates PV1, but the decision to evaluate it on PV1 is post-result; it is therefore **not** retroactively added to the original prospective control set.

The addendum keeps the exact same 2,496-study training subset, fixed B16 encoder, crop, seeds, optimizer, fixed E2 endpoint, 624-study validation partition, TTA, and primary metric. It predeclares only three global comparisons: B29-vs-B20, B29-vs-B33, and B31-vs-B29.

Full protocol and commands: [`PV1_B29_MECHANISTIC_ADDENDUM.md`](PV1_B29_MECHANISTIC_ADDENDUM.md).

Until that addendum completes:

```text
B20   active historical model
B31   original PV1-selected development architecture
B33   frozen simplification comparator
B29   frozen pre-PV1 architecture; addendum pending
B34   not started
```
