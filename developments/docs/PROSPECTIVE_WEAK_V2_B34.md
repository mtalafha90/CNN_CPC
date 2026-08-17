# Prospective weak-validation v2 and B34

> **Frozen post-PV1 mechanism experiment.** PV2 exists to stop further direct optimization against the exposed PV1 validation surface. It is an internal weak-label metric surface, not independent clinical validation and not a historically untouched population.

## Governance sequence

The split policy was committed first (`c7ad1f1f5327fba26c3977a3e91db61b70acbd3a`). B34 was then defined in a separate later commit (`c42e6ab098e0d6e481b60c3293831b4d000e7a46`) before any PV2 membership, training result, or validation result was inspected.

The B34 hypothesis comes from the global B31 context-zero audit, not from target-level PV1 outcomes. The audit showed that zeroing the trained B31 local-context convolution at inference changed the PV1 primary loss by only about `1e-6`, while the trained B31 solution remained clearly better than B29 and B33. This motivates a training-path hypothesis rather than a stronger inference-context mechanism.

## PV2 source and limitation

PV2 uses only the 2,496 studies that belonged to the frozen PV1 training partition:

```text
Original B6-active surface                  3120
├── original PV1 validation                  624  LOCKED; never reused by PV2
└── original PV1 training                    2496  PV2 source pool
    ├── PV2 training                         1997
    └── PV2 validation                        499
```

The 499 PV2 validation studies were historically present in older downstream gradients before PV2 existed. Therefore PV2 is **not** a clean historically unseen holdout. Its valid role is narrower: it is a newly hidden metric surface for a matched retraining experiment defined after PV1 was retired from further architecture invention.

The historical B16 encoder also saw reports from this population. PV2 therefore remains valid only for fixed-encoder downstream mechanism comparisons using the exact shared B16 encoder.

## Frozen assignment

Parent PV1 split SHA-256:

```text
a0032307abb1ab99724eb39fac25332ce131c575f64d823083bb37f5ec20d1e6
```

PV2 salt:

```text
CNN_CPC|prospective-weak-v2|parent-pv1-train|2026-08-17
```

For each UID in the 2,496-study parent PV1 training pool:

```text
key(uid) = SHA256(salt + NUL + StudyInstanceUID)
```

Sort by `key(uid)`. The first 499 become PV2 validation and the remaining 1,997 become PV2 training. The original 624 PV1 validation UIDs are stored in the PV2 manifest as a locked excluded set.

No B6 label, expert label, model output, PV1 result, B29 addendum result, or B31 counterfactual result affects membership. Weak labels are inspected only after assignment for descriptive composition auditing.

## B34 hypothesis

**Hypothesis:** the B31 local-context branch can improve the optimization trajectory even when the learned context weights are not materially needed by the final inference function.

B34 retains B31's train-time scoring path:

```text
H = X + DWConv_k3(LN0(X))
w = softmax(q^T H / sqrt(D))
C = LN0(sum_i w_i X_i)
T = A + tanh(g) * (C - A)
```

where the value sum uses the original slice tokens `X`, exactly as in B31.

At evaluation/inference B34 bypasses the context branch exactly:

```text
H = X
w = softmax(q^T X / sqrt(D))
```

No trained local-context parameter is read by the B34 inference path. The train-time scaffold still contains the same 2,304 zero-initialized depthwise Conv1d parameters as B31, so B34 has the same trainable capacity as B31 during training but the B29-like complementary scorer at inference.

## Essential matched controls

Three runs are required:

```text
PV2-B29   learned complementary query; no local-context scaffold during training or inference
PV2-B31   learned complementary query + local context during training and inference
PV2-B34   learned complementary query + local context during training; exact context bypass at inference
```

B33 remains available as a historical simple comparator but is not required for the primary PV2 B34 mechanism test.

All three required runs use:

```text
same 1,997 PV2 training studies
same exact frozen B16 encoder
same B20 post-resize 90% crop
same B12/B13 all-series policy
same B6 supervision
same optimizer and LR
same augmentation
same fixed E2 endpoint
same five-epoch scheduler horizon
same construction seed offset          +20,000,000
same loader seed offset                +20,100,000
same post-construction RNG seed offset +20,200,000
zero PV2-validation gradients
zero locked-PV1-validation gradients
zero expert gradients
```

## Predeclared PV2 evaluation

Primary metric remains:

```text
macro of per-target B6-weighted soft-label BCE
lower is better
```

Secondary metric remains macro hard-state weak-label ROC AUC where both classes are defined.

Exactly three global paired comparisons are predeclared:

```text
1. B34 - B29   PRIMARY TRAINING-SCAFFOLD TEST
   Same inference-time functional form; B34 alone had the context scaffold during training.

2. B34 - B31   INFERENCE-BYPASS REPLICATION
   Matched training mechanism; B34 removes context only at inference.

3. B31 - B29   REFERENCE CONTEXT-TRAINING COMPARISON
   Checks whether the earlier B31-vs-B29 pattern replicates on PV2.
```

Difference is always candidate macro weighted BCE minus reference macro weighted BCE, so negative favors the candidate.

### Predeclared decision rule

Training-scaffold benefit is supported only if the entire paired 95% interval for `B34 - B29` is below zero.

For the B34-vs-B31 inference simplification, the absolute equivalence margin is frozen at:

```text
±0.001 macro weighted soft BCE
```

B34 and B31 are called equivalent at the chosen metric resolution only if the **entire** paired 95% interval lies inside `[-0.001,+0.001]`.

The combined B34 mechanism is considered successful only when both conditions hold:

```text
B34 - B29 CI entirely < 0
and
B34 - B31 CI entirely inside [-0.001,+0.001]
```

No threshold may be changed after PV2 results are seen.

## Commands

Set paths:

```bash
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"
export B6_ROOT="/media/talafha/Disk_1/CNN_CPC/runs/b6_report_labels_v121"
export SERIES_POLICY="/media/talafha/Disk_1/CNN_CPC/runs/b12_variable_series/audit/series_policy.json"
export B16_ENCODER="/media/talafha/Disk_1/CNN_CPC/runs/b16_full_report/report_ssl/b16_report_encoder.pt"
```

### 1. Create and freeze PV2 manifest

```bash
PYTHONPATH=developments/src \
python -m rsna_knee.prospective_weak_v2 \
  --config config/current_model.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root "$B6_ROOT" \
  --parent-pv1-manifest runs/prospective_weak_v1/split_manifest.json \
  --out runs/prospective_weak_v2/split_manifest.json
```

After creation, archive the manifest and do not modify it.

### 2. Train B29

```bash
PYTHONPATH=developments/src \
python -m rsna_knee.prospective_weak_v2_b29_training \
  --config config/current_model.yaml \
  --data-root "$DATA_ROOT" \
  --split-manifest runs/prospective_weak_v2/split_manifest.json \
  --parent-pv1-manifest runs/prospective_weak_v1/split_manifest.json \
  --b6-root "$B6_ROOT" \
  --series-policy "$SERIES_POLICY" \
  --report-ssl-checkpoint "$B16_ENCODER" \
  --out-root runs/prospective_weak_v2/b29
```

### 3. Train B31 and B34

```bash
for MODEL in b31 b34; do
  PYTHONPATH=developments/src \
  python -m rsna_knee.prospective_weak_v2_training \
    --model "$MODEL" \
    --config config/current_model.yaml \
    --data-root "$DATA_ROOT" \
    --split-manifest runs/prospective_weak_v2/split_manifest.json \
    --parent-pv1-manifest runs/prospective_weak_v1/split_manifest.json \
    --b6-root "$B6_ROOT" \
    --series-policy "$SERIES_POLICY" \
    --report-ssl-checkpoint "$B16_ENCODER" \
    --out-root runs/prospective_weak_v2
done
```

Do not select checkpoints using PV2 validation. All runs stop at the fixed E2 endpoint.

### 4. Evaluate once all three checkpoints exist

```bash
PYTHONPATH=developments/src \
python -m rsna_knee.prospective_weak_v2_eval \
  --config config/current_model.yaml \
  --data-root "$DATA_ROOT" \
  --split-manifest runs/prospective_weak_v2/split_manifest.json \
  --parent-pv1-manifest runs/prospective_weak_v1/split_manifest.json \
  --b6-root "$B6_ROOT" \
  --b29-checkpoint runs/prospective_weak_v2/b29/model.pt \
  --b31-checkpoint runs/prospective_weak_v2/b31/model.pt \
  --b34-checkpoint runs/prospective_weak_v2/b34/model.pt \
  --out-root runs/prospective_weak_v2/eval \
  --n-bootstrap 5000
```

The evaluator uses the same low-memory sequential model-loading policy as corrected PV1 evaluation.

## Expected artifacts

```text
runs/prospective_weak_v2/
├── split_manifest.json
├── b29/
│   ├── model.pt
│   ├── training_audit.json
│   └── history.json
├── b31/
│   ├── model.pt
│   ├── training_audit.json
│   └── history.json
├── b34/
│   ├── model.pt
│   ├── training_audit.json
│   └── history.json
└── eval/
    ├── b29_predictions.csv
    ├── b31_predictions.csv
    ├── b34_predictions.csv
    ├── paired_predictions.csv
    └── comparison.json
```

## Interpretation boundary

PV2 cannot promote B34 to an independently validated model. Even a successful result only supports the global training-scaffold mechanism under a fixed weak-label/frozen-encoder contract. The original PV1 result remains the stronger prospective architecture-selection record, and hidden competition or new external expert-labelled evidence remains necessary before replacing B20 as the active historical model.

Do not create target-wise scaffold masks, target-specific switches, blends, kernel retunes, B34.1, or a new equivalence margin from PV2 outcomes.
