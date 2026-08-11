# B15 — ImageNet to knee-MRI SSL to B13 hierarchy

> **Status — 2026-08-11:** **IMPLEMENTED / NOT YET RUN.** Package `0.24.0` adds the matched B13-v2 control, B15 MRI-domain SSL, B15 downstream training, strict weak-v2 evaluation and the predeclared paired gate.

## Frozen question

B13 remains the development champion (`0.6293565948`). B14 showed that retaining all `K x 16` slice tokens did not improve global macro AUC, and the completed 17,475-series audit rejected slice-count undersampling as the primary bottleneck.

B15 therefore asks one representation question:

```text
Does adapting the successful ImageNet ConvNeXt-Tiny encoder to competition knee MRI
before the unchanged B13 weakly-supervised hierarchy improve global 12-target ranking?
```

## Frozen weak-v2 surface

```text
surface                 weak_b6_holdout_v2
active B6 studies       3120
weak-train studies      2497
holdout studies          623
holdout usable cells    2875
manifest SHA-256
1a1b07bd690bae3cbb945773c4fcb1c3b0d0f6aa1dd18649d62859aeeb4603d1
```

The manifest is frozen. Do not regenerate it from model performance.

## Two matched downstream arms

### B13-v2 control

```text
ImageNet ConvNeXt-Tiny
        -> B13 hierarchical one-token-per-series model
        -> B6 v1.2.1 training on the 2,497 v2 weak-train studies
```

### B15 candidate

```text
ImageNet ConvNeXt-Tiny
        -> knee-MRI same-study multi-instance contrastive SSL
        -> same B13 hierarchical model
        -> same B6 v1.2.1 training on the same 2,497 studies
```

Both arms construct the same seeded downstream hierarchy before loading the encoder state. The downstream architecture, sampling, B6 soft-target policy, train-only target-balancing derivation, optimizer, augmentation, four epochs and TTA are identical.

## B15 SSL pool and leakage contract

B15 uses the stricter image-held-out SSL policy:

```text
competition studies            4407
fully labelled gold             -58
non-gold studies               4349
frozen v2 weak holdout          -623
------------------------------------
B15 SSL studies                3726
```

Forbidden during SSL:

```text
gold studies/images
v2 holdout studies/images
B6 labels
report labels
model-selection feedback from gold or weak-v2 scores
```

Every eligible repaired real MRI series is retained for each SSL study. The implementation uses five distributed sampled 2.5D positions per acquisition and feeds two distributed positions per real series into the contrastive batch. Examples from the same knee study are positives; examples from the other study in the mini-batch are negatives. This is described as **MICLe-style same-study contrastive adaptation**, not as an exact reproduction of a published implementation.

Frozen SSL optimization:

```text
ImageNet initialization       IMAGENET1K_V1
input normalization          ImageNet mean/std
SSL epochs                   4 full passes
study batch                  2
sampled positions/series     5
used positions/series        2
projection dim               256
encoder LR                   5e-5
projector LR                 5e-4
minimum LR                   1e-6
weight decay                 1e-4
temperature                  0.15
grad clip                    1.0
train gap choices            [1,2]
center jitter                +/-2
```

## Frozen downstream recipe

```text
architecture                 B13 hierarchical series-token model
weak-train studies           2497
slices/series                16
image size                   224
batch size                   2
full batches/epoch           1249
transformer layers           2
transformer heads            8
pathology layers             1
dropout                      0.25
encoder LR                   1e-5
head LR                      1e-4
minimum LR                   1e-6
weight decay                 1e-4
epochs                       4
TTA                          [-1,0,1]
```

B6 policy stays `positive=0.85, weight=0.50; negative=0.05, weight=1.00; uncertain/unmentioned ignored`. Target-balance multipliers are recomputed from the **2,497 training studies only** so the 623 holdout labels do not influence training indirectly.

## Run order

```bash
cd /media/talafha/Disk_1/CNN_CPC
git pull --ff-only origin main
conda activate rsna-knee
python -m pip install -e .

python -c "import rsna_knee; print(rsna_knee.__version__)"
python -m compileall -q src tests
pytest -q tests/test_b15_mri_ssl.py tests/test_weak_validation.py tests/test_b13_imagenet_init.py
```

Expected package version: `0.24.0`.

Set paths:

```bash
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"
export B6_ROOT="runs/b6_report_labels_v121"
export SERIES_POLICY="runs/b12_variable_series/audit/series_policy.json"
export WEAK_V2="runs/weak_holdout_v2"
```

### 1. B15 MRI SSL

```bash
rsna-knee-b15-ssl \
  --config configs/b15_mri_ssl.yaml \
  --data-root "$DATA_ROOT" \
  --weak-holdout-root "$WEAK_V2" \
  --out-root runs/b15_mri_ssl
```

Required before downstream B15:

```text
4 epochs
full_coverage = true every epoch
budget_limited = false every epoch
gold_studies_used = 0
v2_holdout_studies_used = 0
```

Checkpoint:

```text
runs/b15_mri_ssl/b15_ssl_encoder.pt
```

### 2. Matched B13-v2 control

```bash
rsna-knee-b13-v2 \
  --config configs/b15_mri_ssl.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root "$B6_ROOT" \
  --series-policy "$SERIES_POLICY" \
  --weak-holdout-root "$WEAK_V2" \
  --out-root runs/b13_v2_control
```

### 3. B15 downstream

```bash
rsna-knee-b15 \
  --config configs/b15_mri_ssl.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root "$B6_ROOT" \
  --series-policy "$SERIES_POLICY" \
  --weak-holdout-root "$WEAK_V2" \
  --ssl-checkpoint runs/b15_mri_ssl/b15_ssl_encoder.pt \
  --out-root runs/b15_mri_ssl/downstream
```

For both downstream arms, each epoch must report exact full study/cell/series coverage for the frozen 2,497-study v2 train partition.

### 4. Weak-v2 control evaluation

```bash
rsna-knee-b15-weak-eval \
  --config configs/b15_mri_ssl.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b13_v2_control/b13_v2_control.pt \
  --b6-root "$B6_ROOT" \
  --weak-holdout-root "$WEAK_V2" \
  --mode control \
  --out-root runs/b13_v2_control/weak_eval
```

### 5. Weak-v2 B15 evaluation

```bash
rsna-knee-b15-weak-eval \
  --config configs/b15_mri_ssl.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b15_mri_ssl/downstream/b15_model.pt \
  --b6-root "$B6_ROOT" \
  --weak-holdout-root "$WEAK_V2" \
  --mode b15 \
  --out-root runs/b15_mri_ssl/weak_eval
```

Both evaluations use the strict study bootstrap: a replicate is usable only when all 12 target AUCs are defined.

### 6. Predeclared paired gate

```bash
rsna-knee-b15-compare \
  --config configs/b15_mri_ssl.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root "$B6_ROOT" \
  --weak-holdout-root "$WEAK_V2" \
  --control-predictions runs/b13_v2_control/weak_eval/weak_predictions.csv \
  --b15-predictions runs/b15_mri_ssl/weak_eval/weak_predictions.csv \
  --out runs/b15_mri_ssl/weak_eval/b13_v2_vs_b15.json
```

B15 passes only if all three conditions hold:

```text
raw macro delta B15-control > 0
paired median delta > 0
P(B15 > B13-v2-control) >= 0.95
```

If the gate fails, reject B15 without SSL tuning from weak-v2 results. If the gate passes, B15 earns **one** evaluation on the repeatedly reused 58-study gold development surface. Gold remains development confirmation, not independent validation.

## Not allowed inside B15

```text
changing slice count
changing image size
changing hierarchy depth/heads
changing downstream LR or epoch count
using v2 holdout images in SSL
using B6/report/gold labels in SSL
tuning SSL hyperparameters from weak-v2 outcome
per-target B13/B15 winner mixing
```

A later DINOv2/foundation-encoder experiment remains a separate B16 question if B15 fails.
