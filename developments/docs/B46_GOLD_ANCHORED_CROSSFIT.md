# B46 — Gold-Anchored Cross-Fitted Supervision Test

## Status

**COMPLETED / NO SUPPORT FOR GOLD ANCHOR AT THE FROZEN WEIGHT.**

B46 was frozen before creation of the gold-fold manifest, before any B46
training, and before any B46 out-of-fold result was inspected. All five fixed-E2
folds completed and the pooled 58-study OOF evaluation passed its leakage checks.

| Frozen OOF comparison | Macro AUC |
|---|---:|
| B42 parent | 0.683120 |
| B46 gold-anchored OOF | 0.678174 |
| B46 − B42 | **−0.004946** |

The paired 5,000-replicate bootstrap gave a 95% CI of
`[−0.014664, +0.003402]` and `P(B46 > B42) = 0.1296`; only 5 of 12 targets
improved and every leave-one-target-out delta was negative. This meets B46's
predeclared **no-support** rule. The 4.0 gold-cell weight, fold assignment, and
all B42 parent settings remain closed to post-hoc tuning.

B46 follows the post-B45 plateau retrospective. It tests the highest-priority unresolved hypothesis: whether the current ceiling is substantially caused by mismatch between report-derived weak labels and the official expert image labels.

B46 is **not** a new image architecture. The complete B42 constant-area native-aspect rectangular sparse-MIL image/model contract is retained.

Permanent root:

```text
runs/079_Experiment_B46_gold_anchored_crossfit/
└── b46_gold_anchored_crossfit/
```

## Scientific question

```text
Does adding clean official expert supervision, without changing the B42 image
representation or optimization endpoint, improve cross-fitted ranking on the
58 official expert studies?
```

The 58 expert studies have been reused extensively as diagnostics, so they are no longer treated as pristine untouched validation. B46 instead uses them in a formally cross-fitted manner: a study can enter training for four fold models, but its own OOF prediction must come only from the fifth model whose gradients excluded that study.

## Why B46 exists

The late experiment line produced three displayed hidden-score ties:

```text
B37  0.714
B41  0.714
B42  0.714
```

Meanwhile architecture and geometry variants continued to move the repeatedly reused Expert-58 score by only a few thousandths. B40 reduced weak-label training loss while not improving expert AUC. Phase-9 supervision effects were strongly pathology-dependent. These observations make objective/label mismatch a more important unresolved bottleneck than another top-k, crop, plane-router or center-count variant.

B46 therefore uses the only 696 official clean target cells (`58 x 12`) as a controlled training anchor while preserving leakage-free OOF predictions.

## Frozen parent contract

B46 retains B42 exactly:

```text
full-native percentile normalization
90% native center crop
constant-area native-aspect resize, reference area 448^2
reflection pad only to stride 32
ragged per-series encoding
32 deterministic 2.5D centers, gap=1
6x6 local grid
top-k=8
temperature=1.0
local auxiliary weight=1.0
zero-start target-wise sparse residual gate
final ConvNeXt stage/output norm trainable
B34 non-encoder hierarchy frozen
head LR=1e-4
encoder-tail LR=5e-6
weight decay=1e-4
grad clip=1.0
effective studies/update=2
exactly 2 epochs
no checkpoint selection
TTA [-1,0,+1]
```

The starting checkpoint is the same full-fill B34/Phase-9 base checkpoint used by B42. B46 fold models are trained from that common base, not fine-tuned from the completed B42 endpoint. Thus the intended B42-vs-B46 difference is the training supervision composition.

## Frozen supervision contract

Historical weak/report supervision is unchanged:

```text
report-only studies  4,349
weak usable cells    34,010
B6-preserved all-target LLM fill
```

For fold `f`, B46 adds official labels from the other four gold folds:

```text
held-in gold studies       46 or 47
held-out gold studies      11 or 12
held-in gold targets       exact official hard 0/1
held-in gold active cells  12 per study
gold cell weight           4.0
held-out gold gradients    exactly zero
```

The gold weight **4.0** is frozen before OOF. There is no weight sweep.

Target-balance multipliers are computed from the historical weak supervision **only** and then frozen. Gold additions do not recompute or alter per-target balancing. This prevents the gold anchor from silently changing the relative weak objective across targets.

One clean gold cell therefore contributes four raw weight units before the already-frozen weak-derived target multiplier. This is intended as a conservative anchor: gold cells matter materially but do not dominate the tens of thousands of weak cells.

## Frozen five-fold assignment

The fold manifest is created once from the official 58 x 12 binary labels.

```text
n_folds       5
fold sizes    12 / 12 / 12 / 11 / 11
salt          CNN_CPC|B46|gold-crossfit|2026-08-25
```

Assignment is deterministic greedy multilabel stratification:

1. compute class-state rarity for every study across all 12 targets;
2. assign rarer studies first;
3. for each study choose a non-full fold minimizing squared positive-count deviation from capacity-scaled global prevalence;
4. break ties with SHA-256 using the frozen salt.

Because fold sizes are fixed, balancing positive counts also balances negatives.

The exact generated `gold_folds.json` becomes immutable once created. Every fold checkpoint records its SHA-256.

B46 is label-stratified, not site-grouped. Site/scanner grouping is intentionally deferred to a separate domain-shift audit because site identity is not an official train.csv field and DICOM institution metadata may be incomplete/anonymized. Do not retroactively change B46 folds after inspecting site metadata.

## Leakage rule

For every OOF row:

```text
StudyInstanceUID = X
prediction(X) comes from fold model f
X is in heldout_gold_uids for model f
X is absent from training_gold_uids for model f
heldout_gold_studies_used_in_gradient = 0
```

The evaluator refuses a checkpoint if any of these conditions fail.

## Predeclared OOF decision rule

Primary comparator: historical frozen B42 combined Expert-58 prediction on the exact same 58 UIDs.

Primary quantity:

```text
Delta = B46 cross-fitted OOF macro AUC - B42 combined macro AUC
```

The result is **strong support for label-mismatch as a major bottleneck** only if all are true:

```text
Delta >= +0.010
paired 95% bootstrap CI lower bound > 0
at least 8 of 12 target AUCs improve
all 12 leave-one-target-out macro deltas remain > 0
```

The result is **directional support** if strong support fails but all are true:

```text
Delta >= +0.010
P(B46 > B42) >= 0.90
at least 7 of 12 target AUCs improve
all 12 leave-one-target-out macro deltas remain > 0
```

The result is **no support at the frozen weight** if either:

```text
Delta < +0.005
OR
P(B46 > B42) < 0.65
```

Otherwise the result is **inconclusive**.

The leave-one-target-out requirement prevents another apparent aggregate win driven entirely by one pathology, as occurred in earlier supervision experiments.

## Governance after OOF

Do **not** use the B46 OOF result to change:

```text
gold weight 4.0
fold membership
number of folds
B42 geometry
32 centers
6x6 grid
top-k
learning rates
encoder depth
epoch count
target subset
per-target loss weights
thresholds
model mixtures
```

The completed result closes gold anchoring at this frozen weight. It does not
authorise a B46.1 weight sweep, fold change, target-wise mixture, or a final
all-58-gold model. Any future supervision experiment requires a separately
declared source and endpoint.

## Implementation

```text
config/b46_gold_anchored_crossfit.yaml
developments/src/rsna_knee/b46_gold_crossfit.py
developments/src/rsna_knee/b46_gold_crossfit_manifest.py
developments/src/rsna_knee/b46_gold_crossfit_training.py
developments/src/rsna_knee/b46_gold_crossfit_eval.py
developments/tests/test_b46_gold_crossfit.py
```

## Historical local sequence

```bash
cd /media/talafha/Disk_1/CNN_CPC
conda activate rsna-knee
git pull --ff-only origin main
export PYTHONPATH="$PWD/developments/src:${PYTHONPATH:-}"

pytest -q developments/tests/test_b46_gold_crossfit.py
```

Do not create the manifest unless tests pass.

Set frozen paths:

```bash
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"
export LABELS_ROOT="/media/talafha/Disk_1/CNN_CPC/runs/067_Experiment_LLM_FILL_ALL_b6_preserved_llm_fill_all_targets/b6_plus_llm_fill_all"
export SERIES_POLICY="/media/talafha/Disk_1/CNN_CPC/runs/020_Experiment_B12_variable_series/b12_variable_series/audit/series_policy.json"
export BASE_CHECKPOINT="/media/talafha/Disk_1/CNN_CPC/runs/067_Experiment_LLM_FILL_ALL_b6_preserved_llm_fill_all_targets/b6_plus_llm_fill_all_ft1/train/llm-filled/model.pt"
export B46_ROOT="$PWD/runs/079_Experiment_B46_gold_anchored_crossfit/b46_gold_anchored_crossfit"
export B46_MANIFEST="$B46_ROOT/gold_folds.json"
mkdir -p "$B46_ROOT"
```

Create the manifest exactly once:

```bash
python -m rsna_knee.b46_gold_crossfit_manifest \
  --data-root "$DATA_ROOT" \
  --out "$B46_MANIFEST"
```

Archive the printed SHA-256. Do not recreate the manifest after training begins.

Run all five preflights before training:

```bash
for FOLD in 0 1 2 3 4; do
  python -m rsna_knee.b46_gold_crossfit_training \
    --config config/b46_gold_anchored_crossfit.yaml \
    --data-root "$DATA_ROOT" \
    --labels-root "$LABELS_ROOT" \
    --series-policy "$SERIES_POLICY" \
    --base-checkpoint "$BASE_CHECKPOINT" \
    --fold-manifest "$B46_MANIFEST" \
    --fold "$FOLD" \
    --out-root "$B46_ROOT" \
    --preflight-only || exit 1
done
```

Required for every fold:

```text
[B46 gold preflight] ... PASS
[B46 fold N preflight] PASS
```

Train folds one at a time. Do not launch multiple folds concurrently on the single GPU.

The preferred way to do that is the runner, which verifies the frozen manifest
SHA before every fold, refuses to overwrite any existing checkpoint, and keeps
the Python exit status intact through `tee`:

```bash
bash developments/scripts/run_b46_all_folds.sh
```

**The runner is resumable, and the sequence is longer than one working
session.** A fold that already has a complete checkpoint is skipped rather than
retrained, so running the script again carries on from the first fold that has
none. Nothing is resumed *inside* a fold: each fold is a fixed two-epoch run
from the common base checkpoint, so an interrupted fold simply starts again from
the base. This is an operational convenience only. It does not change the fold
manifest, the gold weight, the architecture, the optimiser, the epoch count or
the decision rule.

Two cases stop the runner rather than being guessed about: a manifest whose
SHA-256 does not match (exit 3), and a checkpoint that exists but is empty
because a save was cut off part-way (exit 4). Deleting a checkpoint is the
operator's decision.

The equivalent loop, if the runner is not used:

```bash
for FOLD in 0 1 2 3 4; do
  python -m rsna_knee.b46_gold_crossfit_training \
    --config config/b46_gold_anchored_crossfit.yaml \
    --data-root "$DATA_ROOT" \
    --labels-root "$LABELS_ROOT" \
    --series-policy "$SERIES_POLICY" \
    --base-checkpoint "$BASE_CHECKPOINT" \
    --fold-manifest "$B46_MANIFEST" \
    --fold "$FOLD" \
    --out-root "$B46_ROOT" \
    2>&1 | tee "$B46_ROOT/fold_${FOLD}/training.log" || exit 1
done
```

After all five fixed-E2 checkpoints exist, evaluate once:

```bash
python -m rsna_knee.b46_gold_crossfit_eval \
  --config config/b46_gold_anchored_crossfit.yaml \
  --data-root "$DATA_ROOT" \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --fold-manifest "$B46_MANIFEST" \
  --run-root "$B46_ROOT" \
  --out-root "$B46_ROOT/oof" \
  --n-bootstrap 5000 \
  2>&1 | tee "$B46_ROOT/oof.log"
```

Required final line:

```text
B46 GOLD-ANCHORED CROSSFIT OOF: PASS
```

## Expected artifacts

```text
b46_gold_anchored_crossfit/
├── gold_folds.json
├── fold_0/
│   ├── b46_fold0_model.pt
│   ├── training_audit.json
│   ├── history.json
│   └── recovery_latest.pt
├── ... fold_1 through fold_4 ...
└── oof/
    ├── b46_oof_predictions.csv
    ├── b46_global_oof_predictions.csv
    └── crossfit.json
```

## Interpretation boundary

B46's OOF prediction is gradient-clean per study, but the 58-study population has influenced earlier project hypotheses and is not independent hidden evidence. The value of B46 is causal/mechanistic: it directly tests whether clean target anchoring helps the unchanged B42 family. A hidden submission is not automatically authorized by B46 and is not part of this protocol.
