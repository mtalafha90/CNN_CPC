# B42 — constant-area native-aspect rectangular sparse MIL

## Status

**IMPLEMENTED / PREFLIGHT NOT RUN / NOT TRAINED.**

B42 remains a prospective fixed endpoint. Its scientific contract was defined
before implementation and before any B42 Expert-58 or hidden competition result.
B37 and B41 remain immutable.

Implementation files:

```text
config/b42_constant_area_aspect_sparse.yaml
developments/src/rsna_knee/b42_constant_area_aspect_sparse_mil.py
developments/src/rsna_knee/b42_constant_area_aspect_sparse_training.py
developments/tests/test_b42_constant_area_aspect_sparse.py
```

Permanent run root:

```text
runs/077_Experiment_B42_constant_area_aspect_sparse_mil/
└── b42_constant_area_aspect_sparse_mil/
```

## Motivation

B37's direct square resize achieved the proven hidden Kaggle score `0.714`, but
rectangular acquisitions are stretched. B41 corrected the geometry by fitting
the 90% crop inside a `448x448` square and zero-padding the remainder. On the
reused Expert-58 surface:

```text
                         B37 E2          B41 E2          B41 - B37
Global macro             0.6794831901    0.6717205944    -0.0077625956
Combined macro           0.6858177916    0.6778722842    -0.0079455074
Sparse residual gain    +0.0063346016   +0.0061516898     approximately equal
Focal-six combined       0.5841648772    0.5674468541    -0.0167180231
```

The sparse residual contribution remained almost unchanged while the global
representation fell. B42 therefore tests one specific mechanism: whether B41
lost useful representation quality because correct aspect ratio was purchased
by reducing anatomical occupancy/effective spatial sampling.

## Frozen B42 geometry

After full-native-volume normalization and the same central 90% native crop as
B37/B41, let the retained matrix be `H x W`. Define

```text
A0 = 448 * 448 = 200704 pixels
s  = sqrt(A0 / (H * W))
h  = round(H * s)
w  = round(W * s)
```

The same scale factor is applied to both axes, so the anatomy is never stretched
and `h*w` stays approximately equal to the B37 pixel budget.

Examples:

```text
576x576   -> 448x448
576x1152  -> 317x634
1152x576  -> 634x317
```

The resized rectangle is then reflection-padded independently in height and
width only to the next multiple of 32. This is stride alignment, not square
padding. Total added margin is always less than 32 pixels on each axis.

For the important 2:1 example:

```text
576x1152 retained crop
-> 317x634 one isotropic antialiased resize
-> 320x640 thin reflection-aligned tensor
-> approximately 10x20 final ConvNeXt feature map
```

B41 instead placed only `224x448` anatomy inside a `448x448` square. B42 thus
retains essentially the same total anatomical/feature-cell budget as B37 while
preserving B41's correct in-plane aspect ratio.

## Ragged-series encoding

Different MRI series in one study may have different rectangular shapes. B42
never pads them to a shared square. The dataset returns a Python list:

```text
study
├── series 1: [32,3,H1,W1]
├── series 2: [32,3,H2,W2]
└── ...
```

Each readable series is encoded independently at its own rectangle. For every
series, the unchanged ConvNeXt encoder produces one real rectangular feature
map. That map is used in two ways:

1. global average pooling feeds the unchanged frozen B34 hierarchy;
2. adaptive average pooling to `6x6` feeds the unchanged B36/B37 sparse-MIL head.

The local grid remains `6x6` and sparse pooling remains top-k `8`; B42 does not
change local token count or the pathology head.

## Exact effective batch-2 objective

B37 used study micro-batches of two. B42 cannot stack two arbitrary ragged
studies without recreating large padding, so it processes the two studies
sequentially before one optimizer step.

The implementation does **not** simply average the two per-study losses. The
historical target-balanced BCE denominator is known from each study's frozen
supervision weights before the forward pass. For a two-study optimizer batch,
B42 computes each study's effective denominator mass and scales its backward
loss by

```text
study_mass / (study1_mass + study2_mass)
```

before accumulating gradients. Algebraically, this reconstructs the same
weighted numerator/denominator objective that B37 would obtain from a stacked
batch of two, while allowing the first rectangular graph to be freed before the
second is encoded. If both studies have zero usable cells, graph-connected zero
losses are retained just as in the historical implementation.

B42 deliberately reuses B37's construction seed and loader seed. The B42 model
adds no trainable parameters, so sparse-head initialization and shuffled study
order are matched to B37 as closely as the changed geometry permits.

## Frozen model/training contract

Unchanged from B37:

```text
base checkpoint                 exact full-fill B34 checkpoint
training studies                4349 report-only
training series                 24035
supervision cells               34010
expert labels in gradients      0
2.5D centres                    32 deterministic centres
triplet gap                     1
targets                         all 12
local grid                      6x6
sparse top-k                    8
temperature                     1.0
local auxiliary weight          1.0
trainable encoder depth         final ConvNeXt stage + output norm only
head LR                         1e-4
encoder-tail LR                 5e-6
weight decay                    1e-4
grad clip                       1.0
effective studies/update        2
training duration               exactly 2 epochs
TTA evaluation offsets          [-1,0,+1]
```

B42-only choices:

```text
reference anatomical area       448^2 = 200704 pixels
resize                           one isotropic bilinear antialiased resize
aspect ratio                     preserved
stride alignment                multiple of 32 independently per axis
alignment padding               reflect
shared square padding           none
ragged series encoding          yes
```

No stochastic MRI augmentation is introduced.

## Required preflight

Before training, B42 must pass:

- square `448x448` forward path;
- `320x640` representative wide path;
- `640x320` representative tall path;
- a higher-aspect constant-area path;
- the real two-study batch with the largest series counts;
- finite global and combined logits;
- nonzero encoder-tail gradients;
- nonzero sparse evidence-head gradients;
- zero gradients in the frozen non-encoder B34 hierarchy;
- recorded host/CUDA peak memory;
- no optimizer step.

## Local test and preflight

```bash
cd /media/talafha/Disk_1/CNN_CPC
git pull --ff-only origin main
conda activate rsna-knee
export PYTHONPATH="$PWD/developments/src:${PYTHONPATH:-}"

pytest -q developments/tests/test_b42_constant_area_aspect_sparse.py
```

Define the frozen artifacts:

```bash
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"
export LABELS_ROOT="/media/talafha/Disk_1/CNN_CPC/runs/067_Experiment_LLM_FILL_ALL_b6_preserved_llm_fill_all_targets/b6_plus_llm_fill_all"
export SERIES_POLICY="/media/talafha/Disk_1/CNN_CPC/runs/020_Experiment_B12_variable_series/b12_variable_series/audit/series_policy.json"
export BASE_CHECKPOINT="/media/talafha/Disk_1/CNN_CPC/runs/067_Experiment_LLM_FILL_ALL_b6_preserved_llm_fill_all_targets/b6_plus_llm_fill_all_ft1/train/llm-filled/model.pt"
export B42_ROOT="/media/talafha/Disk_1/CNN_CPC/runs/077_Experiment_B42_constant_area_aspect_sparse_mil/b42_constant_area_aspect_sparse_mil"

mkdir -p "$B42_ROOT"
```

Run **preflight only** first:

```bash
python -m rsna_knee.b42_constant_area_aspect_sparse_training \
  --config config/b42_constant_area_aspect_sparse.yaml \
  --data-root "$DATA_ROOT" \
  --labels-root "$LABELS_ROOT" \
  --series-policy "$SERIES_POLICY" \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --out-root "$B42_ROOT" \
  --preflight-only \
  2>&1 | tee "$B42_ROOT/preflight.log"
```

Do not start training unless the final line is:

```text
[B42 preflight] PASS
```

## Fixed training command after PASS

```bash
systemd-run --user \
  --unit=b42-training.service \
  --collect --same-dir \
  /usr/bin/systemd-inhibit --what=sleep:idle --mode=block \
  --who="B42 training" --why="Protect fixed B42 endpoint" \
  "$CONDA_PREFIX/bin/python" -m rsna_knee.b42_constant_area_aspect_sparse_training \
    --config config/b42_constant_area_aspect_sparse.yaml \
    --data-root "$DATA_ROOT" \
    --labels-root "$LABELS_ROOT" \
    --series-policy "$SERIES_POLICY" \
    --base-checkpoint "$BASE_CHECKPOINT" \
    --out-root "$B42_ROOT"
```

Monitor with:

```bash
systemctl --user status b42-training.service --no-pager
journalctl --user -u b42-training.service -f
```

Expected fixed-endpoint artifacts:

```text
b42_model.pt
recovery_latest.pt
history.json
training_audit.json
preflight.log
```

## Evaluation plan

After the fixed E2 endpoint exists, the B42 evaluator must compare the matching
preprocessing paths for:

```text
historical 224 full-fill base
B37 E2 direct-square 448
B41 E2 aspect-fit + zero-pad 448
B42 E2 constant-area rectangular native-aspect
```

Report global macro, combined macro, focal-six mean, per-target AUC, sparse
residual increment, paired B42-minus-B37 bootstrap, paired B42-minus-B41
bootstrap, rectangular tensor-size distribution, and feature-cell occupancy.
Expert-58 remains reused development evidence only.

## Hidden-test governance

B42 is one fixed candidate, not a geometry sweep. Do not change reference area,
padding mode, aspect handling, grid size, top-k, learning rates, target subset,
or epoch count after Expert-58. If the fixed endpoint completes successfully,
one unchanged hidden Kaggle submission is justified. Promotion still requires
hidden competition evidence, with B37's `0.714` as the benchmark.
