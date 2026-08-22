# B38 — 448 global-only encoder-tail ablation

## Status

**IMPLEMENTED / NOT STARTED.**

B38 is the next permanent experiment after B37 and the native-resolution audit.
It is a fixed post-B37 ablation, not a modification of B37: B37 remains frozen.

## Question

B37 changed several things together: its in-plane resolution, native-order crop,
slice density, sparse local evidence branch, local auxiliary loss, and encoder
tail. B38 asks the narrower question:

> Can the higher-resolution global B34 path improve using only the historical
> sixteen slice centres and one trainable encoder tail stage?

It keeps image resolution and tail adaptation, but removes every sparse-MIL
mechanism. Thus, it does not claim to test the full B37 combined model.

## Frozen B38 contract

~~~text
input preprocessing
  full native-volume percentile normalization
  -> fixed 90% centre crop at native resolution
  -> ONE antialiased bilinear resize to 448x448
  -> the historical B34 sixteen deterministic 2.5D centres, gap=1

model
  B34 global hierarchy only
  B34 non-encoder aggregation always frozen and evaluated
  final ConvNeXt stage + output norm trainable
  all earlier encoder stages frozen
  no local grid
  no sparse-MIL head
  no residual gate
  no local auxiliary loss

optimization
  B6-preserved, all-target LLM-fill supervision
  report-only studies = 4,349
  MRI series = 24,035
  supervision cells = 34,010
  fixed seed = 2026
  micro-batch = 2
  tail reference LR = 1e-4
  encoder-tail scale = 0.05
  effective encoder-tail LR = 5e-6
  fixed epochs = 2
  no expert-label gradients
  no expert checkpoint selection
~~~

The B38 preprocessor is intentionally not the historical B20
resize-then-crop path. It normalizes the full native volume, crops 90% at
native resolution, and resizes only once. The **through-plane positions** are
still exactly the historic 16-centre B34 positions.

## Why this is memory-safe enough to test

B38 keeps the successful B37 low-host-memory operating policy:

- num_workers: 0
- pin_memory: false
- no per-worker series cache
- completed variable-size batches are explicitly released before constructing
  another batch
- periodic RSS, available-host-memory, and CUDA telemetry

It also has half as many slice triplets as B37 (16 rather than 32) and does not
retain a 6x6 spatial feature grid or local auxiliary branch.

## Run protocol

Use the **same exact base checkpoint, LLM-fill label export, and frozen series
policy** used for B37. Do not substitute a no-Synovitis label export, a
retargeted-label checkpoint, or a checkpoint selected on Expert-58.

~~~bash
cd /media/talafha/Disk_1/CNN_CPC
conda activate rsna-knee
export PYTHONPATH="$PWD/developments/src:$PYTHONPATH"

# Set these to the exact existing B37 inputs.
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"
export LLM_LABELS="/absolute/path/to/b6_plus_llm_fill_all"
export SERIES_POLICY="/absolute/path/to/series_policy.json"
export BASE_MODEL="/absolute/path/to/llm-fill/model.pt"
export B38_ROOT="$PWD/runs/073_Experiment_B38_highres_448_global_tail_ablation/b38_highres_global_tail"

test -s "$BASE_MODEL"
test -f "$LLM_LABELS/training_targets.csv"
test -f "$SERIES_POLICY"

# Required no-step forward/backward test on the worst padded two-study batch.
python -m rsna_knee.b38_highres_global_training \
  --config config/b38_highres_global_448.yaml \
  --data-root "$DATA_ROOT" \
  --labels-root "$LLM_LABELS" \
  --series-policy "$SERIES_POLICY" \
  --base-checkpoint "$BASE_MODEL" \
  --out-root "$B38_ROOT" \
  --preflight-only
~~~

Only after a preflight prints '[B38 preflight] PASS', launch the immutable
two-epoch endpoint from a fresh process:

~~~bash
python -m rsna_knee.b38_highres_global_training \
  --config config/b38_highres_global_448.yaml \
  --data-root "$DATA_ROOT" \
  --labels-root "$LLM_LABELS" \
  --series-policy "$SERIES_POLICY" \
  --base-checkpoint "$BASE_MODEL" \
  --out-root "$B38_ROOT" \
  2>&1 | tee "$B38_ROOT/training.log"
~~~

A successful run writes:

~~~text
$B38_ROOT/
├── b38_model.pt
├── history.json
├── training_audit.json
└── recovery_latest.pt
~~~

## Reused Expert-58 diagnostic

After the fixed-E2 checkpoint is written, run one diagnostic replay:

~~~bash
python -m rsna_knee.b38_highres_global_eval \
  --config config/b38_highres_global_448.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint "$B38_ROOT/b38_model.pt" \
  --base-checkpoint "$BASE_MODEL" \
  --out-root "$B38_ROOT/expert58"
~~~

This records the historical 224 full-fill B34 replay and B38’s 448 global
prediction with the same [-1, 0, +1] centre-offset TTA. It writes
expert58.json and both prediction CSVs.

Expert-58 is reused development evidence. B38 is already frozen, so neither
its macro AUC, target AUCs, nor bootstrap interval may be used to change the
resolution, crop, centres, tail depth, learning rate, labels, targets, or epoch
count. Use independent hidden competition evidence to decide whether B38 is
promoted, archived, or followed by a different prospective experiment.
