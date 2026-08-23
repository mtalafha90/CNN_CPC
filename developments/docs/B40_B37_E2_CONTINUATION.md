# B40 — B37 E2 one-epoch optimizer-reset continuation

## Purpose

B40 is the next independent training candidate while B39 is awaiting hidden
competition evaluation. It does **not** modify B37 or B39.

The completed B37 endpoint used exactly two epochs. Its `b37_model.pt` retains
the full trained model state but not the AdamW moment buffers. A genuine resumed
third B37 epoch is therefore impossible from the saved artifact. B40 records
this fact explicitly rather than pretending otherwise:

```text
start point     exact immutable B37 fixed-E2 model parameters
new training    exactly one full-coverage epoch (called absolute epoch 3)
optimizer       fresh AdamW with the same B37 parameter groups and rates
changed science  optimization duration only
unchanged        data, labels, crop, 448 input, 32 centres, 6x6 grid,
                 top-k=8, auxiliary loss, trainable encoder tail, and seed
```

The new optimizer state is an unavoidable and declared difference. B40 saves its
own optimizer/scaler state in `recovery_latest.pt`, so any future recovery does
not lose it again.

## Fixed B40 contract

- Parent: completed B37 checkpoint, immutable fixed E2 only.
- Parent fingerprint: verified before any DICOM is read.
- Supervision: B37's all-target B6-preserved LLM-fill export, 4,349
  report-only studies / 24,035 series / 34,010 usable cells.
- Image path: B37 90% native centre crop, one 448 resize, 32 centres.
- Model: B37 sparse-MIL 6x6 grid, top-k 8, one encoder-tail stage.
- Optimizer: fresh AdamW; head LR `1e-4`, encoder-tail LR `5e-6`, weight decay
  `1e-4`, gradient clip `1.0`.
- Duration: exactly one extra epoch; no checkpoint selection.
- Expert labels: never used in gradients, stopping, or selection.

## Local run

Use the same B37 input files that already passed B37's training preflight. Do
not substitute an experimental label export or a different base checkpoint.

```bash
cd /media/talafha/Disk_1/CNN_CPC
git pull --ff-only origin main
conda activate rsna-knee
export PYTHONPATH="$PWD/developments/src${PYTHONPATH:+:$PYTHONPATH}"

export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"
export PARENT_B37="$PWD/runs/071_Experiment_B37_highres_448_sparse_mil/b37_highres_sparse_mil/b37_model.pt"
export BASE_MODEL="/path/to/the/exact/B37-base/model.pt"
export LABELS_ROOT="/path/to/the/exact/B37-fill-export"
export SERIES_POLICY="/path/to/the/exact/B37-series_policy.json"
export B40_ROOT="$PWD/runs/075_Experiment_B40_b37_e2_optimizer_reset_continuation/b40_b37_e2_continuation"

test -s "$PARENT_B37"
test -s "$BASE_MODEL"
test -f "$LABELS_ROOT/training_targets.csv"
test -f "$SERIES_POLICY"
mkdir -p "$B40_ROOT"
```

First run the no-step memory/gradient preflight:

```bash
python -m rsna_knee.b40_b37_e2_continuation \
  --config config/b40_b37_e2_continuation.yaml \
  --data-root "$DATA_ROOT" \
  --labels-root "$LABELS_ROOT" \
  --series-policy "$SERIES_POLICY" \
  --parent-checkpoint "$PARENT_B37" \
  --base-checkpoint "$BASE_MODEL" \
  --out-root "$B40_ROOT" \
  --preflight-only
```

Only after `[B40 preflight] PASS`, launch the fixed one-epoch endpoint from a
fresh user service. This avoids putting the memory-heavy process back inside the
GNOME terminal cgroup that `systemd-oomd` killed during the first B37 attempt.

```bash
systemd-run --user \
  --unit=b40-training.service \
  --collect --same-dir \
  /usr/bin/systemd-inhibit --what=sleep:idle --mode=block \
  --who="B40 training" --why="Protect B40 endpoint" \
  "$CONDA_PREFIX/bin/python" -m rsna_knee.b40_b37_e2_continuation \
    --config config/b40_b37_e2_continuation.yaml \
    --data-root "$DATA_ROOT" \
    --labels-root "$LABELS_ROOT" \
    --series-policy "$SERIES_POLICY" \
    --parent-checkpoint "$PARENT_B37" \
    --base-checkpoint "$BASE_MODEL" \
    --out-root "$B40_ROOT"

systemctl --user status b40-training.service --no-pager
journalctl --user -u b40-training.service -f
```

Expected output files:

```text
b40_model.pt
recovery_latest.pt       # includes optimizer and scaler state
history.json
training_audit.json
```

## Evaluation and promotion

After the fixed E3 checkpoint is written, compare it to the same B37 E2 parent
and historical 224 base through the same three-offset Expert-58 replay:

```bash
python -m rsna_knee.b40_highres_sparse_eval \
  --config config/b40_b37_e2_continuation.yaml \
  --data-root "$DATA_ROOT" \
  --parent-checkpoint "$PARENT_B37" \
  --checkpoint "$B40_ROOT/b40_model.pt" \
  --base-checkpoint "$BASE_MODEL" \
  --out-root "$B40_ROOT/expert58" \
  2>&1 | tee "$B40_ROOT/expert58.log"
```

It writes `$B40_ROOT/expert58/b40_vs_b37_expert58.json`, including B40-minus-B37
macro and focal-six deltas plus a paired bootstrap. This is reused development
evidence only: it may not select a different B40 duration or alter B40 settings.

B40 remains a new candidate. B37 and B39 stay immutable. A B40 Kaggle
submission should be created only as a separately documented inference endpoint
after this diagnostic and should never replace B39 retroactively.
