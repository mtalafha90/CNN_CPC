# B41: native-aspect-preserving 90% crop sparse-MIL ablation

## Status

**COMPLETED / KAGGLE `0.714` / NOT PROMOTED ABOVE B37 OR B42.**

B41 completed its frozen endpoint and, after the hidden-safe dual-T4 execution
repair, produced a displayed Kaggle score of `0.714`. The score is a displayed
tie with B37 and B42, not evidence for a further B41 preprocessing sweep.

## Purpose

The completed B37 endpoint achieved a Kaggle score of `0.714`. B37 normalized
each full native DICOM volume, cropped the central 90% in native coordinates,
then directly resized the retained matrix to `448x448`. That direct resize
stretches rectangular source matrices.

The completed native-resolution header audit found `24,371` model-eligible
series, all internally consistent in matrix and PixelSpacing, but with genuine
matrix diversity. In particular, it found `258` `640x1280` series (1.06%), as
well as `238` `320x300`, `232` `640x540`, and `148` `384x348` series. B41 is an
isolated answer to that geometry issue; it is not a change to B37 or B40.

B40 has now completed its separate one-epoch optimizer-reset continuation from
B37 E2. Although its training losses continued to fall, its reused Expert-58
combined macro AUC changed from `0.6858177916` for B37 E2 to `0.6847721365` for
B40 E3 (`-0.0010456551`), while the focal-six mean changed by only
`+0.0012755596`. The paired bootstrap 95% interval for B40 minus B37 was
`[-0.0128037347, +0.0096924830]`, with `P(B40 > B37)=0.429`. B40 is therefore
completed and not promoted. B41 was a distinct geometry/preprocessing
candidate rather than a continuation of B40.

## Frozen B41 procedure

1. Percentile-normalize the complete native DICOM volume.
2. Take the central 90% of each native slice matrix.
3. Resize the retained crop exactly once with antialiased bilinear interpolation
   using one common height/width scale factor.
4. Centre-pad the resized crop with zeros into the fixed `448x448` canvas.
5. Keep B37's 32 deterministic 2.5D triplets, 6x6 feature grid, target-specific
   top-k=8 sparse-MIL pooling, B67 all-target fill supervision, one encoder-tail
   stage, memory-safe batch-two loader, and fixed two-epoch endpoint.

For a `640x1280` source series, the intended geometry is:

```text
640x1280 native image
  -> 576x1152 central 90% crop
  -> 224x448 one aspect-preserving resize-to-fit
  -> 448x448 canvas with 112 zero rows above and below
```

The black margins are intentional. They preserve the retained anatomy's matrix
aspect ratio; they do not represent missing DICOM pixels.

“Keep the original” here means preserve the original in-plane aspect ratio and
all pixels inside the 90% crop until the single necessary fixed-canvas resize.
A 448-based CNN cannot retain arbitrary native matrices without resampling.
This policy does not standardize physical millimetres per pixel; PixelSpacing
still varies across acquisitions.

## What B41 does not change

- B37's completed checkpoint and 0.714 score remain immutable.
- B40 is completed/not-promoted and remains immutable as its own duration test.
- B41 does not change the crop fraction, 448 canvas size, slice count, 6x6 grid,
  top-k, labels, base checkpoint, learning rate, encoder stages, or duration.
- No expert labels enter B41 gradients, checkpoint selection, early stopping, or
  hidden-test inference.

## Local commands

Run these commands on the local training machine after pulling `main`.

```bash
cd /media/talafha/Disk_1/CNN_CPC
git pull --ff-only origin main

export PYTHON_BIN=/home/talafha/anaconda3/envs/rsna-knee/bin/python
export DATA_ROOT=/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection
export LABELS_ROOT=/media/talafha/Disk_1/CNN_CPC/runs/067_Experiment_LLM_FILL_ALL_b6_preserved_llm_fill_all_targets/b6_plus_llm_fill_all
export SERIES_POLICY=/media/talafha/Disk_1/CNN_CPC/runs/020_Experiment_B12_variable_series/b12_variable_series/audit/series_policy.json
export BASE_CHECKPOINT=/media/talafha/Disk_1/CNN_CPC/runs/067_Experiment_LLM_FILL_ALL_b6_preserved_llm_fill_all_targets/b6_plus_llm_fill_all_ft1/train/llm-filled/model.pt
export B41_ROOT=/media/talafha/Disk_1/CNN_CPC/runs/076_Experiment_B41_native_aspect_90crop_sparse_mil/b41_highres_aspect_sparse_mil

mkdir -p "$B41_ROOT"

"$PYTHON_BIN" -m rsna_knee.b41_highres_aspect_sparse_training \
  --config config/b41_highres_aspect_sparse_448.yaml \
  --data-root "$DATA_ROOT" \
  --labels-root "$LABELS_ROOT" \
  --series-policy "$SERIES_POLICY" \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --out-root "$B41_ROOT" \
  --preflight-only 2>&1 | tee "$B41_ROOT/preflight.log"
```

Only continue when the command ends with `[B41 preflight] PASS`.

Start the fixed two-epoch endpoint in a user service rather than the terminal
cgroup that was previously killed under memory pressure:

```bash
systemd-run --user \
  --unit=b41-training \
  --collect \
  /usr/bin/systemd-inhibit \
    --what=sleep:idle \
    --mode=block \
    --who="B41 training" \
    --why="Protect fixed B41 endpoint" \
    "$PYTHON_BIN" -m rsna_knee.b41_highres_aspect_sparse_training \
      --config config/b41_highres_aspect_sparse_448.yaml \
      --data-root "$DATA_ROOT" \
      --labels-root "$LABELS_ROOT" \
      --series-policy "$SERIES_POLICY" \
      --base-checkpoint "$BASE_CHECKPOINT" \
      --out-root "$B41_ROOT"

journalctl --user -u b41-training.service -f
```

After the service succeeds, evaluate the one fixed checkpoint with the matching
aspect-preserving Expert-58 loader:

```bash
"$PYTHON_BIN" -m rsna_knee.b41_highres_aspect_sparse_eval \
  --config config/b41_highres_aspect_sparse_448.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint "$B41_ROOT/b41_model.pt" \
  --base-checkpoint "$BASE_CHECKPOINT" \
  --out-root "$B41_ROOT/expert58" \
  --n-bootstrap 5000 2>&1 | tee "$B41_ROOT/expert58.log"
```

`Expert-58` is a reused development diagnostic only. B41 is not changed after
that diagnostic; hidden competition evidence is required for promotion.
