# B10 — physical-scale normalization

> **Status — 2026-08-10:** **IMPLEMENTED / PREDECLARED / LABEL-FREE GEOMETRY AUDIT PENDING.** B10-v1 is the next one-shot development experiment after B8 and B9 were rejected. B7.1 remains the retained champion at macro AUC `0.5644802945`.

## Motivation

The current B7.1 preprocessing ultimately converts every selected MRI slice to a `224 x 224` matrix. That normalizes matrix dimensions but not physical scale. For example, a scan acquired near `0.15 mm/pixel` and one acquired near `0.33 mm/pixel` can show the same anatomy at different effective network scales after direct resize.

B10 tests whether scanner/protocol scale harmonization improves generalization while preserving the successful B7.1 architecture and supervision recipe.

## Single scientific change versus B7.1

B10 keeps historical B7.1 dual routing and inserts one preprocessing stage:

```text
native DICOM pixels
  -> plane-specific canonical in-plane PixelSpacing
  -> center crop/pad to canonical physical FOV
  -> unchanged 224 x 224 resize
  -> unchanged B7.1 model
```

Everything else remains frozen:

- historical B7.1 dual six-stream routing;
- B5 competition-only image-report encoder initialization;
- B6 v1.2.1 weak supervision;
- 3,120 active weak-training studies;
- 14,123 usable weak cells;
- 16 sampled positions/stream;
- B7.1 ConvNeXt/Transformer/pathology-query architecture;
- batch size 2;
- 4 epochs;
- 1,560 batches/epoch;
- encoder LR `1e-5` and head LR `1e-4`;
- augmentation policy;
- TTA `[-1,0,1]`;
- 5,000 study-level bootstrap replicates;
- zero gold-gradient use;
- zero gold early stopping.

B9 strict routing is **not** inherited because B9 reduced macro AUC to `0.5334962669`.

## Label-free geometry policy

B10 does not hand-pick a target spacing or FOV. The command `rsna-knee-b10-audit` examines only the exact 3,120 active B6 weak-training studies under historical B7.1 routing.

For every selected series it records, when available:

- DICOM `PixelSpacing`;
- Rows / Columns;
- physical row/column FOV;
- SliceThickness;
- SpacingBetweenSlices;
- manufacturer and model;
- field strength.

For each anatomical plane separately, the frozen B10-v1 policy uses the **median valid row/column PixelSpacing** and **median valid row/column physical FOV**. At least 95% of selected series must have valid in-plane geometry or the audit fails. A selected series with missing PixelSpacing uses the historical resize path rather than being discarded.

The audit writes a SHA-256 signature of the exact selected-series mapping. B10 training refuses to run if the current B7.1 routing no longer matches the policy source.

Through-plane spacing is audited but **not normalized in B10-v1**. This keeps the first experiment isolated to in-plane physical scale.

## Install/update

```bash
cd /media/talafha/Disk_1/CNN_CPC
git pull --ff-only origin main
conda activate rsna-knee
python -m pip install -e .
python -c "import rsna_knee; print(rsna_knee.__version__)"
```

Expected package version:

```text
0.15.0
```

## Tests

```bash
python -m compileall -q src tests
pytest -q \
  tests/test_b7_weak_supervision.py \
  tests/test_b9_strict_routing.py \
  tests/test_b10_physical_scale.py
```

## Step 1 — freeze the label-free physical policy

```bash
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"

rsna-knee-b10-audit \
  --config configs/b10_physical_scale.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root runs/b6_report_labels_v121 \
  --out-root runs/b10_physical_scale/audit
```

Inspect before any GPU training:

```bash
cat runs/b10_physical_scale/audit/spacing_audit.json
cat runs/b10_physical_scale/audit/physical_scale_policy.json
```

Required checks:

```text
uses_gold_labels           false
active_weak_training_studies 3120
routing_mode               historical B7.1 dual routing
geometry_coverage          >= 0.95
```

Also inspect the plane-specific `target_spacing_mm` and `target_fov_mm`. Do not alter them after seeing gold performance.

## Step 2 — train B10

Only after the audit is accepted:

```bash
rsna-knee-b10 \
  --config configs/b10_physical_scale.yaml \
  --data-root "$DATA_ROOT" \
  --b5-checkpoint runs/b5_report_ssl/b5_encoder.pt \
  --b6-root runs/b6_report_labels_v121 \
  --physical-policy runs/b10_physical_scale/audit/physical_scale_policy.json \
  --out-root runs/b10_physical_scale
```

Expected training contract per completed epoch:

```text
batches                       1560
study_draws                   3120
active_supervision_cells_seen 14123
positive_cells_seen            6871
negative_cells_seen            7252
budget_limited                 false
```

## Step 3 — frozen gold evaluation

Do not modify spacing/FOV from the gold labels. After four complete epochs:

```bash
rsna-knee-b10-eval \
  --config configs/b10_physical_scale.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b10_physical_scale/b10_model.pt \
  --out-root runs/b10_physical_scale/gold_eval
```

Primary benchmark:

```text
B7.1 macro AUC = 0.5644802945
```

Primary statistical comparison: aligned B7.1 -> B10 study-level paired bootstrap with 5,000 replicates.

## Decision discipline

B10-v1 is one controlled experiment. Do not search multiple target spacings/FOVs on the 58 gold studies and still call the result B10-v1. Do not combine B10 with teacher-student pseudo-labeling, new routing, scanner augmentation, or through-plane normalization until B10-v1 is evaluated on its own.
