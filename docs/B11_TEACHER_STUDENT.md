# B11 — conservative B7.1 teacher–student completion

> **Status — 2026-08-10:** **IMPLEMENTED / PREDECLARED / PSEUDO-LABEL AUDIT PENDING.** B7.1 remains the retained development champion at macro AUC `0.5644802945`. B10 physical-scale normalization scored `0.5523982721` and is rejected as a global replacement.

## Motivation

The frozen B6 export contains 4,349 report-only studies but only 3,120 have at least one usable report-derived target cell. Across all report-only studies there are `4,349 x 12 = 52,188` possible target cells, while B6 directly supervises only 14,123. B11 asks whether a completed B7.1 MRI model can conservatively add information to a subset of the remaining B6-unsupervised cells without allowing pseudo-label mass to dominate the original report supervision.

## Single scientific change versus B7.1

The B11 student keeps:

- B5 competition-only image–report encoder initialization;
- historical B7.1 dual routing;
- historical B7.1 legacy resize (B10 normalization is not inherited);
- the same ConvNeXt/Transformer/pathology-query architecture;
- frozen B6 v1.2.1 supervision;
- frozen B6-derived target-balance multipliers;
- the same optimizer, learning rates, augmentation and four-epoch schedule;
- MRI-only inference;
- TTA `[-1,0,1]` and 5,000 bootstrap replicates;
- zero gold-gradient use and zero gold early stopping.

The only new supervision comes from a **frozen completed B7.1 teacher** on cells where the B6 training weight is exactly zero.

The B11 student is **not initialized from the teacher**. It starts from the same B5 checkpoint as B7.1 so the comparison does not silently add four more fine-tuning epochs to the retained model.

## Frozen pseudo-label policy

For each B6-unsupervised target cell, the B7.1 teacher is evaluated with the same three center-offset views `[-1,0,1]`.

A cell is accepted only when:

```text
teacher mean >= 0.90  OR  teacher mean <= 0.10
AND
max(view probability) - min(view probability) <= 0.05
```

The pseudo-target is the teacher mean probability itself (soft target).

Base pseudo weight:

```text
0.20
```

However, for each target independently, total pseudo base-weight mass is capped at:

```text
25% of that target's original B6 base-weight mass
```

If the raw pseudo mass exceeds that cap, all accepted pseudo cells for the target are scaled down uniformly. B6-supervised cells are never overwritten.

Viability gates before student training:

```text
>= 500 accepted pseudo cells overall
>= 25 accepted pseudo cells for every target
```

If B11-v1 fails these gates, do not loosen thresholds in-place. Define a separately named policy before any gold evaluation.

## Step 1 — install and test

```bash
cd /media/talafha/Disk_1/CNN_CPC
git pull --ff-only origin main
conda activate rsna-knee
python -m pip install -e .
python -c "import rsna_knee; print(rsna_knee.__version__)"

python -m compileall -q src tests
pytest -q \
  tests/test_b7_weak_supervision.py \
  tests/test_b9_strict_routing.py \
  tests/test_b10_physical_scale.py \
  tests/test_b11_teacher_student.py
```

Expected package version:

```text
0.16.0
```

## Step 2 — freeze B11 pseudo labels

```bash
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"

rsna-knee-b11-pseudo \
  --config configs/b11_teacher_student.yaml \
  --data-root "$DATA_ROOT" \
  --teacher-checkpoint runs/b7_1_full_coverage/b7_model.pt \
  --b6-root runs/b6_report_labels_v121 \
  --out-root runs/b11_teacher_student/pseudo
```

Outputs:

```text
runs/b11_teacher_student/pseudo/
├── pseudo_labels.csv
├── pseudo_policy.json
└── pseudo_summary.json
```

Inspect **before student training**:

```bash
cat runs/b11_teacher_student/pseudo/pseudo_summary.json
cat runs/b11_teacher_student/pseudo/pseudo_policy.json
```

Important quantities:

```text
pseudo_cells
combined_active_studies
newly_activated_studies
per-target pseudo positive/negative counts
per-target pseudo_scale
per-target applied_pseudo_weight_mass
teacher_checkpoint_sha256
pseudo_labels_sha256
selected_series_signature
```

## Step 3 — B11 student training

Only after the label-free pseudo audit passes:

```bash
rsna-knee-b11 \
  --config configs/b11_teacher_student.yaml \
  --data-root "$DATA_ROOT" \
  --b5-checkpoint runs/b5_report_ssl/b5_encoder.pt \
  --b6-root runs/b6_report_labels_v121 \
  --pseudo-root runs/b11_teacher_student/pseudo \
  --out-root runs/b11_teacher_student
```

B11 trains four complete passes over every study with at least one B6 or accepted pseudo target. Because pseudo labels may activate some of the 1,229 previously inactive report-only studies, the study count and batches per epoch are derived from the frozen pseudo audit rather than forced to B7.1's 3,120/1,560 values.

The checkpoint records the expected B6 and pseudo cell counts for every full epoch. Do not run gold evaluation unless all four epochs report `full_coverage: true` and `budget_limited: false`.

## Step 4 — frozen gold evaluation

```bash
rsna-knee-b11-eval \
  --config configs/b11_teacher_student.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b11_teacher_student/b11_model.pt \
  --out-root runs/b11_teacher_student/gold_eval
```

Primary benchmark:

```text
B7.1 macro AUC = 0.5644802945
```

Then compare B7.1 -> B11 with the same aligned 5,000-replicate paired bootstrap.

## Development caution

The pseudo-cell selection itself does not use gold labels. Nevertheless, the B7.1 teacher was retained after repeated development on the same 58-study gold surface, so any B11 score remains a development/model-selection estimate. Do not tune pseudo thresholds, per-target pseudo weights, target-specific teacher acceptance, or B7.1/B11 ensembles from the same 58 labels and present them as independent validation.
