# B11.1 — calibration-aware quantile teacher tails

> **Status — 2026-08-10:** **PSEUDO AUDIT PASSED / STUDENT IMPLEMENTED / TRAINING READY.** B7.1 remains the retained development champion at macro AUC `0.5644802945`.

## Why B11.1

B11-v1 found 4,794 pseudo-cells but failed its predeclared viability gate because a single absolute `0.10/0.90` teacher-confidence rule was badly mismatched to target-specific probability calibration. Only 23 accepted cells were positive; Medial Meniscus and Synovitis had zero accepted cells.

A label-free diagnostic showed that TTA predictions were generally stable while absolute probability ranges varied strongly by pathology. B11.1 therefore uses **relative per-target teacher tails** rather than one global probability cutoff.

## Frozen pseudo policy

For each target separately, among cells with B6 weight exactly zero:

1. derive the teacher-probability 5th and 95th percentiles from all 4,349 non-gold studies;
2. require TTA probability range `<= 0.05`;
3. stable bottom 5% tail -> pseudo target `0.10`;
4. stable top 5% tail -> pseudo target `0.90`;
5. base pseudo weight `0.10`;
6. cap total pseudo weight mass per target at `15%` of original B6 base-weight mass;
7. never overwrite B6 supervision.

The B7.1 teacher and all thresholds are label-free with respect to the 58 gold studies.

## Frozen audit result

```text
B6 cells                 14123
pseudo cells               3656
combined cells             17779
B6 active studies           3120
combined active studies     3454
newly activated studies      334
viability_passed             true
```

Every target passed the predefined gates:

```text
>= 2500 pseudo cells overall
>= 100 pseudo cells per target
>= 50 stable low-tail cells per target
>= 50 stable high-tail cells per target
```

Observed pseudo tails are well represented across all 12 targets. Total low-tail cells = `1864`; total high-tail cells = `1792`. Synovitis alone reached the 15% pseudo-mass cap, scaling its per-cell pseudo weight from `0.10` to approximately `0.08242`; all other targets retain weight `0.10`.

Frozen pseudo CSV SHA-256:

```text
94f914f3548fab17f67ae0bf1906424bac850268c09ce5febede72b2ed7246b6
```

## Student scientific contract

The B11.1 student starts from the **same B5 encoder initialization as B7.1**, not from the B7.1 teacher. This prevents silently adding another four epochs to the retained checkpoint.

Everything except the added pseudo supervision remains B7.1-equivalent:

- historical B7.1 dual routing;
- legacy direct resize; no B10 physical normalization;
- same ConvNeXt/Transformer/pathology-query architecture;
- same frozen B6 v1.2.1 supervision;
- same B6-derived target-balance multipliers;
- batch size 2;
- same optimizer, learning rates, augmentation and cosine schedule;
- exactly four full-coverage epochs;
- TTA `[-1,0,1]` and 5,000 bootstrap replicates;
- zero gold gradients and zero gold early stopping.

Because the audit activates 3,454 studies, each full epoch is:

```text
studies                    3454
batches                    1727
B6 cells                  14123
pseudo cells               3656
combined cells            17779
pseudo low cells           1864
pseudo high cells          1792
```

## Install / test

```bash
cd /media/talafha/Disk_1/CNN_CPC
git pull --ff-only origin main
conda activate rsna-knee
python -m pip install -e .
python -c "import rsna_knee; print(rsna_knee.__version__)"
```

Expected:

```text
0.18.0
```

Run:

```bash
python -m compileall -q src tests
pytest -q \
  tests/test_b11_1_quantile_pseudo.py \
  tests/test_b11_1_student.py \
  tests/test_b7_weak_supervision.py
```

## Train B11.1

Use the already frozen successful audit; **do not regenerate/tune it from gold results**.

```bash
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"

rsna-knee-b11-1 \
  --config configs/b11_1_quantile_teacher.yaml \
  --data-root "$DATA_ROOT" \
  --b5-checkpoint runs/b5_report_ssl/b5_encoder.pt \
  --b6-root runs/b6_report_labels_v121 \
  --pseudo-root runs/b11_1_quantile_teacher/pseudo \
  --out-root runs/b11_1_quantile_teacher
```

Each completed epoch must report:

```text
expected_full_coverage_batches 1727
expected_full_coverage_studies 3454
b6_cells_expected_per_full_epoch 14123
pseudo_cells_expected_per_full_epoch 3656
pseudo_low_cells_expected_per_full_epoch 1864
pseudo_high_cells_expected_per_full_epoch 1792
full_coverage true
budget_limited false
```

Do not run gold evaluation if any of the four epochs fails those integrity checks.

## Frozen gold evaluation

After four complete epochs:

```bash
rsna-knee-b11-1-eval \
  --config configs/b11_1_quantile_teacher.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b11_1_quantile_teacher/b11_1_model.pt \
  --out-root runs/b11_1_quantile_teacher/gold_eval
```

Primary benchmark:

```text
B7.1 macro AUC = 0.5644802945
```

Then run the same aligned 5,000-replicate B7.1 -> B11.1 paired bootstrap. Do not tune target-specific winners or pseudo parameters from the resulting 58-study comparison.
