# B11.1 — calibration-aware quantile teacher tails

> **Status — 2026-08-10:** IMPLEMENTED / PREDECLARED / LABEL-FREE PSEUDO AUDIT PENDING.

B11-v1 failed its predeclared viability gate despite finding 4,794 pseudo-cells because absolute B7.1 teacher probabilities are strongly target-dependent. Only 23/4,794 accepted cells were positive, Medial Meniscus and Synovitis had zero accepted cells, and Lateral OA had only 21.

A label-free diagnostic showed that this was primarily a calibration problem rather than TTA instability. Examples: Medial Meniscus probabilities span about 0.18–0.89 with stable views, Synovitis about 0.72–0.89 with extremely stable views, while ACL/MCL/Lateral Meniscus are compressed toward low probabilities. Therefore a single global 0.10/0.90 probability gate is inappropriate.

## B11.1 frozen policy

For each target separately, among cells with B6 weight exactly zero:

1. compute the teacher-probability 5th and 95th percentiles using all 4,349 non-gold studies;
2. retain only predictions with TTA range <= 0.05;
3. accept the stable bottom 5% tail as a low pseudo-label and the stable top 5% tail as a high pseudo-label;
4. map accepted tails to soft targets 0.10 and 0.90;
5. use base pseudo weight 0.10;
6. cap total pseudo weight mass independently per target at 15% of the original B6 base-weight mass;
7. never overwrite a B6-supervised cell.

All quantile thresholds are derived without gold labels. B7.1 remains the frozen teacher. Historical B7.1 routing and legacy resize are retained.

## Viability gate before any student implementation/training

The B11.1 pseudo audit must satisfy all of:

```text
>= 2500 pseudo cells overall
>= 100 pseudo cells per target
>= 50 stable low-tail cells per target
>= 50 stable high-tail cells per target
```

If it fails, do not tune the quantile fractions or TTA threshold on gold labels.

## Run

```bash
cd /media/talafha/Disk_1/CNN_CPC
git pull --ff-only origin main
conda activate rsna-knee
python -m pip install -e .
python -c "import rsna_knee; print(rsna_knee.__version__)"

python -m compileall -q src tests
pytest -q tests/test_b11_teacher_student.py tests/test_b11_1_quantile_pseudo.py

export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"
rsna-knee-b11-1-pseudo \
  --config configs/b11_1_quantile_teacher.yaml \
  --data-root "$DATA_ROOT" \
  --teacher-checkpoint runs/b7_1_full_coverage/b7_model.pt \
  --b6-root runs/b6_report_labels_v121 \
  --out-root runs/b11_1_quantile_teacher/pseudo
```

Inspect:

```bash
cat runs/b11_1_quantile_teacher/pseudo/pseudo_summary.json
cat runs/b11_1_quantile_teacher/pseudo/pseudo_policy.json
```

Do not train a B11.1 student until this label-free viability audit passes and the artifacts are frozen.
