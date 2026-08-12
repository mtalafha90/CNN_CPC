# Local Real-Data Training Runbook

> **Current stage — 2026-08-12:** package `0.24.1`. **B13 is the reused-gold development champion at macro AUC `0.6293565948`. B15 is fully completed: it passed frozen weak-v2 but scored `0.6209002783` on its single reused-gold confirmation and did not replace B13. The next step is a B6 report-state audit, not another B15 training run.** See [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).

## Environment

```text
Conda environment: rsna-knee
GPU: NVIDIA RTX A4500 Laptop GPU
precision: bf16
one visible GPU
```

Verified local paths used in the completed campaign:

```text
repo:      /media/talafha/Disk_1/CNN_CPC
data root: /media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection
```

Start a terminal:

```bash
export REPO="/media/talafha/Disk_1/CNN_CPC"
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"
export CUDA_VISIBLE_DEVICES=0
cd "$REPO"
conda activate rsna-knee
```

## Pull/install

```bash
git checkout main
git pull --ff-only origin main
python -m pip install -e .
python -c "import rsna_knee; print(rsna_knee.__version__)"
```

Expected current package version:

```text
0.24.1
```

## Current measured ladder

```text
B0 random                         0.4762536432
B1 strong SSL                    0.5030284974
B4 frozen SSL + classical        0.5137567459
B5 image-report SSL              0.5243650851
B7-v1 weak supervision           0.5397724412
B7.1 full coverage               0.5644802945
B8 spatial anatomy               0.5300962807
B9 strict routing                0.5334962669
B10 physical scale               0.5523982721
B11.1 quantile pseudo labels     0.5506902702
B12 all real series              0.5660915179
B13 ImageNet hierarchy           0.6293565948   DEVELOPMENT CHAMPION
B14 ImageNet full tokens         0.6197914249   REJECTED GLOBALLY
B15 MRI SSL hierarchy            0.6209002783   NO GLOBAL GOLD IMPROVEMENT
```

B11-v1 failed its viability gate; B12.1 was implemented but skipped.

## Key retained artifacts

```text
runs/b5_report_ssl/b5_encoder.pt
runs/b6_report_labels_v121/
runs/b12_variable_series/audit/series_policy.json
runs/b13_imagenet/b13_model.pt
runs/weak_holdout_v2/
runs/b15_mri_ssl/b15_ssl_encoder.pt
runs/b13_v2_control/b13_v2_control.pt
runs/b15_mri_ssl/downstream/b15_model.pt
runs/b13_v2_control/weak_eval/
runs/b15_mri_ssl/weak_eval/
runs/b15_mri_ssl/gold_confirmation/
```

Preserve these artifacts; they are the campaign audit trail.

## Frozen weak-v2 contract

```text
surface                 weak_b6_holdout_v2
train studies           2497
holdout studies          623
holdout cells           2875
report-group overlap       0
manifest SHA
1a1b07bd690bae3cbb945773c4fcb1c3b0d0f6aa1dd18649d62859aeeb4603d1
```

Do not regenerate this split from model outcomes.

## B15 completed training integrity

### SSL

```text
SSL studies             3726
series/pass            20534
batches/epoch           1863
4 full epochs
loss 2.70946 -> 2.47569
gold images used            0
v2 holdout images used      0
```

### Matched downstream arms

Both control and B15 used:

```text
2497 studies
13974 real MRI series
11248 B6 cells
5464 positive / 5784 negative
1249 batches/epoch
4 complete epochs
```

Control final loss: `0.6622741637`.  
B15 final loss: `0.6065262400`.  
Training loss is not a selection metric.

## B15 completed validation

Weak-v2:

```text
control                0.5652498118
B15                   0.7319060415
paired median         +0.1675245839
95% paired CI         [+0.1124433208,+0.2165156305]
P(B15 > control)       1.0000
predeclared gate       PASS
```

One-look reused gold:

```text
B15                   0.6209002783
95% CI               [0.5706720829,0.6675892903]
B13                   0.6293565948
raw B15-B13          -0.0084563164
```

B13 remains retained.

## Evaluation runtime note

The B15 weak evaluator uses three-view TTA and can create multiprocessing cleanup warnings with several workers. The completed evaluations succeeded with the standard config. If a terminal/session is unstable, runtime-only worker settings may be reduced without changing the scientific evaluation contract, provided model/TTA/bootstrap settings remain frozen.

Do not change:

```text
b7_eval_batch_size = 2
b7_eval_tta_offsets = [-1,0,1]
b7_n_slices = 16
b7_image_size = 224
b7_n_bootstrap = 5000
```

## Current next task

Do **not** rerun B15 or change its SSL/downstream hyperparameters from the gold result.

The next diagnostic is a B6 state audit on the already-reused gold studies:

```text
positive
negated
uncertain
unmentioned
```

For each target/state quantify counts, expert-positive fraction, expert-negative fraction and coverage. In particular, do not assume `unmentioned = negative`.

Any new supervision policy must receive a new version/name and be frozen before model evaluation.

## Reporting discipline

The 58 gold studies are repeated development/model-selection data. Weak-v2 is B6 teacher agreement only. Do not select target-specific winners, optimize ensemble weights, regenerate weak-v2, retune B15 from gold, or describe local development AUC as independent hidden-test performance.

The hidden Kaggle evaluation remains the next genuinely independent model-performance signal.