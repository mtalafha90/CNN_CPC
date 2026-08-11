# B13 — ImageNet encoder initialization protocol

> **Status — 2026-08-11:** IMPLEMENTED / PREDECLARED / TRAINING READY. Package `0.21.0`.

## Scientific question

Does replacing the B5 competition-only encoder protocol with a standard publicly
available ImageNet-pretrained ConvNeXt-Tiny encoder improve the frozen B12.1
hierarchical all-series model?

B13 keeps the B12.1 architecture and training surface fixed and changes the
encoder initialization protocol:

```text
B12.1
ConvNeXt-Tiny encoder <- B5 competition-only SSL checkpoint
input normalization   <- B5 checkpoint policy

B13
ConvNeXt-Tiny encoder <- torchvision IMAGENET1K_V1 weights
input normalization   <- standard ImageNet mean/std
```

The ImageNet weights and their expected normalization are treated as one coherent
encoder-initialization protocol. The repository does **not** describe this as a
literal weight-only change because standard ImageNet normalization differs from
the historical competition-only B5 path.

## Competition-rule status

The competition rules supplied by the repository owner were checked before this
experiment was finalized. Their External Data and Tools section permits external
data and models when they are publicly/equally accessible or otherwise satisfy
the competition reasonableness standard, unless specifically prohibited by the
Host. No competition-specific prohibition on publicly available pretrained models
was present in the supplied rules.

The conservative default remains `pretrained: false`; B13 opts in explicitly with:

```yaml
allow_external_pretrained: true
pretrained: true
```

## Clean experiment separation

B12.1 and B13 have separate trainers, evaluators, variants and CLI commands:

```text
B12.1 trainer   rsna-knee-b12-1
B12.1 evaluator rsna-knee-b12-1-eval
B12.1 checkpoint runs/b12_1_hierarchical/b12_1_model.pt

B13 trainer     rsna-knee-b13
B13 evaluator   rsna-knee-b13-eval
B13 checkpoint  runs/b13_imagenet/b13_model.pt
```

`b12_1_training.py` is competition-only again and requires the B5 checkpoint.
B13 has no B5 checkpoint argument at all.

## Frozen controls versus B12.1

```text
same hierarchical learned series-token architecture
same frozen B12 all-series mapping
same 17,475 eligible training MRI series
same mapping SHA-256:
5c4bb1c52294e45f9e83274c5c07d198dc54811c49b96111b7c8439bd7bcd376
same 3,120 active studies
same 14,123 B6 supervised cells
same 6,871 positive / 7,252 negative cells
same B6 v1.2.1 target/weight policy
same 16 sampled 2.5D positions per series
same 224x224 MRI resize
same plane/fluid/fat metadata embeddings
same 8-head learned per-series attention pooling
same 2-layer study Transformer
same pathology-query heads
same batch size 2
same seed and DataLoader seed offsets
same optimizer
same encoder LR 1e-5
same head LR 1e-4
same augmentation
same four full epochs
same TTA [-1,0,1]
same 5,000 bootstrap replicates
zero gold gradients
zero gold early stopping
```

B13's contract code rejects changes to the optimizer, architecture, epoch count,
augmentation, series mapping, TTA or bootstrap settings.

## Shared initialization control

To keep non-encoder random initialization controlled, B13 first constructs the
complete B12.1 architecture from the same seed using a non-pretrained encoder.
Only after all shared trainable parameters exist does it load the torchvision
ImageNet state into `model.encoder`.

This prevents the external pretrained-weight construction from shifting the RNG
draws used for the study Transformer, pathology tokens, target heads or learned
series-pooling module.

## Install and test

```bash
cd /media/talafha/Disk_1/CNN_CPC
git pull --ff-only origin main
conda activate rsna-knee
python -m pip install -e .
python -c "import rsna_knee; print(rsna_knee.__version__)"
```

Expected:

```text
0.21.0
```

Run the focused regression tests:

```bash
python -m compileall -q src tests
pytest -q \
  tests/test_b12_1_hierarchical.py \
  tests/test_b13_imagenet_init.py
```

The ImageNet-specific test may download the torchvision weights on first use.
The normal torchvision cache is typically under:

```text
~/.cache/torch/hub/checkpoints/
```

## Train B13

Internet is needed only if the ImageNet checkpoint is not already cached.

```bash
export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"

rsna-knee-b13 \
  --config configs/b13_imagenet_init.yaml \
  --data-root "$DATA_ROOT" \
  --b6-root runs/b6_report_labels_v121 \
  --series-policy runs/b12_variable_series/audit/series_policy.json \
  --out-root runs/b13_imagenet
```

There is deliberately **no `--b5-checkpoint` argument**.

Every complete epoch must report:

```text
batches                         1560
study_draws                     3120
active_supervision_cells_seen  14123
positive_cells_seen             6871
negative_cells_seen             7252
series_instances_seen          17475
expected_series_instances      17475
max_series_in_any_batch           14
full_coverage                   true
full_series_coverage            true
budget_limited                  false
```

Do not run gold evaluation unless all four epochs satisfy this contract.

## Frozen gold development evaluation

```bash
rsna-knee-b13-eval \
  --config configs/b13_imagenet_init.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b13_imagenet/b13_model.pt \
  --out-root runs/b13_imagenet/gold_eval
```

Primary comparison versus the B5-initialized parent B12.1:

```bash
python -m rsna_knee.cli evaluate \
  --train-csv "$DATA_ROOT/train.csv" \
  --oof runs/b12_1_hierarchical/gold_eval/gold_predictions.csv \
  --compare-oof runs/b13_imagenet/gold_eval/gold_predictions.csv \
  --n-bootstrap 5000 \
  --out runs/b13_imagenet/gold_eval/b12_1_vs_b13.json
```

Secondary comparison versus B12:

```bash
python -m rsna_knee.cli evaluate \
  --train-csv "$DATA_ROOT/train.csv" \
  --oof runs/b12_variable_series/gold_eval/gold_predictions.csv \
  --compare-oof runs/b13_imagenet/gold_eval/gold_predictions.csv \
  --n-bootstrap 5000 \
  --out runs/b13_imagenet/gold_eval/b12_vs_b13.json
```

And versus the retained B7.1 benchmark:

```bash
python -m rsna_knee.cli evaluate \
  --train-csv "$DATA_ROOT/train.csv" \
  --oof runs/b7_1_full_coverage/gold_eval/gold_predictions.csv \
  --compare-oof runs/b13_imagenet/gold_eval/gold_predictions.csv \
  --n-bootstrap 5000 \
  --out runs/b13_imagenet/gold_eval/b71_vs_b13.json
```

`probability_b_better` is the probability that the `--compare-oof` model is
better than the first `--oof` model under the aligned bootstrap.

## Interpretation policy

The 58 fully labelled studies have been reused throughout sequential development,
so B13's score is a development/model-selection estimate rather than independent
validation. Do not tune target-specific winners, ImageNet variants, normalization,
learning rates, epoch counts or ensemble weights from the B13 gold result.
