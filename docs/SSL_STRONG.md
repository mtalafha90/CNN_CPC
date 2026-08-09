# Strong competition-data MRI SSL

> Current campaign status is summarized in [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).

Strong SSL was introduced after the random baseline and fold-safe supervised report teacher showed that the representation itself needed improvement.

## Motivation

Reference results before strong SSL:

```text
B0 random-init macro AUC    = 0.4762536432
report-teacher macro AUC    = 0.49245  (rejected as general teacher)
```

The original short SSL schedule covered only about one corpus pass and used one central triplet per stream. The strong schedule increased non-gold coverage while preserving the rule that no gold study participates in SSL pretraining.

## Strong schedule

```yaml
ssl_output_dir: runs/ssl_strong
ssl_epochs: 8
ssl_max_batches_per_epoch: 1000
ssl_batch_size: 3
ssl_n_slices: 9
ssl_positions_per_stream: 2
ssl_projection_dim: 256
ssl_temperature: 0.15
ssl_metadata_weight: 0.25
ssl_lr: 0.0002
ssl_min_lr: 0.000001
ssl_weight_decay: 0.0001
ssl_noise_std: 0.01
pretrained: false
allow_external_pretrained: false
```

## Data/leakage contract

- only the 4,349 non-gold competition MRI studies are used;
- all 58 gold studies are excluded;
- no ImageNet/external checkpoint is used;
- same-study multi-view examples provide representation positives;
- plane and fluid/structural sequence metadata are auxiliary objectives.

Checkpoint source metadata is `competition_training_data`.

## Completed pretraining run

```text
completed epochs           8
max batches/epoch          1000
completed batches          8000
study draws                24000
approx corpus passes       5.52
active 2.5D examples       238274
loss                       ~3.434 -> ~2.862
```

The loss decreased monotonically across the completed schedule.

Checkpoint:

```text
runs/ssl_strong/ssl_encoder.pt
```

Coverage/history:

```text
runs/ssl_strong/coverage.json
runs/ssl_strong/history.json
```

## Reproduction

```bash
python -m rsna_knee.cli pretrain \
  --config configs/train_local_ssl_pretrain.yaml
```

Focused tests:

```bash
pytest -q \
  tests/test_ssl_multiview.py \
  tests/test_model.py \
  tests/test_sampling_pairing.py
```

## B1 supervised probe

The strong checkpoint was then used to initialize the same Stage-1 architecture/hyperparameters as B0.

```bash
python -m rsna_knee.cli train --config configs/train_local_ssl_strong.yaml --fold 0
python -m rsna_knee.cli train --config configs/train_local_ssl_strong.yaml --fold 1
python -m rsna_knee.cli train --config configs/train_local_ssl_strong.yaml --fold 2
```

Final B1 result:

```text
macro AUC = 0.5030284974
95% CI    = [0.4474281231, 0.5566718294]
```

Compared with B0:

```text
raw point gain          ~ +0.02677
paired median gain        +0.02646
paired 95% CI            [-0.04464, +0.09870]
P(B1 > B0)                0.771
```

## Interpretation

The point estimate supports useful in-domain MRI representation learning, but the 58-study gold set is too small to establish a precise effect.

Subsequent experiments showed:

- B2 lower encoder LR did not improve B1 (`0.4993`);
- B3 pathology-aware MIL did not improve pooled B1 (`0.4945`);
- B4 frozen strong-SSL features produced the best standalone point estimate (`0.5138`).

Thus the strong SSL encoder remains the key representation baseline.

## Relationship to B5

B5 starts from this completed strong SSL checkpoint and adds report-semantic alignment using only the 4,349 report-only competition studies.

The initial B5 comparison deliberately keeps the B4 downstream probe fixed. This asks whether report alignment improves the representation rather than whether another classifier happens to fit the 58 gold cases better.

**B5 is currently running; no B5 performance result is available yet.**
