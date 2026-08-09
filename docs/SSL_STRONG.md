# Strong competition-data MRI SSL candidate

The frozen random-initialization Stage-1 baseline achieved:

```text
58-study macro OOF AUC = 0.4762536431847217
95% CI                 = [0.42321067215526303, 0.5301667206369031]
```

The fold-safe report-teacher ensemble achieved only `0.49245` macro OOF and was rejected as a general replacement teacher. The next candidate therefore strengthens the MRI representation itself before pathology supervision.

## Why the original SSL schedule was too weak

The original defaults used:

```yaml
ssl_epochs: 4
ssl_max_batches_per_epoch: 300
ssl_batch_size: 4
ssl_n_slices: 5
```

That is only about 4,800 study draws in total for 4,349 non-gold studies, or roughly one corpus pass. In addition, the original SSL objective used only the middle sampled 2.5D triplet from each available sequence.

Version 0.6.0 keeps the leakage-safe non-gold-only data scope but adds multi-position sequence sampling and explicit coverage diagnostics.

## Strong local schedule

Create the strong SSL config from the already verified machine-local config:

```bash
cd /media/talafha/Disk_1/CNN_CPC
conda activate rsna-knee
cp configs/train_local.yaml configs/train_local_ssl_pretrain.yaml

python - <<'PY'
from pathlib import Path
import yaml

p = Path('configs/train_local_ssl_pretrain.yaml')
c = yaml.safe_load(p.read_text())

c['ssl_output_dir'] = 'runs/ssl_strong'
c['ssl_epochs'] = 8
c['ssl_max_batches_per_epoch'] = 1000
c['ssl_batch_size'] = 3
c['ssl_n_slices'] = 9
c['ssl_positions_per_stream'] = 2
c['ssl_projection_dim'] = 256
c['ssl_temperature'] = 0.15
c['ssl_metadata_weight'] = 0.25
c['ssl_lr'] = 2e-4
c['ssl_min_lr'] = 1e-6
c['ssl_weight_decay'] = 1e-4
c['ssl_noise_std'] = 0.01

# This candidate remains competition-data only.
c['pretrained'] = False
c['allow_external_pretrained'] = False
c['ssl_encoder_checkpoint'] = None
c['ssl_checkpoint_source'] = None

p.write_text(yaml.safe_dump(c, sort_keys=False))
print(p)
PY
```

## Regression tests

```bash
pytest -q \
  tests/test_ssl_multiview.py \
  tests/test_model.py \
  tests/test_sampling_pairing.py
```

## Pretrain

```bash
python -m rsna_knee.cli pretrain \
  --config configs/train_local_ssl_pretrain.yaml
```

Expected files:

```text
runs/ssl_strong/ssl_encoder.pt
runs/ssl_strong/history.json
runs/ssl_strong/coverage.json
```

The runtime guard remains active. If the 8.5-hour work budget cannot safely start another epoch, SSL stops cleanly and preserves the completed encoder.

Inspect coverage:

```bash
cat runs/ssl_strong/coverage.json
cat runs/ssl_strong/history.json
```

A useful strong run should show several effective corpus passes rather than approximately one.

## Build the Stage-1 SSL fine-tuning config

Only after `ssl_encoder.pt` exists:

```bash
cp configs/train_local.yaml configs/train_local_ssl_strong.yaml

python - <<'PY'
from pathlib import Path
import yaml

p = Path('configs/train_local_ssl_strong.yaml')
c = yaml.safe_load(p.read_text())
c['output_dir'] = 'runs/stage1_ssl_strong'
c['ssl_encoder_checkpoint'] = str(Path('runs/ssl_strong/ssl_encoder.pt').resolve())
c['ssl_checkpoint_source'] = 'competition_training_data'
c['pretrained'] = False
c['allow_external_pretrained'] = False
c['cotrain_stage1_root'] = None
c['cotrain_stage1_candidates'] = None
p.write_text(yaml.safe_dump(c, sort_keys=False))
print(p)
PY
```

Then train the same three nested folds without changing any pathology-training hyperparameters:

```bash
python -m rsna_knee.cli train --config configs/train_local_ssl_strong.yaml --fold 0
python -m rsna_knee.cli train --config configs/train_local_ssl_strong.yaml --fold 1
python -m rsna_knee.cli train --config configs/train_local_ssl_strong.yaml --fold 2
```

Evaluate:

```bash
python -m rsna_knee.cli evaluate \
  --train-csv "$DATA_ROOT/train.csv" \
  --oof \
    runs/stage1_ssl_strong/fold0/oof.csv \
    runs/stage1_ssl_strong/fold1/oof.csv \
    runs/stage1_ssl_strong/fold2/oof.csv \
  --n-bootstrap 2000 \
  --out runs/stage1_ssl_strong/evaluation.json
```

## Decision rule

Compare against the frozen random baseline `0.4762536432`. Do not select the strong-SSL candidate from outer OOF for downstream fold-specific Stage 2. If it is retained as a Stage-1 candidate, candidate choice for each outer fold still uses that fold's `inner_macro_auc` only through `select-stage1`.

The strong SSL experiment is intended to answer one question cleanly: **does substantially better in-domain knee-MRI representation learning move the image student away from chance before any more aggressive supervision or architecture changes are introduced?**
