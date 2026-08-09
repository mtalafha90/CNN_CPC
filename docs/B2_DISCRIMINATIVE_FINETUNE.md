# B2 discriminative SSL fine-tuning

B2 tests one hypothesis only: the strong in-domain SSL encoder may be useful, but updating it at the same learning rate as randomly initialized Transformer/pathology layers can overwrite those features too quickly.

## Evidence motivating B2

The controlled B0/B1 experiments gave:

- B0 random-init 58-study OOF macro AUC: `0.4762536432`.
- B1 strong-SSL 58-study OOF macro AUC: `0.5030284974`.
- paired bootstrap median B1-B0 difference: `+0.02646`.
- paired 95% interval: `[-0.04464, +0.09870]`.
- bootstrap probability B1 is better: `0.771`.
- nested B0/B1 selection produced only `0.4789929240`, illustrating that the tiny inner folds are too noisy to exploit B1 reliably.

B1 also reduced supervised training loss much faster than B0 while inner AUC often fell after a few epochs. B2 therefore preserves the B1 representation but reduces only the encoder learning rate.

## Single intervention

B2 keeps all B1 settings unchanged except:

```yaml
encoder_lr: 0.00001
```

The normal supervised learning rate remains:

```yaml
lr: 0.0001
```

The encoder therefore starts at one tenth of the head/Transformer LR. Both parameter groups use the same `CosineAnnealingLR` schedule and the existing `min_lr` floor. No encoder freezing is used in B2; freezing is reserved for a later candidate so it is not confounded with differential LR.

## Isolation

`src/rsna_knee/discriminative_training.py` temporarily replaces only the optimizer factory while calling the normal `training.train_fold`. The original optimizer factory is restored in `finally`, so standard B0/B1 training is unchanged.

The run writes `finetune_policy.json` in each fold directory to record the exact optimizer intervention.

## Local config

Create B2 from the already tested B1 config:

```bash
python - <<'PY'
from pathlib import Path
import yaml

src = Path('configs/train_local_ssl_strong.yaml')
dst = Path('configs/train_local_ssl_b2.yaml')
c = yaml.safe_load(src.read_text())

c['output_dir'] = 'runs/stage1_ssl_b2'
c['encoder_lr'] = 1e-5

# Preserve every B1 control setting.
c['ssl_encoder_checkpoint'] = str(Path('runs/ssl_strong/ssl_encoder.pt').resolve())
c['ssl_checkpoint_source'] = 'competition_training_data'
c['pretrained'] = False
c['allow_external_pretrained'] = False
c['cotrain_stage1_root'] = None
c['cotrain_stage1_candidates'] = None

dst.write_text(yaml.safe_dump(c, sort_keys=False))
print(dst)
PY
```

## Test

```bash
pytest -q tests/test_discriminative_training.py tests/test_model.py tests/test_sampling_pairing.py
```

## Fold 0 gate

Run Fold 0 first:

```bash
python -m rsna_knee.discriminative_training \
  --config configs/train_local_ssl_b2.yaml \
  --fold 0
```

or, after editable install:

```bash
rsna-knee-b2 --config configs/train_local_ssl_b2.yaml --fold 0
```

B1 Fold-0 inner macro AUC was `0.5524299965`; B0 Fold-0 inner macro AUC was `0.5723522253`. B2 should improve on B1 without using outer Fold-0 performance for tuning.

If Fold 0 is technically healthy, complete Fold 1 and Fold 2 unchanged before drawing a final conclusion.
