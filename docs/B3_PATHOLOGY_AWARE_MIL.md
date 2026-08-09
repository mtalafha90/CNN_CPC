# B3 pathology-aware stream MIL

B3 is the next controlled architecture candidate after B0 random-init, B1 strong SSL, and B2 differential encoder LR.

## Motivation

The completed experiments show that strong in-domain SSL improves the full 58-study OOF macro AUC from `0.4762536432` to `0.5030284974`, but the tiny inner folds are noisy and B2's lower encoder LR does not produce stable selection gains. The current Transformer model can also fit the weak-supervision objective very rapidly.

B3 therefore changes model capacity and inductive bias rather than optimizer hyperparameters.

## Architecture

B3 keeps the same competition-data SSL ConvNeXt slice encoder, but removes:

- the global MRI Transformer over all stream/slice tokens;
- the pathology-to-pathology Transformer;
- global cross-attention from every pathology query to every MRI token.

Each pathology instead uses a compact two-stage MIL head:

1. target-specific attention over sampled 2.5D positions inside each stream;
2. target-specific attention over the six MRI streams.

The stream attention includes a predeclared **soft** anatomical/sequence prior. Priors are positive for every stream, are never fitted from outer OOF results, and are not hard masks. Missing streams are masked dynamically. A bounded learned residual and content-dependent attention can override the prior.

The six streams remain, in order:

```text
sagittal_fluid
sagittal_structural
coronal_fluid
coronal_structural
axial_fluid
axial_structural
```

## Controlled B3 settings

B3 uses the B1 supervised optimizer again:

```yaml
lr: 0.0001
```

Do **not** carry `encoder_lr` from B2. The single B3 intervention is architecture/capacity plus soft stream priors.

## Local config

```bash
python - <<'PY'
from pathlib import Path
import yaml

src = Path('configs/train_local_ssl_strong.yaml')
dst = Path('configs/train_local_ssl_b3.yaml')
c = yaml.safe_load(src.read_text())

c['output_dir'] = 'runs/stage1_ssl_b3'
c.pop('encoder_lr', None)
c['b3_prior_strength'] = 1.0
c['b3_prior_residual_scale'] = 0.50
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
pytest -q tests/test_pathology_model.py tests/test_model.py tests/test_sampling_pairing.py
```

## Fold-0 gate

```bash
python -m rsna_knee.pathology_training \
  --config configs/train_local_ssl_b3.yaml \
  --fold 0
```

or after editable installation:

```bash
rsna-knee-b3 --config configs/train_local_ssl_b3.yaml --fold 0
```

B3 writes `architecture_policy.json` beside the normal fold outputs so the exact priors and architecture contract are auditable.

The candidate must be judged by the same inner-selection and pooled OOF procedure as B0/B1. Outer fold performance must not be used to tune the B3 prior matrix.
