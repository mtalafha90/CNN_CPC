# B3 pathology-aware stream MIL

B3 tested whether a lower-capacity, pathology-aware MIL head could use the strong competition-only SSL encoder more effectively than the global Transformer/pathology-query architecture.

> Current campaign status is summarized in [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).

## Motivation

B0 random initialization reached `0.4762536432` macro OOF AUC and B1 strong SSL reached `0.5030284974`, while B2's lower encoder learning rate did not improve reliably. B3 therefore changed architecture/capacity rather than optimizer hyperparameters.

## Architecture

B3 keeps the strong competition-data SSL ConvNeXt slice encoder but removes:

- the global MRI Transformer over all stream/slice tokens;
- the pathology-to-pathology Transformer;
- global cross-attention from every pathology query to every MRI token.

Each pathology instead uses a compact two-stage MIL head:

1. target-specific attention over sampled 2.5D positions inside each stream;
2. target-specific attention over the six MRI streams.

The stream attention includes a predeclared **soft** anatomical/sequence prior. Priors are positive for every stream, were not fitted from outer OOF results, and are not hard masks. Missing streams remain dynamically masked and learned content-dependent residual attention can override the prior.

The six streams are:

```text
sagittal_fluid
sagittal_structural
coronal_fluid
coronal_structural
axial_fluid
axial_structural
```

## Controlled settings

B3 returned to the B1 supervised optimizer:

```yaml
lr: 0.0001
b3_prior_strength: 1.0
b3_prior_residual_scale: 0.50
```

The B2 `encoder_lr` intervention was not carried into B3.

## Reproduction

```bash
pytest -q tests/test_pathology_model.py tests/test_model.py tests/test_sampling_pairing.py

rsna-knee-b3 --config configs/train_local_ssl_b3.yaml --fold 0
rsna-knee-b3 --config configs/train_local_ssl_b3.yaml --fold 1
rsna-knee-b3 --config configs/train_local_ssl_b3.yaml --fold 2
```

B3 writes `architecture_policy.json` beside the normal fold outputs so the priors and architecture contract remain auditable.

## Final result

```text
pooled macro AUC = 0.4944652486
95% CI           = [0.4314514263, 0.5578825232]
```

Per-target AUC:

| Target | AUC |
|---|---:|
| ACL | 0.5650 |
| MCL | 0.5329 |
| Medial Meniscus | 0.5577 |
| Lateral Meniscus | 0.5925 |
| Medial OA | 0.5659 |
| Lateral OA | 0.4429 |
| PF OA | 0.5097 |
| Effusion | 0.3764 |
| Synovitis | 0.4456 |
| Baker's | 0.4112 |
| Contusion | 0.4615 |
| Fracture | 0.4722 |

Against B1, the paired median B3-B1 difference was about `-0.00806`, 95% CI `[-0.06105, +0.04045]`, with `P(B3 > B1)=0.3808`.

B3 improved some targets, especially ACL and Lateral Meniscus, but degraded others. Selecting target-specific B1/B3 winners after inspecting outer OOF would be optimistic and is not used.

## Fixed B1+B3 ensemble follow-up

A predeclared 50:50 rank ensemble gave:

```text
macro AUC = 0.5048038179
```

The gain over B1 was only about `+0.00178`, with `P(rank ensemble > B1)=0.5578`. This is effectively neutral.

## Decision

**Rejected as a global replacement for B1.** The pathology-aware architecture contains some complementary signal but did not improve pooled macro AUC reliably. No post-hoc target-wise model picking is permitted from these outer OOF results.
