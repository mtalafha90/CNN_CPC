# B2 discriminative SSL fine-tuning

B2 tested one hypothesis: the strong in-domain SSL encoder might be useful, but updating it at the same learning rate as randomly initialized Transformer/pathology layers could overwrite those features too quickly.

> Current campaign status is summarized in [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).

## Evidence motivating B2

The controlled B0/B1 experiments gave:

- B0 random-init 58-study OOF macro AUC: `0.4762536432`.
- B1 strong-SSL 58-study OOF macro AUC: `0.5030284974`.
- paired bootstrap median B1-B0 difference: `+0.02646`.
- paired 95% interval: `[-0.04464, +0.09870]`.
- bootstrap probability B1 is better: `0.771`.
- nested B0/B1 selection produced only `0.4789929240`, illustrating that the tiny inner folds are too noisy to exploit B1 reliably.

B1 also reduced supervised training loss rapidly while inner AUC often fell after a few epochs. B2 therefore preserved the B1 representation but reduced only the encoder learning rate.

## Single intervention

B2 kept all B1 settings unchanged except:

```yaml
encoder_lr: 0.00001
```

The normal supervised learning rate remained:

```yaml
lr: 0.0001
```

The encoder therefore started at one tenth of the head/Transformer LR. Both parameter groups used the same `CosineAnnealingLR` schedule and existing `min_lr` floor. No encoder freezing was used in B2.

## Isolation

`src/rsna_knee/discriminative_training.py` temporarily replaces only the optimizer factory while calling the normal `training.train_fold`. The original optimizer factory is restored in `finally`, so standard B0/B1 training is unchanged.

Each fold writes `finetune_policy.json` to record the exact optimizer intervention.

## Reproduction

```bash
pytest -q tests/test_discriminative_training.py tests/test_model.py tests/test_sampling_pairing.py

rsna-knee-b2 --config configs/train_local_ssl_b2.yaml --fold 0
rsna-knee-b2 --config configs/train_local_ssl_b2.yaml --fold 1
rsna-knee-b2 --config configs/train_local_ssl_b2.yaml --fold 2
```

## Final result

B2 completed all three folds.

```text
pooled macro AUC = 0.4993244663
95% CI           = [0.4512751879, 0.5464103264]
```

Per-target AUC:

| Target | AUC |
|---|---:|
| ACL | 0.4841 |
| MCL | 0.5442 |
| Medial Meniscus | 0.5649 |
| Lateral Meniscus | 0.5068 |
| Medial OA | 0.5411 |
| Lateral OA | 0.5242 |
| PF OA | 0.5508 |
| Effusion | 0.4484 |
| Synovitis | 0.3883 |
| Baker's | 0.5580 |
| Contusion | 0.4588 |
| Fracture | 0.4222 |

Paired B1-versus-B2 analysis gave a median B2-B1 difference of about `-0.00395`, 95% CI `[-0.05905, +0.05269]`, with `P(B2 > B1)=0.4506`.

## Decision

**Rejected.** Reducing the encoder learning rate did not produce a stable improvement over B1. The observed pattern is more consistent with small-gold-set variance / weak-supervision and downstream-model limitations than with simple catastrophic forgetting.

No further tuning of the B2 encoder LR is planned on the same 58-study outer OOF set.
