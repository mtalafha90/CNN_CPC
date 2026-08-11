# B13 — ImageNet encoder initialization

> **Status:** predeclared, implemented, training ready. Not yet evaluated.

## Single scientific change

The ConvNeXt-Tiny slice encoder starts from **ImageNet-1k weights** instead of
the **B5 competition-only SSL checkpoint**. Everything else is copied unchanged
from B12.1.

## Why this experiment

The ladder so far spans a range narrower than its own measurement noise:

```text
B0  random init         0.4763
B12 best point estimate 0.5661   95% CI [0.5095, 0.6244]
ladder span             0.090
single-measurement CI   0.115
```

Architecture changes (B8 spatial tokens, B9 routing, B12 all-series, B12.1
hierarchical pooling) have all been unresolvable against that noise. A common
cause would explain this better than four independent null results.

Every one of those experiments builds on the same B5 encoder, and that encoder
was trained at a scale far below what self-supervised learning needs:

```text
SSL schedule    8 epochs x 1000 batches x batch_size 3
                = 24,000 samples = 8,000 optimizer steps
                ~ 5.5 passes over the 4,349-study corpus
```

Two problems compound here. Contrastive objectives learn from in-batch
negatives, and at `batch_size 3` each anchor sees roughly four of them;
published contrastive setups use 256-4096 and degrade sharply below ~256. And
8,000 steps is orders of magnitude short of the hundreds of epochs such methods
normally require. The encoder that all downstream experiments sit on is
therefore likely to be weak, and rearranging the layers above a weak encoder
would not be expected to help.

B13 tests that hypothesis directly by substituting an encoder that is known to
have learned general visual features.

## Deliberately unchanged: the encoder learning rate

`b7_encoder_lr` stays at `1e-5`.

That rate was arguably far too low for a weakly initialized encoder, and
raising it is a reasonable separate experiment. But `1e-5` is the standard rate
for fine-tuning a well-pretrained backbone, so leaving it alone is what keeps
B13 a genuine one-variable comparison. Changing initialization *and* schedule
together would leave the result uninterpretable.

A regression test (`tests/test_b13_imagenet_init.py`) pins this: the B13 config
may differ from B12.1 only in the initialization keys and the experiment name.

## Implementation contract

`build_b12_1_model` previously hardcoded `pretrained_weights=False`, so setting
the config flag alone would have produced a B12.1 model while reporting itself
as B13 — a null result that looks real. The flag is now plumbed through, and
the two initialization sources are mutually exclusive:

```python
build_b12_1_model(spec, encoder_state=b5_encoder)      # competition-only SSL
build_b12_1_model(spec, pretrained_weights=True)       # ImageNet
build_b12_1_model(spec, encoder_state=..., pretrained_weights=True)  # ValueError
```

External weights additionally require `allow_external_pretrained: true`;
`pretrained: true` alone raises, so external weights can never load by accident.

## Running it

ImageNet weights are downloaded by torchvision on first use, so the machine
needs internet **once**. They are cached in `~/.cache/torch/hub/checkpoints/`.
For an offline or no-internet notebook environment, copy that cached file
across and set `TORCH_HOME` to its parent.

```bash
python -m rsna_knee.b12_1_training \
  --config configs/b13_imagenet_init.yaml \
  --data-root /path/to/rsna-knee-abnormality-detection \
  --b6-root runs/b6 \
  --series-policy runs/b12/series_policy.json \
  --out-root runs/b13_imagenet
```

`--b5-checkpoint` is **not** passed: under `pretrained: true` the B5 encoder is
replaced entirely, and supplying both is rejected.

Then evaluate exactly as B12.1 does, and compare paired against the retained
benchmark rather than reading the point estimate alone:

```bash
rsna-knee evaluate \
  --train-csv /path/to/train.csv \
  --oof runs/b13_imagenet/oof.csv \
  --compare-oof runs/b12_1_hierarchical/oof.csv \
  --n-bootstrap 5000
```

## Interpreting the result

Read `probability_b_better` and the paired CI, not the point estimate. The
58-study surface gives a ±0.06 interval, so a change under roughly 0.08 is not
resolvable from a single run.

- **Clearly better** — the initialization hypothesis is supported, and the next
  step is to match the schedule to the new initialization (encoder LR, epochs,
  effective batch), each as its own controlled change.
- **Tied or worse** — encoder initialization is not the binding constraint, and
  attention should move to supervision quality or validation power instead of
  further architecture work.

Either outcome is informative, which is the point of running it.
