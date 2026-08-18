# Replacing the encoder with DINOv3

The working model is frozen. This document covers the one experiment allowed to
change it: swapping the frozen report-aligned encoder for DINOv3's
self-supervised ConvNeXt, and nothing else.

## Why this is the experiment worth running

Eight consecutive experiments changed how the model aggregates slices, series
and pathology queries on top of the frozen encoder. Together they moved the
reused-expert estimate by roughly `+0.015`, with every interval crossing zero.
When that many interventions downstream of a fixed representation do nothing,
the representation is the remaining variable.

The usual objection is that swapping the encoder discards the domain
adaptation. The project's own numbers say there is little to discard:

```text
B13  ImageNet ConvNeXt-Tiny            0.6293565948
B15  + knee-MRI SSL                    0.6209002783   (worse)
B16  + full-report semantic alignment  0.6349770242
```

The whole adaptation pipeline is worth about `+0.0056` over plain ImageNet
initialisation, and the knee-MRI SSL step made things worse.

## Why the ConvNeXt variant rather than a ViT

The swap changes only the representation, because the geometry matches exactly:

```text
output width      768   identical to the frozen encoder
input size        224   identical
normalisation     ImageNet mean/std, identical
classifier head   none to strip
weight loading    published under a standard ConvNeXt architecture, no key mapping
```

Nothing above the encoder sees a different shape, so series pooling, the study
Transformer and the pathology queries are untouched. Encoder compute is also
comparable, which matters because the encoder forward dominates runtime: 16
slices x several series x three inference views per study. A ViT-B/16 would be
roughly three to four times the encoder cost and a ViT-L far more.

## Variants

```text
tiny    27.8M params   768-d   drop-in
small   49.5M params   768-d   drop-in
base    87.6M params  1024-d   refused
large  196.2M params  1536-d   refused
```

ConvNeXt tiny and small share channel widths and differ only in depth, so both
emit 768-d features. Base and large are wider and would change the study
representation as well as the features; the encoder refuses them rather than
reshaping silently.

Start with `tiny`: it matches the frozen encoder's capacity, so the comparison
isolates pretraining rather than confounding it with model size. `small` is
available afterwards as a capacity step that still drops in.

## What is held fixed

Everything the frozen contract pins:

```text
crop                deterministic centred 90%
slice sampling      16 positions per series, 2.5D triplets, gap 1
study representation 2-layer Transformer, 8 heads, ff x2
head                12 pathology queries, 1 layer
inference           slice offsets [-1, 0, 1], averaged
endpoint            fixed second epoch
supervision         unchanged
seeds               unchanged
expert labels       never in gradients
encoder             frozen, not updated during training
```

`tests/test_frozen_working_model.py` asserts these, so a change to any of them
fails the suite rather than quietly invalidating the comparison.

## How the swap is performed

The head is built first and its encoder replaced afterwards -- the same shape
of operation as loading report-aligned weights into it:

```python
from rsna_knee.dinov3_encoder import attach_dinov3_encoder

attach_dinov3_encoder(model, variant="tiny", pretrained_weights=True)
```

The replacement inherits the existing encoder's channel count and
normalisation setting, and is refused unless the widths match.

## Weights and licence

Weights resolve through `timm` as `convnext_tiny.dinov3_lvd1689m`, which
publishes the DINOv3 checkpoints under standard ConvNeXt architectures. That
avoids a hand-written key conversion entirely.

They are released under Meta's DINOv3 licence, which permits commercial use.
Read `LICENSE.md` in the upstream repository before relying on that, and note
that competition inference runs offline: the weights must be attached as a
dataset rather than downloaded at runtime.

## What has and has not been verified

Verified here, without network access:

- the DINOv3 encoder emits 768-d features and accepts 224x224 input;
- its normalisation statistics are identical to the frozen encoder's;
- a forward pass through the full model produces finite 12-target logits;
- the encoder is frozen after attachment, with the head still receiving
  gradients;
- wider variants are refused rather than reshaped.

**Not verified here:** that the published DINOv3 weights download and load. That
needs the real checkpoint and network access, so it must be confirmed on the
machine that has both, before a training session is committed:

```bash
python -c "
import timm
m = timm.create_model('convnext_tiny.dinov3_lvd1689m', pretrained=True, num_classes=0)
print('loaded', m.num_features)
"
```

## Reading the result

The comparison is one controlled change against the frozen model, trained
identically. Score it on the prospective weak surface using **macro AUC**, not
soft BCE: a representation change moves calibration, and a proper scoring rule
would happily report a confident improvement that means nothing for a
rank-based metric.

The published evidence is mixed and worth holding in mind. DINOv3 sets a strong
medical baseline without medical pretraining, but does best on modalities that
resemble natural images, and MRI work argues that acquisition-physics
invariances absent from natural-image training matter. So the honest prior is
plausible but unproven, and the experiment is worth running precisely because
the answer is not obvious.
