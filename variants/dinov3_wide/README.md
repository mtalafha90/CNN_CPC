# Wide DINOv3 encoder — experimental variant

Runs the working model with DINOv3 **base** (1024-d) or **large** (1536-d),
which the supported interface refuses.

This is deliberately outside the supported path. Nothing here is frozen, and a
result from it is not directly comparable to a frozen-width run — see the
caveat at the end, which is the most important thing on this page.

## Why base needs no projection layer

The obvious way to fit a 1024-d encoder under a 768-d head is a projection, and
that would be the wrong move: a randomly initialised 1024→768 layer is a third
variable on top of pretraining and capacity.

It turns out not to be necessary. The head is already width-agnostic —
`HierarchicalSeriesKneeMILNet` reads `self.encoder.out_dim` and sizes the study
Transformer, the pathology queries and the output projection from it. What it
is not is *encoder*-agnostic: it constructs `ConvNeXtSliceEncoder` directly.

So the variant substitutes the encoder class during construction and lets the
head build itself at whatever width follows. No projection, no extra layer.

Two frozen contracts also assert exact parameter counts written as literals for
a 768-d encoder: the complementary query and gate are one vector each (`d`) and
the depthwise k=3 context is `3d`. Their intent is "exactly these shapes for
this width", so the variant re-evaluates those expressions at its own `d`. Both
substitutions are undone afterwards, and tests assert that — including when the
body raises.

## Widths

```text
tiny    27.8M    768-d   head unchanged
small   49.5M    768-d   head unchanged
base    87.6M   1024-d   head widened
large  196.2M   1536-d   head widened
```

## Running it

```bash
python -m variants.dinov3_wide.train \
  --supervision all-script \
  --dinov3-variant base \
  --data-root "$DATA_ROOT" \
  --latin-script-labels "$LATIN_SCRIPT_LABELS" \
  --all-script-labels "$ALL_SCRIPT_LABELS" \
  --series-policy "$SERIES_POLICY" \
  --experiment dinov3_base
```

Output lands in the usual layout, `runs/dinov3_base/train/all-script/`.

Validation and test-set prediction need the variant's own entry points, because
the supported ones rebuild a model at 768-d and cannot reconstruct a wider
checkpoint (they refuse it outright rather than half-loading it):

```bash
python -m variants.dinov3_wide.validate \
  --data-root "$DATA_ROOT" --experiment dinov3_base \
  --checkpoint runs/dinov3_base/train/all-script/model.pt

python -m variants.dinov3_wide.predict \
  --data-root "$DATA_ROOT" --experiment dinov3_base \
  --checkpoint runs/dinov3_base/train/all-script/model.pt
```

Both reuse the supported scoring and submission code and only swap the loader,
so the numbers and the manifest are produced the same way.

No `--encoder-checkpoint`: DINOv3 resolves its own weights through `timm`.
Confirm they download before spending a session:

```bash
python -c "import timm; m=timm.create_model('convnext_base.dinov3_lvd1689m', pretrained=True, num_classes=0); print('loaded', m.num_features)"
```

### Runtime

Measured here on CPU, relative to tiny: base is about **2.5x** the encoder
forward, large considerably more. The encoder dominates the run, so budget
roughly **3.5–4 hours per arm** for base against ~90 minutes for the frozen
model, and check that against your session limit before starting. Large is
likely to exceed a 9-hour budget for two arms.

## How the training loop is reused

The frozen training loop is not copied. Three names are substituted inside
`phase9_matched_supervision_training` for the duration of the call —
the spec factory, the model builder, and the encoder attachment — so the
supervision, population, series exposure, crop, augmentation, optimiser, seeds
and fixed endpoint are the same code, not a reimplementation of it. A copy
would drift from the original; a substitution cannot.

## What is verified, and what is not

Verified without network access, for tiny, base and large:

- the head builds at the encoder's width, with the depthwise context scaling as
  `3d` and no projection anywhere;
- the zero-initialisation contract survives the width change;
- a forward pass yields twelve finite logits and the encoder stays frozen;
- the substituted encoder class and rescaled contracts are restored afterwards,
  including on exceptions, and the supported path still builds at 768.

**Not verified:** that the published DINOv3 weights download and load. That
needs the real checkpoint, so confirm it on the machine that has network access.

## The caveat that decides how to read a result

At base or large the head is widened along with the encoder. So a comparison
against the frozen model moves **two things at once** — encoder pretraining and
head capacity — and a win cannot be attributed to either.

If the question is "does DINOv3 pretraining beat report alignment", use the
supported path at `tiny`, where the head is untouched and pretraining is the
only variable. Use this variant when the question is "does a bigger model help
at all", and read the answer as being about the whole model rather than about
the encoder.
