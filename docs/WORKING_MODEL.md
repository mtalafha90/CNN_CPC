# The working model

One model sits at the top level of this repository. The complete research
lineage that produced it is preserved under [`developments/`](../developments/README.md)
and is not part of the working interface.

## Status

This interface targets the strongest candidate architecture on the project's
internal surfaces. **That is an interface decision, not a promotion.** No model
has been promoted on evidence since B20. Three hidden-competition submissions
have since been recorded (`0.688`, `0.691`, and `0.694` macro AUC), but their
small spread does not establish a replacement model or justify further
leaderboard-driven tuning.

The frozen experiment records under `developments/docs/` therefore still name
B20 as the last promoted model, and they are correct. The disagreement between
those records and this interface is deliberate; an independent result must
still identify a clear, reproducible replacement. See
[`developments/docs/CURRENT_STATUS.md`](../developments/docs/CURRENT_STATUS.md).

## What it does

Each study is a knee MRI examination made up of several acquired series, and
the model predicts twelve binary findings for the study as a whole. The
competition scores the unweighted mean of the twelve ROC AUCs.

```text
MRI study
  -> every eligible real MRI series
  -> 16 sampled slice positions per series
  -> adjacent-slice 2.5D triplets
  -> 224 x 224 tensors
  -> deterministic centred 90% crop, resized back to 224
  -> frozen ConvNeXt-Tiny encoder
  -> attention pooling to one token per series, plus a complementary summary
  -> Transformer over the study's series
  -> 12 pathology queries cross-attending to the study
  -> 12 probabilities
```

Studies carry a varying number of series, so batches are padded and carry a
presence mask rather than being truncated to a fixed count.

## Design decisions worth knowing

**The encoder is frozen.** It was aligned to the reports in an earlier stage
and is not updated during training. Its fingerprint is recorded when a
checkpoint is written and re-verified when one is loaded, so a checkpoint whose
encoder drifted is refused rather than silently scored.

**The crop is fixed, not learned.** Training and inference see exactly the same
geometry.

**Local slice context is training-only.** It shapes the representation during
training and is bypassed exactly at inference, so nothing at prediction time
depends on it.

**Training stops at a fixed epoch.** No checkpoint is selected by looking at a
labelled score. This matters because epoch-to-epoch variation on the small
labelled surface has historically been larger than the differences between
model variants, so choosing an epoch by score would mostly be choosing noise.

**Expert labels never enter the gradient.** The 58 expert-annotated studies are
used only for diagnostics, never for training or checkpoint selection.

## Labels

Training labels come from the radiology reports, not from expert annotation.
Two label surfaces exist:

| Surface    | What it contains                                                        |
|------------|-------------------------------------------------------------------------|
| `latin-script` | the frozen rule-parser labels; multilingual within Latin script, not English-only |
| `all-script`   | the same labels plus cells recovered by translating the Greek- and Cyrillic-script reports before parsing |

The reports are multilingual. The rule parser covers Latin-script vocabulary,
so a substantial share of studies produced no usable labels at all — not
because the reports were silent, but because the parser could not read them.
Translating before parsing recovers most of that population. Whether the extra
labels improve the model is an open question that the aggregate result has not
resolved; see the audit records under `developments/docs/`.

Selecting `all-script` therefore changes what the model learns from, and is a real
experimental choice rather than a formatting detail.

## Interface

```text
model/architecture.py     describe, build and load the working model
model/preprocessing.py    the crop applied to every slice
model/_implementation.py  the single bridge to the preserved implementation
data/dataset.py           studies, series and batching
training/train.py         train on the report labels
validation/validate.py    score the expert studies (a diagnostic, not a test)
testing/test.py           predict the competition test set
```

Only `model/_implementation.py` refers to the historical experiment names. A
test enforces this, so the public interface cannot drift back into experiment
vocabulary.
