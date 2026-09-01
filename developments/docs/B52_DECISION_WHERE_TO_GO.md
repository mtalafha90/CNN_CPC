# Where to go after B51 — a decision built from the whole archive

**Written:** 2026-08-30. Revised the same day after the public leaderboard was
checked. Nothing here is a new result; it is a reading of existing evidence.

## The correction that reframes this document

An earlier draft argued that `0.714` was close to the practical ceiling. Its
reasoning was that the nearest published comparator — CoPAS 2024, *Nature
Communications*, the same twelve knee findings — scores `0.812` in-domain but
falls to `0.721-0.726` on external datasets, and that a 19-site, twelve-language
hidden test is an external-generalisation setting.

**The public leaderboard top is `0.952`.**

That reading was wrong. There is roughly `0.24` of headroom, not `0.01`. Someone
is scoring far above published clinical work on this task using the same data,
so the constraint is not the task and it is not domain shift.

Everything that followed from the ceiling argument is withdrawn.

## What the constraint actually is

The answer is in `config/b42_constant_area_aspect_sparse.yaml` and in one line
of the training code. **The model is barely trained.**

```text
b17_encoder_frozen: true          the ConvNeXt that reads pixels is frozen
b7_encoder_lr: 0.0                its learning rate is exactly zero
b37_encoder_trainable_stages: 1   one stage thaws, at 0.05x = 5e-6
b7_n_slices: 16                   16 slices per series, from series up to 320
b7_batch_size: 2                  two studies per step
b7_max_batches_per_epoch: 1560    3,120 studies per epoch, of 4,349
epochs                            2, fixed, no checkpoint selection
```

That is **3,120 optimiser steps in total**, at batch size 2, with the pixel
encoder essentially fixed at its pretrained weights.

And the augmentation is off. The config lists nine augmentation settings —
noise, slice dropout, centre jitter, rotation, translation, scale, gamma, bias
field — then notes that B42 "is constructed with `train=false` and therefore
remains deterministic". Both `b42_constant_area_aspect_sparse_training.py:431`
and `b51_full_population_training.py:202` call
`make_b7_dataset_config(..., train=False)`, and that flag zeroes **every one**
of them:

```python
noise_std   = ... if train else 0.0
slice_dropout = ... if train else 0.0
center_jitter = ... if train else 0
rotation_deg  = ... if train else 0.0
translate_frac = ... if train else 0.0
scale_jitter  = ... if train else 0.0
gamma_jitter  = ... if train else 0.0
bias_field_strength = ... if train else 0.0
```

A frozen encoder, no augmentation, 3,120 steps, and 16 slices out of as many as
320. Against that, an architecture ablation moving `0.003` is not a finding
about architecture.

## Why a careful project ended up here

The governance is genuinely good science. Prospective hypotheses, frozen
endpoints, fixed seeds, no checkpoint selection, no tuning from results — these
are why the archive can state what it knows. `B40` proving that lower training
loss did not raise expert AUC is a real result that many projects never obtain.

But those same rules also **froze a deliberately minimal training regime in
place**, and then required every later experiment to keep it fixed. Determinism
was chosen so runs would be comparable. Two epochs was chosen so nothing could
be selected post hoc. The encoder was frozen so supervision could be isolated.

Each choice was defensible for causal inference. Together they describe a model
that has not been trained, and eight consecutive architecture experiments have
been small perturbations of it.

**Competition ranking and causal inference want opposite things.** "Do not tune
from this result" is correct for the first and fatal for the second.

## The decision

Fork the work. Keep the scientific line if it is valuable in its own right, but
start a competition line that does ordinary competition practice. Ranked by
expected gain per hour:

### 1. Unfreeze the encoder

`b7_encoder_lr: 0.0` and `b17_encoder_frozen: true` mean the part of the network
that actually reads pixels is fixed. One stage thaws at `5e-6`. On 4,349 studies
there is no reason not to fine-tune the whole backbone at a real learning rate.

This is almost certainly the single largest lever in the repository.

### 2. Turn augmentation on

`train=False` disables all nine augmentations. The machinery exists, is
configured, and is switched off. Turning it on costs one flag.

> **This was wrong about the cost, and B52 acted on it.** The flag sets fields
> that `B42ConstantAreaAspectDataset` never reads, so B52 turned it on and
> nothing happened. Applying the augmentation where this dataset can actually
> see it takes a dataset subclass, which is B53. See
> `B53_AUGMENTATION_APPLIED.md`.

### 3. Train properly, and select on validation

Two epochs at 100% and 90.5% of peak learning rate is not a training run. Use a
real schedule to convergence, and **select the checkpoint on a validation
split** — the thing the governance forbids and the thing competitions require.

The scanner-grouped split from B50 already exists and is a reasonable surface.

### 4. Use more of the volume

16 slices from a 320-slice series discards most of the acquisition. `slice_coverage.py`
exists, has never been run, and was marked `GO` twice; it will say how much of a
cruciate ligament the current sampling can even see.

### 5. Ensemble

Seeds and folds, rank-averaged. The metric is macro AUC, so only ranking matters
and calibration does not. This is reliable and needs no new ideas.

## What to stop

- **Do not submit B51.** Not because it will score badly, but because `+0.011`
  is noise against a `0.24` gap, and the submission would teach nothing about
  the real problem.
- **Stop the teacher work.** Four measurements today closed it, and
  `RAISING_AUC_TO_080`'s Tier 1 was already executed: B42 trains on the
  LLM-filled merge and scored `0.714`.
- **Stop architecture ablations on the frozen baseline.** Any mechanism measured
  on an untrained model is measured through a floor.

## One thing worth knowing about the labels

`0.952` is far above published clinical AI for these twelve findings. One
plausible explanation is that the hidden labels are report-derived rather than
expert-adjudicated, which would make them more self-consistent and more
predictable than the 58 expert studies.

The archive contains a hint. Hidden scores have run consistently **above** the
Expert-58 surface — `0.694` hidden against roughly `0.66` local, `0.714` against
`0.683` — an offset the retrospective records as about `+0.033`.

If that is right, then the report-derived validation surfaces are the better
proxy for the hidden test and Expert-58 is the misleading one, which would
partly reverse today's earlier conclusion about B51. It does not change the
ranked plan: a `0.011` question is not worth answering before the training
regime is fixed. But it does mean **model selection should be done on a
report-labelled held-out split, not on the 58 expert studies.**

## What would make this analysis wrong

- **Unfreezing and training properly does not move the hidden score.** Then the
  limitation is representational after all, and B47 becomes the next question.
- **The leaderboard top used external data or pretrained medical weights not
  available here.** Then part of the `0.24` is not reachable, though the training
  regime still is.
- **The hidden labels are expert-adjudicated after all.** Then `0.952` is
  extraordinary and deserves understanding before imitation.
