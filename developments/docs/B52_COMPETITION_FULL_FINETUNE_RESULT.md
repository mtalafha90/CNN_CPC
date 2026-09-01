# B52 — training the model, and what it was worth

**Status:** first run complete on the B50 gate's `train` rows. Not a submission
candidate. The full-data run is the follow-up.

> **Correction, found after both runs.** B52 reports three changes. Only two of
> them happened. `augment=True` sets fields on a `DatasetConfig` that
> `B42ConstantAreaAspectDataset` never reads, so every epoch trained on
> byte-identical pixels. Measured: two draws of the same study with augmentation
> on are identical to each other and to the same study with it off, to `0.0`.
>
> The gains below are real and came from the encoder training and the completing
> schedule. The trainer's `augment=...` line, and the `augmentation_enabled`
> field in every B52 checkpoint, describe a flag rather than a behaviour and
> should not be read as evidence that augmentation ran.
>
> `B53_AUGMENTATION_APPLIED.md` has the two causes, the measurement, and the
> corrected experiment.

## The question

Eight architecture experiments since B37 searched for a missing mechanism and
none moved the hidden score. B52 asked whether there was a mechanism missing at
all, or whether the model underneath those experiments had simply never been
trained.

The frozen contract every one of them inherited:

```text
b17_encoder_frozen: true               the network that reads pixels is frozen
b7_encoder_lr: 0.0                     its learning rate is exactly zero
b37_encoder_trainable_stages: 1        one stage of five, at 0.05x = 5e-6
b7_max_batches_per_epoch: 1560         3,120 studies per epoch, of 4,349
epochs                                 2, fixed, no checkpoint selection
make_b7_dataset_config(train=False)    all nine augmentations zeroed
```

3,120 optimiser steps at batch size 2, no augmentation, pixel encoder fixed.

## What B52 changed, and what it held fixed

```text
changed     five encoder stages instead of one, at 0.10x rather than 0.05x
            augmentation on
            six epochs with a cosine whose T_max equals the epochs run
            checkpoint chosen by validation, not fixed at epoch 2

fixed       B42 geometry, asserted through require_b42_contract
            the sparse-MIL head, grid 6x6, top-k 8
            the supervision policy and the merged B6+LLM label export
            the Phase-9 llm_fill base checkpoint
            the B50 scanner-grouped gate
```

Trainable parameters went from the head's 50,712 plus one encoder stage to
**46,506,660**: encoder `27,503,232`, study hierarchy `18,952,716`, head `50,712`.

## Result

Gate `fa8eb88ff3d5fc493e82de7bf5067c48ecc76e0e26c4e5b646b554a99730326f`,
1,447 training studies, 548 unseen-scanner validation studies, seed 2026.

```text
epoch   train      validation   macro AUC    minutes
  1     1.174172   1.083415     0.771784     132.6
  2     1.056359   1.071391     0.783640     116.5
  3     1.012619   1.076867     0.787542     116.5
  4     0.971698   1.060799     0.799119     116.3
  5     0.943779   1.115156     0.802666     116.6     <- selected
  6     0.932740   1.079549     0.794551     123.4
```

Twelve hours in total. Peak GPU memory `1.39 GiB` of 16.

Against the two arms B50 ran on the identical surface:

```text
B50 frozen control          0.763117
B50 adapted hierarchy       0.774336
B52 best (epoch 5)          0.802666

B52 - B50 frozen control   +0.039549
B52 - B50 adapted          +0.028330
```

B50's adapted hierarchy — the effect that required a matched-arm experiment to
establish, and which B51 then carried to the full population — was worth
`+0.011219`. Training the model is worth **3.5 times** that on the same studies.

## The turnover is the informative part

Epoch 6 fell to `0.794551`. Two other signals agree that learning had stopped:
train loss moved only `-0.011` in that epoch against `-0.028` the epoch before,
and validation *loss* improved while AUC fell — the model became better
calibrated and slightly worse at ranking.

On 1,447 studies, six epochs is past the point of return. That is why the
full-data follow-up keeps six rather than the eight originally planned: the peak
arrived late in the schedule, so the schedule shape is worth reproducing, but
there is no case for extending beyond it.

## What this establishes, and what it does not

**Establishes:** the constraint was the training regime. An architecture
measured on a model with a frozen encoder, no augmentation and 3,120 optimiser
steps is measured through a floor, and eight experiments were.

**Does not establish:** anything about the hidden score. This is a
report-derived label surface. B50 gained `+0.011` here and lost `0.012` against
expert truth on Expert-58, and the archive records two earlier cases of the same
divergence. Two things differ this time — the magnitude, and that the mechanism
is the model learning to read images rather than a small architectural edit —
but neither is proof.

**Not a submission candidate.** It trained on a third of the population, and the
B42 submission loader requires `training_studies == 4349`.

## Governance

B52's validation number is a **selection statistic**: it is the maximum over
epochs on the surface used to choose the checkpoint, so it is optimistically
biased by construction. It must not be quoted as an effect size, and it is not
comparable to `0.714`. The checkpoint records this in its own `governance` field.

This is deliberately not the frozen-endpoint policy the scientific line uses.
Checkpoint selection on a held-out split is competition practice and is exactly
what the governance elsewhere in this archive forbids, for good reasons that do
not apply to a leaderboard entry. B52 sits beside that line and takes nothing
from it; every experiment recorded before it remains as it was.

## Next

```text
1. full data, six epochs      ~3,800 studies instead of 1,447
2. convert and submit         the only measurement that settles the question
3. only then, further levers  more slices, no checkpointing, ensembling
```

The archive's earlier plateau diagnosis in `POST_B45_PLATEAU_RETROSPECTIVE.md`
and the ceiling argument in the first draft of `B52_DECISION_WHERE_TO_GO.md`
both assumed the model was trained. It was not. Conclusions drawn from those
eight experiments about resolution, crop, centre count, top-k, plane routing and
gold anchoring were all measured through the same floor and would be worth
revisiting **only after** the training regime is settled — not before.
