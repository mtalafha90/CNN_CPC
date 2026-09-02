# B52 — training the model, and what it was worth

**Status:** both runs complete. Not a submission candidate.

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

## Completed hidden result

**COMPLETED / KAGGLE `0.716`.**

```text
0.694   B37 family, 224 base
0.707   B49 native tiled multiscale
0.713   B51 adapted hierarchy
0.714   B42 / B41 constant-area          the standing reference
0.716   B52 competition full fine-tune   <- this run
```

`+0.002` over B42, and the first score this project has put above `0.714`.

**Read it carefully.** Kaggle displays three decimals, and `+0.002` on roughly
1,300 studies is the same order as the `-0.001` that separated B51 from B42 --
which was read, correctly, as no difference. One result cannot be evidence of
an effect when a result of the same size in the other direction was read as
noise. What can be said is that five architectures now span `0.694` to `0.716`,
and B52 sits at the top of that range rather than in the middle of it.

The same confound as B51 applies: this ran through the hidden-safe contract,
which drops an undecodable series instead of ending the run, and the notebook
imported pydicom after the GDCM install without invalidating caches. The
decoders were later measured available on that image, so the risk is small,
but the hidden log is not visible and the count of dropped series cannot be
recovered. `0.716` is either clean or a floor.

### What B52 changed, and what that does not license

```text
epochs                      2, fixed     ->  6, best selected
encoder trainable stages    1            ->  5
training population         4,349        ->  3,801 plus the seen-scanner splits
checkpoint selection        none         ->  best held-out epoch
```

Four changes at once, so this result cannot attribute anything to any one of
them. B52 was always a competition endpoint rather than an experiment, and it
is recorded as one. Nothing here identifies which change mattered, and nothing
here authorises tuning epochs, stages, or the selection rule against `0.716`.

### The local number, for scale

B52's selected epoch scored `0.834998` on 548 unseen-scanner studies with
report-derived labels. The hidden test returned `0.716` against expert labels.
**These measure different things on different studies and are not comparable.**
The gap is not a drop; the two surfaces have never been on the same scale.

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

## The full-data run

3,801 training studies, the same 548 unseen-scanner validation surface, seed
2026, six epochs, 26.9 hours on an RTX A4500.

```text
epoch   train      validation   macro AUC    minutes
  1     1.124810   1.075395     0.777063     269.1
  2     1.054743   1.028828     0.815093     275.3
  3     1.002861   1.049431     0.832568     260.3
  4     0.965207   1.013574     0.828500     262.7
  5     0.928085   1.010989     0.834998     263.2     <- selected
  6     0.903380   1.015765     0.833541     283.2
```

```text
B50 frozen control                     0.763117
B50 adapted hierarchy                  0.774336
B52, gate train rows, 1,447 studies    0.802666
B52, --all-data, 3,801 studies         0.834998

B52 all-data - frozen control         +0.071881
B52 all-data - adapted hierarchy      +0.060662
B52 all-data - B52 on 1,447 studies   +0.032332
```

### It flattened rather than turned over

The 1,447-study run peaked at epoch 5 and **fell** to `0.794551` at six, a drop
of `0.008`. This one peaked at epoch 5 and came back to `0.833541`, a drop of
`0.0015`. The last four epochs span `0.0065` in total.

That is a plateau, not an overfit turnover, and three signals agree on what it
is: train loss kept falling the whole way (`-0.221` end to end), validation loss
went flat after epoch 2, and validation AUC went flat with it. A model that
keeps fitting the training data while the held-out score stops moving is
memorising.

More epochs at these settings would not help. **What this describes is exactly
the condition augmentation addresses**, which is what B53 tests -- and this run
had none, whatever its own log said. See the correction at the top.

## Next

```text
1. B53                        the same regime with augmentation actually applied
2. convert and submit         no launcher pins a B52/B53 checkpoint yet
3. only then, further levers  more slices, no checkpointing, ensembling
```

The archive's earlier plateau diagnosis in `POST_B45_PLATEAU_RETROSPECTIVE.md`
and the ceiling argument in the first draft of `B52_DECISION_WHERE_TO_GO.md`
both assumed the model was trained. It was not. Conclusions drawn from those
eight experiments about resolution, crop, centre count, top-k, plane routing and
gold anchoring were all measured through the same floor and would be worth
revisiting **only after** the training regime is settled — not before.
