# Where the score is actually lost — a whole-repository review

**Written:** 2026-09-07, while B54 v2 trains. Nothing here is a new experiment.
It is a reading of what the archive already contains, looking for the largest
unspent levers.

## The scoreboard

```text
0.694   B37 family, 224 base
0.707   B49 native tiled multiscale
0.713   B51 adapted hierarchy
0.714   B42 / B41 / B37 constant-area        the standing reference
0.716   B52 competition full fine-tune       the best recorded
0.952   public leaderboard top
```

Eleven completed experiments span `0.694` to `0.716` — a range of `0.022`. The
gap to the top is `0.236`, roughly **eleven times the entire spread of
everything this project has ever tried.**

That ratio is the most important number in the archive. It means the search has
been happening inside a region where nothing that has been varied matters much.

## The finding that should redirect the work

`POST_B45_PLATEAU_RETROSPECTIVE` diagnosed the constraint as an untrained model,
and it was right. B52 fixed the training regime and the local surface moved a
great deal:

```text
B50 frozen control                    0.763117
B52, full data, best epoch            0.834998
                                     +0.071881   on 548 unseen-scanner studies
```

**That `+0.072` arrived at the leaderboard as `+0.002`.**

This is not an isolated case. It is the third time the archive has recorded the
same divergence:

```text
mechanism                     local movement      hidden movement
B50 adapted hierarchy            +0.011219           -0.001  (B51 0.713 vs 0.714)
B52 trained regime               +0.071881           +0.002  (0.716 vs 0.714)
teacher accuracy vs model         --                 Pearson +0.09
```

Three independent attempts, three failures to transfer. **The local
report-derived validation surface does not predict the hidden score.** Neither
does the 58-study expert surface, which has moved in the opposite direction
twice.

This project currently has **no ruler that tracks the thing it is trying to
maximise.** Every decision since B37 has been made on instruments that have now
been shown, repeatedly, not to point at the target. That is a more serious
problem than any missing architecture, because it means further careful
experiments will keep producing confident local numbers that do not arrive.

### The exchange rate, measured

**Added 2026-09-08.** The 1,447-study control was submitted and scored
**0.708**. That gives the first clean local-to-hidden calibration this project
has ever had:

```text
                     training studies    local (548)    Kaggle
086  B52             1,447               0.802666       0.708
087  B52             3,801               0.834998       0.716
                                        +0.032332      +0.008
```

Same architecture, same teacher, same split, same 548 validation studies. The
**only** difference is 2.6 times the training data, and both ends were measured
on both surfaces.

**Local movement reaches the leaderboard at roughly one quarter.** `+0.032`
became `+0.008`. That single ratio does more work than any result in this
archive, because every decision here is made on the local surface and nobody
knew what its numbers were worth.

Two consequences, both uncomfortable:

**More data is not the lever.** 2.6x the studies bought `+0.008`. The population
caveat attached to a dozen results in this archive is worth less than a
hundredth of AUC, and the remaining data is already in use.

**The gap cannot be closed on this surface.** Reaching `0.952` from `0.716`
needs `+0.236` hidden. At the measured exchange rate that is roughly `+0.94`
local — from a surface already sitting at `0.835`, whose ceiling is `1.0`. The
arithmetic does not work. **Whatever the leaders are doing does not show up on
this validation surface at all**, which means it is either a different label
process, a different data regime, or a capability this pipeline does not have.

One caveat, stated so it is not forgotten: this is one comparison between two
similar models, and `0.008` on ~1,300 studies is only four times the `0.002`
this archive already reads as noise. The ratio is a working estimate, not a
constant. But its direction has now been confirmed four times, and never once
contradicted.

### Two readings, and they lead to different work

**Reading A — the surfaces are wrong and the hidden test is reachable.** The
local splits share scanners, sites and a label process with the training data in
ways the hidden set does not, so local gains are partly fitted to those
regularities. The remedy is better validation design, not more mechanisms.

**Reading B — the gains are real but capped by something not yet varied.** A
model can genuinely improve at ranking these studies and still sit at `0.716` if
the binding constraint is elsewhere: resolution, volume coverage, ensembling,
or the label process itself.

The archive cannot currently distinguish these, and that is worth saying plainly
rather than choosing the more comfortable one.

## The three largest unspent levers

Ranked by expected gain per GPU-hour, with what is already known about each.

### 1. Finish B53. Augmentation has never completed a run here.

**This is the clearest unfinished business in the repository.**

B52's full-data run left textbook evidence of memorising:

```text
train loss           -0.221 end to end, falling the whole way
validation loss      flat after epoch 2
validation AUC       flat after epoch 3, last four epochs span 0.0065
```

A model that keeps fitting the training data while the held-out score stops
moving is memorising, and augmentation is the standard remedy. `B53` applies it.
It ran three epochs of six and was stopped:

```text
epoch    B52 no aug   B53 aug     delta    B53 gain
1          0.777063   0.770903   -0.0062          -
2          0.815093   0.794150   -0.0209    +0.0232
3          0.832568   0.809708   -0.0229    +0.0156
```

Augmentation is *expected* to lose early — that is the mechanism. B52's gains
were finished by epoch 3; B53 was still climbing at `+0.0156` when it stopped.
To reach B52's final `0.833541` it needs `+0.024` over three epochs, and its own
trajectory (`+0.0232`, `+0.0156`, decelerating) makes that a genuine coin-flip.
**The three epochs that would have settled it are the three that were not run.**

Two things have changed since, both in its favour:

* `training_resume.py` now exists, built for B54. B53's doc says "there is no
  resume"; that is no longer true, and wiring it in is small.
* The measured pace with workers was 3.0–3.3 h/epoch against B52's 4.5.

**Cost:** roughly 10 hours if resumed with workers, 19 from scratch.
**Why it is first:** the diagnosis is specific, the remedy is standard, and it is
the only lever here that a published result would call obligatory.

### 2. Ensemble. There is no ensembling code in the repository at all.

`ls developments/src/rsna_knee/ | grep -iE "ensembl|blend|rank_avg"` returns
nothing. Eleven trained checkpoints exist and not one submission has ever
combined two of them.

The metric makes this unusually cheap: macro AUC depends only on ranking, so
rank-averaging needs no calibration, no weight fitting and no new ideas. It is
the most reliable single-digit gain in competitive machine learning, and it is
the one standard practice this project has never used.

**The obstacle is the inference budget, and it has a clean answer.**

```text
Kaggle ceiling                     9.00 h
guard (B37_SUBMISSION_MAX_HOURS)   8.25 h
reserve                            0.50 h
one model, ~1,300 studies, 2x T4   ~4.5 h
```

Two models will not fit. But `B42_FAST_TTA_OFFSETS = (-1, 0, 1)` means every
study is scored **three times** with different centre offsets, so:

```text
1 model  x 3 TTA views   ~4.5 h     what runs today
3 models x 1 TTA view    ~4.5 h     same cost, three independent models
```

Model diversity almost always buys more than view diversity. And the value of
those three views **has never been measured** — B39 tried to widen TTA to five
offsets and failed operationally before producing a score, so the marginal
worth of offset averaging is unknown in this repository.

That measurement is cheap and local: score the 548-study surface at one offset
and at three, on a checkpoint that already exists. A few GPU-hours to learn
whether two thirds of every submission's runtime is buying anything.

**Cost:** the local TTA measurement is hours; the ensembling code is small.
**Why it is second:** near-certain gain, no new science, but it needs the TTA
question answered first to make room.

### 3. Port the worker fix into the `developments/` trainer.

B53's standalone run measured **3.0–3.3 hours per epoch** against B52's 4.5 for
the same work — roughly a third off — using `workers=6` with
`sharing=file_system`.

That fix lives in `notebook/b52_standalone.py`. The `developments/` trainers read
`num_workers: 0` from `config/b42_constant_area_aspect_sparse.yaml` and expose no
override, so **the run training right now is paying the old cost.**

This is not a lever on the score. It is a multiplier on every lever below it: a
third more experiments per week, permanently, for a small and well-understood
code change.

**Cost:** small. **Why it is third:** it makes everything else cheaper, so it
should be done before the long runs, not after.

## Two things worth reconsidering, not yet acting on

### RETRACTED: the hidden labels are not report-derived

**Added 2026-09-08.** The section below argued the hidden labels might be
report-derived, which would make Expert-58 the misleading ruler. **That was
wrong, and backwards.** Two independent sources now settle it:

```text
Test ground truth is image-derived by two subspecialty MSK radiologists,
independently, with a third adjudicating disagreements. The same process
was used for the test set. Borderline findings are graded NEGATIVE.
```

Confirmed by host replies in the pinned competition forum threads, reported
separately by two competing teams, and corroborated by this repository's own
earlier compilation in `COMPETITION_SPEC_AND_FIELD_INTEL.md`.

**This explains the exchange rate measured above.** The local surface scores
against report-derived labels; the leaderboard scores against expert
image-derived labels under a severity threshold. They are two different
measuring instruments, so `+0.032` local reaching `+0.008` hidden is not noise
and not shrinkage — it is the fraction of a report-label gain that happens to
coincide with the real target. The same reading applies to B50's `+0.011`
local against `-0.001` hidden, and to the teacher's Pearson `+0.09`.

**And it inverts the teacher conclusion.** The rubric is specificity-biased:

```text
ACL / MCL      high-grade or full-thickness only; low-grade sprain -> NEGATIVE
Meniscus       definite surface contact on >=2 images; degeneration -> NEGATIVE
OA             ~>=1 cm of >50%-thickness cartilage loss
Effusion       moderate or large only; trace and mild -> NEGATIVE
Baker's        moderate or large only
```

A teacher tuned for clinical correctness **over-fires** against this. "Mild
effusion", chondropathy and intrasubstance meniscal degeneration are all real
and all competition-negative. That is structured, target-correlated noise,
which is worse than symmetric noise — and it means the teacher work was not
wasted so much as aimed at the wrong target. A careful human reading only the
report agrees with gold at roughly 82.5%, which is the ceiling report text can
reach whatever the vocabulary does.

The section that follows is kept for the record and should be read as
superseded.

### SUPERSEDED: the teacher may have been measured against the wrong ruler

`THE_TEACHER_IS_NEAR_ITS_CEILING` found that of nine teacher "errors" read
against the report text, **eight were the parser reading correctly and the
expert disagreeing.** Every Contusion row was of that kind.

The archive also records that hidden scores run consistently *above* the
Expert-58 surface — about `+0.033` — and that `0.952` is far above published
clinical work on these same twelve findings (CoPAS 2024: `0.812` in-domain,
`0.721–0.726` external).

Both observations point the same way: **the hidden labels may be report-derived
rather than expert-adjudicated.** If so, then fidelity to reports is the right
target, the 58 expert studies are the misleading ruler, and a large amount of
the teacher work has been graded against the wrong standard.

This is a hypothesis, not a finding, and it is testable in exactly one way that
this project already has: the teacher's own agreement surface predicts the
hidden score, or it does not. Worth holding in mind before the next teacher
decision — not worth a run on its own.

### The report text is still barely used

B16's "full report semantic alignment" was TF-IDF into truncated SVD. The corpus
is multilingual — the teacher doc quotes Turkish and Spanish — and no modern
multilingual text encoder has ever been applied. If Reading B above is right and
the labels are report-derived, this is a large unexplored surface. It is also a
new experiment rather than a tweak, so it does not belong in the next few runs.

## What to stop

* **The spacing conditioning. Closed.** Tested properly on the second run — term
  at 12.65% of its own sum, Expert-58 `-0.004908`, validation `-0.003503`. Both
  surfaces negative. See `TELLING_THE_MODEL_THE_SPACING_CHANGED_NOTHING`.

  It closes with a pattern worth carrying forward. Three times now this project
  has measured a real defect in its inputs and fixed it, and three times the
  model has been indifferent or slightly worse:

  ```text
  teacher accuracy vs model performance     Pearson +0.09
  a rebuilt teacher, 832 more cells         -0.003620 on Expert-58
  telling the model the slice spacing       -0.004908 on Expert-58
  ```

  **A measured flaw in the data is not evidence that fixing it will help.** That
  should raise the bar for the next experiment of this shape, and it is a further
  argument for spending the next runs on ordinary competition engineering — the
  augmentation, the ensemble, the throughput — rather than on another defect.
* **Small architecture edits judged on the 58 expert studies.** The retrospective
  established this and it still holds; the surface is adaptively spent and points
  the wrong way.
* **Reading `+0.002` as an effect.** The archive has correctly called `-0.001`
  noise. `0.716` is the top of a `0.022` band, not a result.

## The recommended order

Revised 2026-09-08, after the exchange rate was measured.

```text
1. port the worker fix                          DONE
2. finish B53 with resume and workers           running
3. recalibrate the teacher to the RUBRIC        free, and the target was wrong
4. physical-mm crop + laterality normalisation  measured at +0.030 LB elsewhere
5. per-finding attention pooling, 12 heads      the field's most repeated win
6. ensemble by input representation, not backbone
```

**Where this stands against the field.** Public inference-only notebooks score
`0.899`; the reference baseline with its shipped weights scores `0.891`; the
top is `0.952`. At `0.716` the gap is structural, not a matter of tuning.

Three corrections to earlier reasoning in this document, all from
`COMPETITION_SPEC_AND_FIELD_INTEL.md` and the 2026-09-08 field research:

* **Resolution is not the lever, and may be negative.** A controlled arm
  measured 384 px against 224 px at `-0.004` for 3.5x the cost. This project
  runs at 448.
* **DINOv3 is not the answer here.** DINOv3-ViT-B/16 sits at `0.771` on the
  competition's own model board while DINOv2-**small** reaches `0.914`, and one
  competitor measured v3 below v2 on an identical pipeline. The unrecorded
  DINOv3 test in this repo is not worth repeating on that evidence.
* **There is no shortcut to find.** Metadata-only models were publicly probed
  at `0.65` random-fold, `0.60` scanner-grouped, and a metadata-only submission
  scored `0.531`. The leaderboard reflects genuine image reading.

Two operational facts that change planning: **five submissions per day**, not a
scarce resource as assumed above; and the default Kaggle GPU is a P100 on which
the preinstalled torch has no `sm_60` kernels, so T4 must be selected
explicitly.

**What the exchange rate does to these estimates.** A B53 that beats B52 by
`+0.02` locally — a good result — predicts about `+0.005` hidden, landing near
`0.721`. Worth having and worth knowing, but it does not change the picture.

**Ensembling is now the most interesting item on the list**, because it is the
one gain that is *not* measured on the local surface and therefore not subject
to the quarter-rate discount. Rank-averaging independent models buys decorrelation
directly on the test set. It is also the only standard practice this project has
never used.

> **Two corrections, added 2026-09-08.**
>
> **The discount claim was an argument, not a measurement.** Variance-reduction
> gains do tend to transfer better than fit gains, but nothing here measured
> that, and it should not have been stated as though something had.
>
> **The checkpoints were not diverse enough to ensemble.** 086, 087 and B54
> share a base checkpoint, an architecture, a seed and a split; rank-averaging
> them lands between members rather than above. B53 and B55 are the first
> genuinely different models this project has, and `ensemble_eval` now measures
> member agreement beside any gain so this cannot be assumed again. See
> `TTA_AND_ENSEMBLING.md`.

**B54 v2 should not be submitted.** At `0.800665` local it sits `0.002` below
086, and the exchange rate predicts `~0.708` — the same number 086 just
returned, for a whole slot. The teacher rebuild is worth approximately nothing
on this surface, which three measurements now agree on.

Steps 2 to 5 are all standard practice that this project has skipped while
running eleven careful experiments on mechanisms an order of magnitude smaller
than the gap. That is the central finding of this review: **the archive's
science is strong and its competition engineering has holes in it**, and the
holes are where the remaining score is.

## What would make this review wrong

* **B53 finishes level with or below B52.** Then memorisation was not the binding
  constraint at six epochs, and the training-regime line is closer to exhausted
  than it looks.
* **Ensembling three checkpoints moves the hidden score by under `0.005`.** Then
  the models are far more correlated than their architectures suggest, and the
  limitation is representational after all — which promotes B47 and the
  volumetric work.
* **A single-offset submission scores the same as three offsets.** Then TTA was
  never buying anything, the runtime budget was always three times larger than it
  appeared, and several past "we cannot afford it" decisions should be revisited.
