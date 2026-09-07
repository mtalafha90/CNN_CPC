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

### The teacher may have been measured against the wrong ruler

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

```text
1. finish the current B54 v2 run and submit it       in progress
2. port the worker fix                               small, multiplies everything
3. measure TTA: 1 offset vs 3, on the 548 surface    hours, unlocks ensembling
4. finish B53 with resume and workers                ~10 h, the real question
5. ensemble the best 2-3 checkpoints                 near-certain gain
6. only then B47, more slices, or new representation
```

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
