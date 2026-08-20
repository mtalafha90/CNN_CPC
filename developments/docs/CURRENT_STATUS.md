# Current project status

**Snapshot:** 2026-08-19
**Package:** `0.30.0`
**Primary metric:** macro ROC AUC across 12 targets
**Best independent evidence:** hidden-test macro AUC **0.694** (encoder
fine-tuned one stage, all-script supervision), submitted 2026-08-19

## The rulers are now calibrated

The first submission ends the project's central measurement problem for the
expert surface. The frozen-encoder model, trained on all 4,349 report-only
studies:

```text
58 expert studies (local)     0.652     understates by about 0.036
hidden competition test       0.688     the reference
```

The expert surface is biased low rather than merely noisy. It should not be
quoted as an estimate of competition performance; it remains useful for
ordering models, not for predicting the score.

**The 499-study surface is not yet calibrated, and an earlier version of this
page said otherwise.** It listed 0.743 as a third row of the table above, as
though one model had been measured on three surfaces. It had not. The 0.743 is
`candidate_macro_auc` from the Phase-9 v2 comparison, and that model was trained
on **3,850** studies with the 499 held out -- necessarily, since scoring a
full-population model on those studies would be scoring it on its own training
data. Subtracting a 3,850-study model's local score from a 4,349-study model's
hidden score does not measure a surface's bias; it measures two different
models. The "overstates by about 0.055" that followed from it was not supported.

Calibrating it properly needed the hidden score of the *same* model that scored
0.743 locally. The third submission supplied it:

```text
Phase-9 v2 candidate (all-script, 3,850 studies, 499 held out)
    499 weak studies (local)    0.7434
    hidden competition test     0.691     overstates by 0.052
```

So the surface does overstate by roughly the amount originally guessed, and now
the figure rests on one model measured twice rather than two models subtracted.
Both local surfaces are biased, in opposite directions, and the truth sits
between them:

```text
58 expert studies      understates by about 0.033   (two pairings)
499 weak studies       overstates  by about 0.052   (one pairing)
```

The 499-study surface is the better instrument despite the larger offset,
because its offset is what gets subtracted while its *noise* is what limits
resolution, and 499 studies carry roughly a third of the noise of 58.

## Three submissions, and the band they fall in

```text
                          train    encoder      hidden
frozen encoder            4,349    frozen       0.688
Phase-9 v2 candidate      3,850    frozen       0.691
fine-tuned, 1 stage       4,349    1 stage      0.694
```

**Every hidden score sits within 0.006 of every other.** The three differ in
whether the encoder learned and in 499 studies of training data, and the whole
spread is smaller than the uncertainty on any one of them.

The middle row is the most informative. It trained on 500 fewer studies than
the row above it and scored no worse. Whatever is limiting this model, it is
not the last 12% of the training population -- which also means the remaining
unlabelled studies are unlikely to be worth much.

```text
                        58 expert    hidden    offset
frozen encoder            0.652       0.688    +0.036
fine-tuned, 1 stage       0.663       0.694    +0.031
difference               +0.011      +0.006
```

The offset held to within 0.005 across the two, the first sign that the expert
surface is biased rather than merely noisy. If that survives a third point,
differences measured on it can be read as roughly tracking hidden differences,
and `hidden ~ expert + 0.033` becomes a usable rough predictor. Two points
cannot establish it.

**Encoder fine-tuning is not established by this.** Both gaps are smaller than
either surface can resolve: 58 studies carry an error near +/-0.16 per target,
and a single hidden score of this size has an uncertainty of roughly +/-0.01.
What can be said is that two surfaces moved the same way by a similar amount,
which is weak positive evidence and a reason to keep probing the line, not a
result. The honest summary is that the recipe has not been shown to hurt and
may help a little.

The practical consequence is a ceiling on this style of experiment. A change
worth roughly +0.005 cannot be told from noise one submission at a time, so the
remaining work should favour changes carrying a mechanism and a measurement
over further small architectural variations.

For context, published work on this task shape - twelve knee findings from MRI
- reports roughly 0.73 to 0.81. A first submission at 0.688 is within reach of
that band, and the long-standing 0.94 target has no published precedent for
this problem.

## Where things stand

**Nothing has been promoted.** Every result below comes from surfaces built in
this repository. The frozen governance position remains that B20 is the last
model promoted on evidence, and no later experiment has cleared a promotion
path.

**The top-level interface targets the B34/B31 architecture.** That is an
interface decision recorded in `docs/WORKING_MODEL.md`, made because B34 is the
strongest candidate on both internal surfaces and has the simplest inference
path. It is not a promotion, and it does not change any frozen experiment
record. The interface and the governance record therefore disagree on purpose,
and the disagreement resolves when a hidden-evaluation result exists.

## The two findings that matter

**1. The reports are multilingual, and the parser could not read a quarter of
them.** Phase 5 inspected the zero-cell population and found every sampled
report contained clear target-relevant statements. Their zero-label status was
parser coverage, not clinical silence. The decisive detail: all 12 sampled
Greek B6-active reports had exactly one usable cell, `Contusion = positive`,
each from the incidental English phrase `bone bruise`. Apparent non-Latin
coverage was embedded English, not parsing.

Translating before running the unchanged parser (Phases 6-8) produced:

```text
rescued studies      1053 / 1229 = 85.68%
coverage             71.74% -> 95.95%
usable cells         14123 -> 18024   (+3901, +27.62%)
by script            Cyrillic 99.54%   Latin 83.22%   Greek 81.43%
```

**2. A powered validation surface now exists.** The prospective weak splits
(PV1/PV2) give 499-624 validation studies, against 58 for the expert surface.
PV2's primary test returned a 95% interval entirely below zero
(`[-0.01257, -0.00399]`, P = 0.9998) — something the expert surface has never
produced in 34 experiments.

## Architecture ladder: essentially flat

Reused 58-study expert surface, paired against B20 `0.6674066371`:

| Experiment | Macro AUC | Outcome |
|---|---:|---|
| B26.2 supervision repair | 0.6663 | closed, not promoted |
| B27.1 pathology routing | 0.6599 | closed, not promoted |
| B28 max-evidence residual | 0.6383 | closed, not promoted |
| B29 complementary pool | 0.6769 | frozen candidate, not promoted |
| B30 projected complementary | 0.6547 | formulation closed |
| B31 local context | **0.6823** | highest on this surface |
| B32 dispersion summary | ~tied | formulation closed |
| B33 uniform mean | 0.6764 | simplification of B29 |
| B34 train-only scaffold | — | B31-equivalent, simpler inference |

Eight experiments, roughly +0.015 of point estimate, every interval crossing
zero. The architecture direction is exhausted in its current form.

## Phase 9 v2 — the matched supervision test

Both arms trained on 3,850 studies with the 499-study PV2 partition held out,
and scored on original frozen labels only, so the evaluation is not circular.

```text
BCE        -0.00988   CI [-0.01990, +0.00008]   P = 0.9742
macro AUC  +0.00322   CI [-0.00847, +0.01508]   P = 0.6897
```

Both aggregates favour the merged supervision; both intervals include zero. The
BCE upper bound sits at `+0.000084`, about as close to excluding zero as is
possible without doing so.

Two per-target results reached significance, but **only one survives correction
for testing 12 targets**:

```text
Contusion  +0.0554  CI [+0.0206, +0.0933]  P=0.9990  two-sided p ~ 0.0020
           survives Bonferroni (0.05/12 = 0.00417) and Benjamini-Hochberg

Effusion   -0.0262  CI [-0.0483, -0.0052]  P=0.0082  two-sided p ~ 0.0164
           survives neither
```

Removing Contusion flips the macro sign (`+0.0032 -> -0.0015`). The entire
aggregate rests on one target.

**This is the third time an aggregate has dissolved into a single target under
audit** — B25X was 96.4% Synovitis, B24X-Density showed the gain was coverage
rather than correction, and Phase 9 is Contusion. Leave-one-target-out belongs
in the standing protocol, not in a post-hoc check.

## Open hypothesis on the Contusion result

Contusion is the only target where the non-Latin population had any
pre-existing label coverage — the incidental `bone bruise` positives from
Phase 5. It is also the only target driving the Phase 9 macro result. That
coincidence is untested.

If the control arm learned a site shortcut from a single-target, positive-only
signal in a distinct scanner population, the +0.055 would be partly the
*removal of a control artefact* rather than new signal in the candidate.

The discriminating check is cheap and non-circular: stratify the stored PV2
per-target results by report script (Latin / Greek / Cyrillic). No retraining,
no change to the rescue set. Concentration in the non-Latin strata supports the
shortcut reading; an even spread across Latin studies does not.

## Blocked

The Phase-7 rescue evidence — `translation_cache.jsonl`,
`full_population_rescue_audit.csv`, `recovered_cells.csv` — was not found at
the attempted local path, so the label-generation mechanism audit cannot run.
`phase9_v2_rescue_mechanism_audit.py` exists and is ready; unlike the other
modules in this campaign it has no test file.

The same path ambiguity affects training: the merged-label export is recorded
under two different roots across the archive. Verify with the pinned
fingerprints before launching anything:

```text
training_targets.csv   c59d78c74743112f09946fd18b64d7726947e6f75b83aabd1f585389a89d045a
recovered_cells.csv    ed094e5d6f77b1558fe63921f2f22b8e1006443c506f00f921d842cde72025d0
```

## Next

1. **Submit.** Zero submissions after 34 experiments and 9 audit phases. A
   Kaggle-run notebook costs none of the 9-hour session budget, and the
   leaderboard is the only measurement not built here. Submitting a *matched
   pair* — `latin-script` versus `all-script` supervision — resolves what
   Phase 9 could not.
2. **Script stratification** of the stored PV2 predictions (CPU, minutes).
3. **Re-score PV1/PV2 on macro AUC with paired bootstrap.** Those surfaces
   currently rank on soft BCE only; macro AUC is recorded without an interval
   and is never used to rank. PV1 itself notes B31 and B33 had near-identical
   macro AUC while the primary metric separated them at P = 0.0050 — so the
   ladder that selected B31 and B34 has never been checked on the metric the
   competition scores.
4. **Locate the Phase-7 artifacts** and run the mechanism audit.

Detailed records for every experiment named here are in this directory and are
frozen; they are not revised when the project's understanding moves on.
