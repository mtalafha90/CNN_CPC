# Telling the model how thick a series is changed nothing

## Status

**TESTED AND REFUTED.** Two runs were needed. The first could not test the idea;
the second could, and the answer is that the model does not benefit from being
told the slice spacing.

```text
                 term size    Expert-58 delta    validation macro
B54 v1              0.07%          +0.000152        0.804168
B54 v2             12.65%          -0.004908        0.800665
```

v1's null measured nothing and was withdrawn — the two arms differed by a
rounding error, so of course they scored alike. v2's term is 180 times larger,
carries 12.65% of the scale of the sum it joins, and moves monotonically with
the spacing. Its ablation is a real measurement, and it is **negative**.

## The two corrections that made the test possible

Both were declared before the re-run, and neither was chosen by looking at a
score.

**Scale.** The term now enters at `reference_scale(metadata_norm)` — the size at
which a weight of Frobenius norm 1 produces a p05-to-p95 spread equal to the
`plane + fluid + fat` sum. On this model that is **90.334485**. Zero
initialisation survives, so the run still starts numerically identical to B52.

**Learning rate.** It was in `study_hierarchy` at 5e-6, the rate that is low
*because those weights are pretrained*. The conditioning is fresh, like the
sparse head, which gets 1e-4 for exactly that reason. It now has its own
parameter group at the head rate.

Together they moved the learned term from 0.07% of its own sum to 12.65%:

```text
weight norm                    11.811044
largest single value            1.191437
spread, 0.80 to 5.00 mm        10.536564
typical plane + fluid + fat    83.273918
ratio                           0.1265        "a real term"
```

The contribution rises with spacing across the whole corpus range — 7.71 at
0.8 mm, 10.85 at 3.3, 13.08 at 8.33 — which is what a term reading real geometry
looks like rather than drifting noise.

## The measurement

One checkpoint, two scoring passes over the 58 expert studies. Same weights,
same studies, same 90% native crop, same three centre offsets. The only
difference is whether one learned vector is added to each series' metadata.

```text
spacing on            0.674627
spacing off           0.679535
the spacing effect   -0.004908
```

Below the `0.03` this surface resolves, so it is not a resolvable effect in
either direction — but it is 32 times v1's delta, it is negative, and the
validation surface agrees on the sign at `-0.003503`. Two surfaces, both
negative, on a term large enough to have moved them.

Per target, five improved and seven worsened, the extremes being MCL `+0.0295`
and Synovitis `-0.0275`. On 58 studies those intervals are far too wide to read
individually and no story should be built from them. **Read the macro.**

## What this overturns, and what it does not

**The measurement that motivated it stands.** The 2.5D triplet really does span
1.59 mm to 16.66 mm across the corpus, 69.1% of studies really do fuse series
whose depths differ at least two-fold, and the model really was never told.
Those are facts about the data and the pipeline, and nothing here touches them.

**What is refuted is the inference.** A measured inconsistency in the input is
not evidence that the model would benefit from having it annotated. This project
has now seen that three times:

```text
teacher accuracy vs model performance          Pearson +0.09
a rebuilt teacher, 832 more cells              -0.003620 on Expert-58
telling the model the slice spacing            -0.004908 on Expert-58
```

Three careful measurements of a real defect, three interventions that addressed
it, three results at or below zero. That pattern is itself the finding, and it
should raise the bar for the next "the data has a flaw, so fix it" experiment.

**What is not established** is that depth does not matter. What was tested is one
specific intervention: a continuous term describing the spacing, added to the
study hierarchy's metadata sum, at a scale and rate where it demonstrably could
influence the model. A null from that does not rule out **resampling the slices
to a constant physical depth**, which changes the pixels rather than annotating
them, and is a much larger change.

## Why it is being dropped rather than tuned

The stopping rule was written down before the number existed: under a ratio of
0.01 the mechanism would be refractory to this parameterisation and should be
dropped; at or above 0.10 it is a real term and the null is about the idea. The
ratio came back at **0.1265**.

So this is the second branch, and it is the one that ends the question. The term
was given a scale where it could matter, a learning rate that fresh parameters
get here, and 4,341 steps to use them. It reached 12.65% of its own sum and made
the model slightly worse on both surfaces. A third attempt at a different scale
would be fitting to an endpoint, and the archive would be right to refuse it.

## The rest of the run

```text
B52 reference          0.678247
B54 v2, spacing off    0.679535     +0.001288
B54 v2, spacing on     0.674627     -0.003620
```

Neither is a resolvable difference on 58 studies. The teacher rebuild — 832 more
cells, 3,513 more quoted, evidence-free osteoarthritis calls cut from 2,632 to
896 — is neither confirmed nor refuted by this. What can be said is that it did
not make the model worse, which is more than the last teacher change managed at
`-0.0399`.

## A note on which arm gets submitted

The B54 submission runs with the conditioning **disabled**, because B42's
inference loop supplies no spacing and the gate in
`b54_competition_submission_dualgpu_fast` now confirms the two arms agree within
this surface's resolution.

The disabled arm also happens to score higher here, and **that is not the
reason**. Choosing an arm because it won on the 58 expert studies would be
selecting on a surface this archive has repeatedly and correctly called
adaptively spent. The reason is the inference path and the measured agreement;
the ordering is a coincidence that removes any tension, not evidence.

## A footnote on the first run

B54 v1 was deleted by an `rm -rf runs/085_B54/train` in the re-run instructions,
which was my error. Nothing scientific was lost — v1 and v2 scored the same on
the validation surface, and v1's ablation had already been shown to measure
nothing — but the checkpoint is not recoverable. `B54_ONE_RUN_RUNBOOK.md` now
renames rather than deletes.
