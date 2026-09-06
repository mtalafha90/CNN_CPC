# Telling the model how thick a series is changed nothing

## Status

**THE MECHANISM WAS NOT TESTED. The null is real and it measures nothing.**

The probe settled it: the learned term is **0.07% of the sum it was added to**.
The ablation compared a model against itself. See "Why the null means nothing"
below — the original reading, that this refuted the idea, was wrong.

## The measurement

One checkpoint, two scoring passes over the 58 expert studies. Same weights,
same studies, same 90% native crop, same three centre offsets. The only
difference is whether one learned vector is added to each series' metadata.

```text
spacing on            0.681223
spacing off           0.681071
the spacing effect   +0.000152
```

**+0.000152.** The surface resolves to roughly ±0.03, so this is two orders of
magnitude below what it could detect — but that framing undersells how firm the
result is. This is not two runs compared through noise. It is the same weights
on the same images, differing only in one added term, and the term moves the
macro AUC by a fifth of a thousandth.

Five of the twelve targets moved by **exactly zero**:

```text
Medial OA           +0.0016      MCL                  0.0000
Fracture            +0.0014      Medial Meniscus      0.0000
Contusion           +0.0013      Lateral Meniscus     0.0000
PF OA               -0.0013      Lateral OA           0.0000
Effusion            +0.0012      Baker's              0.0000
ACL                 -0.0012
Synovitis           -0.0012
```

Exact zeros are expected rather than suspicious: an AUC is a count of correctly
ordered study pairs, so a perturbation smaller than the gap between adjacent
studies' scores changes nothing at all. Six positive, three negative, five
unmoved is what a term with no signal looks like.

## What this does and does not overturn

**The measurement that motivated it stands.** The 2.5D triplet really does span
1.59 mm to 16.66 mm across the corpus, 69.1% of studies really do fuse series
whose depths differ at least two-fold, and the model really was never told.
Those are facts about the data and the pipeline, and this result does not touch
them.

**What is refuted is the inference.** A measured inconsistency in the input is
not evidence that the model would benefit from having it resolved. This project
has now seen that twice: teacher accuracy did not predict model performance
either, at Pearson +0.09.

**What is not established** is that depth does not matter. What was tested is
one specific intervention: a zero-initialised continuous term added to the study
hierarchy's metadata sum, at the hierarchy learning rate of 5e-6, over six
epochs of 1,447 studies. A null from that does not rule out resampling the
slices to a constant physical depth, which changes the pixels rather than
annotating them, and is a much larger change.

## The rest of the run

```text
B52 reference    0.678247
B54              0.681223
                +0.002976    within the -0.020 veto, and inside the noise
```

The teacher rebuild is neither confirmed nor refuted. +0.003 on 58 studies is
nothing, and it also carries the B6 v1.3.1 vocabulary with it. What can be said
is that a teacher with 832 more cells, 3,513 more quoted ones and two thirds
fewer evidence-free osteoarthritis calls did **not** make the model worse — which
is more than the last teacher change managed, at −0.0399.

## Why the null means nothing

`spacing_conditioning_probe` on the trained checkpoint:

```text
weight norm                    0.026101
largest single value           0.000946
spread, 0.80 to 5.00 mm        0.059062
typical plane + fluid + fat   83.272083
ratio                          0.0007
```

**0.07%.** Fourteen times below even the "small but present" band, a hundred and
forty below "a real term". The two arms of the ablation differed by a rounding
error, so of course they scored the same. Nothing about the idea was tested.

## The reasoning error, which was mine

The conditioning was put in the `study_hierarchy` parameter group, at
`hierarchy_lr_scale` 0.05 of the head rate — **5e-6**. The justification written
at the time was that it is part of the study hierarchy, which is true and
irrelevant. The hierarchy rate is low *because those weights are pretrained*
and a large step destroys them. The conditioning is freshly initialised, like
the sparse head, and the head gets **1e-4** for exactly that reason.

So a fresh parameter was given the pretrained-fine-tuning rate. Twenty times too
small, on a term that had to travel from zero.

## But the rate alone does not explain it

Adam's per-parameter step is bounded by roughly the learning rate, so over
1,447 studies at batch 2 for 6 epochs — 4,341 steps — the most any weight could
move is `5e-6 x 4341 = 0.0217`. The largest value actually learned is
`0.000946`, **4.4% of that budget**.

The rate was therefore not the binding constraint. The gradients were small and
inconsistent, which is what a term contributing 0.07% of its own sum produces:
it cannot influence the loss enough to generate a signal telling it to grow.
Zero initialisation plus a free scale is a parameterisation that has to earn its
way to relevance, and at this budget it cannot.

Raising the rate alone would not fix it. Twenty times the rate at the same 4.4%
utilisation lands at a ratio near 0.014 — the bottom edge of "present", still
not a test.

## What an actual test would need

Declared here, before any re-run, so it cannot be tuned to an endpoint:

**Scale the contribution so that a weight of norm 1 produces a spread equal to
the metadata norm.** Then the learned weight's norm *is* the ratio, directly
interpretable, and the term starts within reach of mattering instead of having
to travel four orders of magnitude to get there. Zero initialisation survives —
the model still begins numerically identical to B52.

Give it its own parameter group at the head rate, because it is a fresh
parameter and that is what fresh parameters get here.

That is a 12-hour run. The alternative is to stop, and record the mechanism as
**untested** rather than refuted — which is the honest label either way, and is
what this document now says.
