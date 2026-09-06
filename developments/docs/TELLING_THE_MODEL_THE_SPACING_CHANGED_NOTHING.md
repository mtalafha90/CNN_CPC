# Telling the model how thick a series is changed nothing

## Status

**COMPLETE. MEASURED. A NULL RESULT, AND A CLEAN ONE.**

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

## What is worth doing next, and what is not

**Not worth doing:** re-running with a higher learning rate on the conditioning
until the delta moves. That is fitting to the endpoint, and this project's own
governance forbids it.

**Worth doing, and cheap:** read the learned weight out of the checkpoint and
compare the size of the spacing contribution against the `plane + fluid + fat`
sum it joins. The backstop proved the weight left zero, but not that it grew to
a magnitude capable of influencing anything. Those are different failures with
the same symptom, and one file answers it without any GPU time.
