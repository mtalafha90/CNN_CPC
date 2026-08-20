# Raising the hidden-test macro AUC from 0.694 towards 0.80

> **Status — 2026-08-20.** Written after a full audit of the repository against
> three hidden-test results. Nothing here is a promotion or a result. It is a
> diagnosis, a ranked plan, and an explicit statement of what would falsify
> each part of it.

## Where we actually are

```text
frozen encoder,       4,349 studies    hidden 0.688
Phase-9 v2 candidate, 3,850 studies    hidden 0.691
fine-tuned 1 stage,   4,349 studies    hidden 0.694
```

Three submissions, spread **0.006**. Published work on twelve-finding knee MRI
is described in this archive as roughly 0.73-0.81 — a range that carries no
citation and should be treated as a working impression, not a benchmark.
Reaching 0.80 means closing about **+0.106**: seven times the total movement of
the eight-experiment architecture ladder, every interval of which crossed zero.

Incremental change will not do it. Four ceilings are stacked, each separately
evidenced, and none has been attacked.

## Ceiling 1 — the model is beaten by its own teacher

On the same 696 expert-labelled cells, scoring the parser's four states with
fixed constants (`positive` 0.85, `negated` 0.05, otherwise 0.50):

```text
B6 parser states, mapped to constants     0.7025
B20 trained MRI model                     0.6672
```

A lookup table with no images in it outranks the network by 0.035. Every
experiment since B7 has been competing to reproduce a 0.70 teacher, and losing.

This is a ranking comparison on shared cells, **not a ceiling**: at test time
the model has images and no report, which is a harder task. But it bounds what
the current supervision can teach.

A better teacher already exists here. `b23_llm_labels.py` runs qwen3:14b
locally, with pinned provenance, a resumable cache and an injectable backend.
On the same 696 cells:

```text
                      B6 regex    B23 LLM
state-only macro AUC   0.7025      0.8125
coverage               36.1%       63.7%
sensitivity            0.9749      0.9855
specificity            0.6061      0.5678
paired 95% CI          [+0.0681, +0.1532]      P(B23 > B6) = 1.0000
```

**+0.11, interval nowhere near zero.** Nothing else in this project has
produced a number of that size.

It was shelved by a four-part gate declared in advance. It passed three clauses
and failed one — specificity, by 0.038. Honouring a predeclared rule was
correct. But the gate treated labeller specificity as a veto while the training
policy already contains the instrument for that problem: parser positives carry
weight 0.50 against 1.00 for negatives *because* they are less trustworthy. A
less specific labeller produces less trustworthy positives. That is what the
weight is for, and `--positive-target` now exposes it.

### What the supporting experiments do and do not show

B24X isolated the mechanism cleanly: **filling B6-silent cells captured 103.3%
of the full B23 gain, while replacing B6's existing decisions added nothing**
(B23 − Density `[-0.0100, +0.0035]`). So the value is coverage, not correction
— which also means the specificity objection bites less than the gate assumed,
since B6's own calls need not be overridden.

What is still unknown: **no model has ever been trained on LLM labels and
evaluated on expert or hidden data.** B25X trained on an LLM cache but scored
on the B6-derived weak-v2 surface, where a model trained on different labels is
penalised by construction. That design cannot separate "LLM labels do not help"
from "the ruler was made of the labels being replaced".

## Ceiling 2 — the encoder throws away every location

`model.py:52` global-average-pools the encoder's 7x7x768 feature map to a single
768-vector per slice. A 224x224 slice becomes 768 numbers with **no retained
localisation whatsoever**.

Eight of the twelve targets are focal. For those, "somewhere in this slice" is
close to the least useful summary available.

The whole-study path compounds it:

```text
~5 series x ~30 slices x ~320^2 native pixels   ~15,000,000 pixels
                       reaching the study Transformer        3,840 numbers
```

And the capacity sits on the wrong side of that bottleneck: 9.46M parameters of
study Transformer run over a sequence of median length **5**, and 4.73M more
over the 12 pathology tokens **with no image input at all** — 75% of the head,
spent on sequences of length 5 and 12, while the 16-slices-to-1-vector pooling
gets 2.36M and the encoder that reads pixels is frozen.

`forward_spatial` exists at `model.py:54-71` and is unreachable from the
deployed architecture. B8 tried spatial tokens and lost, but B8 changed two
things at once: it kept spatial detail *and* pushed ~350 tokens into the
Transformer.

### Resolution — the most-repeated open question in the archive

```text
native  ->  resize 224  ->  centre-crop 90% (202^2)  ->  resize 224
```

Two irreversible resamplings in the wrong order: the crop happens *after* the
resize, so effective support is 202x202 and the final tensor carries nothing the
202^2 crop did not.

Phase 3 records `PixelSpacing` spanning **0.073-1.172 mm** and `Rows` spanning
**160-1280** — a 16-fold range squashed to a fixed 224, with physical-scale
normalisation disabled. Four separate documents name in-plane resolution as
open future work. **It has never been tested**, and `b20_crop_focus.py` now
rejects any other crop fraction by assertion.

Related: **B10 physical-scale normalisation was rejected on a point estimate
with no confidence interval recorded at all** (0.5524), as was B9 strict series
routing (0.5335). Given the 16-fold spacing range, B10 deserves re-examination
rather than inheritance.

## Ceiling 3 — training stops at full speed

```text
scheduler   CosineAnnealingLR(T_max=5)
epochs run  2
epoch 1 trains at   1.000e-04    100.0% of peak
epoch 2 trains at   9.055e-05     90.5% of peak
STOP
never used          6.58e-05, 3.52e-05, 1.045e-05, 1.00e-06
```

The cosine never anneals. Training halts after ~4,350 optimiser steps having
**never once trained at a reduced learning rate**. The config still declares
`b7_epochs: 5`; that key is dead and 2 is hard-coded.

B22 tested epochs 1-5 and found E2 best, with E1, E3, E4 and E5 all
significantly *worse* — intervals entirely below zero:

```text
        training loss     expert AUC
E2        0.6382            0.6574     best
E3        0.6088            0.6387
E4        0.5891            0.6137
E5        0.5681            0.6283
```

Loss falls monotonically while accuracy falls with it — the signature of a model
**fitting wrong labels better**. B22 said so and named the next priority as
*"label / development-selection problem"*. Eight architecture experiments
followed instead.

So training length is not wrong; it is **capped by label quality**, which ties
Ceiling 3 to Ceiling 1. Longer training should only be retried on better labels.

What B22 did *not* test is a cosine that completes inside two epochs. `T_max=2`
costs the same 90 minutes and anneals properly.

Note also that fixed-E2 was inherited unquestioned by B26 through B34, PV1, PV2
and Phase 9 v2 — including when the training population changed from 3,120 to
2,496, 1,997, 3,850 and 4,349 studies, which changes what one epoch means.

## Ceiling 4 — the weakest targets are thin structures that sampling can miss

Per-target AUC on the 499-study PV2 holdout, the largest labelled surface:

```text
Contusion        0.855      Medial OA        0.850
Fracture         0.800      Synovitis        0.789   (only 6 negatives)
PF OA            0.762      Lateral OA       0.758
Medial Meniscus  0.741      Baker's          0.713
Lateral Meniscus 0.678      ACL              0.653
                            MCL              0.605
```

**MCL is genuinely weak** — worst on every surface at every scale, and *below
chance* on the 58-study surface for essentially the whole lineage (B13 0.556,
B16 0.458, B17 0.438, B20 0.463, B29 0.469, B30 0.456). Contributing facts: only
9 positives among the 58 gold studies; B6 supervision 271 pos / 1,089 neg; and
B23 names *"periligamentous fluid with intact MCL → MCL overcall"* as a known
labelling failure.

**ACL is weak on expert truth while fine on report truth** — 0.653 on 499
studies, but 0.47-0.53 on the expert surface across six models. The model
predicts what the *report* says about the ACL while failing to track what the
images show. This is the B15 weak/gold divergence appearing at target level, and
no document reconciles it.

Now connect that to a finding nobody has connected it to. Phases 2 and 3
measured:

```text
763 series (3.13%) hold 15.24% of all 819,078 slices
mean TTA source coverage for them            52.75%
for the 320-slice series                     24.38%
```

Both phases recorded `Investigate >78-slice series structurally: GO`. It was
never done. And `tools/slice_coverage.py`, written later, states the risk
exactly: *"a structure four slices thick can fall entirely between two
samples."*

The cruciate and collateral ligaments are precisely such structures, and they
are precisely the two weakest targets. **That is a hypothesis, not a result** —
but it is a cheap one to test, because the tool already exists and has never
been run.

### A correction to an earlier reading

Synovitis looked like a large opportunity: its training labels are 434 positive
against 17 negative, and on the weak-v2 surface it scores **0.2370** with B6
labels against **0.9123** with LLM-filled labels.

That reading does not survive the larger surface. On the 499-study holdout
Synovitis scores **0.789**, not 0.24 — and that estimate rests on 6 negative
cells. More decisively, B26.2 filled 171 Synovitis cells at 100% manual label
accuracy and expert Synovitis AUC **fell** 0.8375 → 0.7826. The repo's
conclusion is that the 17 negatives reflect reporting habit — synovitis is
stated when present, rarely negated — not a missing-label defect to patch.

The B25X macro gain was 96.4% Synovitis. Its eleven-target gain was **+0.0024**.
Synovitis is therefore not the prize; it was an artefact of one surface.

## The plan, ordered by evidence per GPU hour

### Tier 0 — nearly free, do first

**0a. `T_max=2` instead of 5.** Same compute, same endpoint, proper annealing.
One 90-minute run. If it beats plain E2, the fixed endpoint was never the
constraint.

**0b. Run `tools/slice_coverage.py` on the long-series tail.** CPU only,
minutes. Tests Ceiling 4 before spending any GPU on it.

**0c. Real test-time augmentation.** Current TTA is three views of one comb
shifted by +/-1 slice, on a comb of median stride 1.9 — close to a no-op.
Horizontal flip, multi-scale and multi-crop need **no training at all**.
Contract-locked in three places, so unlocking is deliberate.

**0d. Script-stratify the stored PV2 predictions.** CPU, minutes. Tests whether
the Contusion result is a site shortcut, which the whole Phase-9 macro rests on.

### Tier 1 — the largest evidenced effect

**1. Train on LLM labels and evaluate honestly.** Re-run `b23_llm_labels.py`
over the full corpus, then train on **B6 preserved, LLM used only where B6 is
silent** — the B24X-Density formulation, which captured 103.3% of the gain
without overriding a single B6 decision, and which sidesteps the specificity
objection that closed B23.

Score on the **expert surface and the hidden test**, never on a parser-derived
surface. Compensate residual positive noise with `--positive-target` near 0.70.

**Falsified if:** the resulting model scores at or below 0.694 on hidden data.

### Tier 2 — the structural changes

**2. Stop discarding location.** Keep a small amount of spatial detail through
the slice encoder — 2x2 or 3x3 pooled tokens rather than one global average —
changing *one* thing, unlike B8.

**3. Raise in-plane resolution, and crop before resizing.** Never tested,
currently forbidden by assertion, and named as open work in four documents.

**4. Acquisition-aware sampling for long 3D series**, if Tier 0b supports it.

### Tier 3 — only after Ceiling 1 is lifted

Longer training on better labels; more encoder stages; seed ensembling.

## Arithmetic, stated as an assumption rather than a promise

The model reaches 0.6672 against a 0.7025 teacher, and 0.694 hidden against an
expert-surface offset of about +0.033. If a B23-quality teacher at 0.8125 were
followed to a similar degree, the hidden result would land near 0.80.

**That extrapolation is not sound, and `b23_llm_labels.py` says so itself:**
*"That is not a ceiling and not a ratio ('95% of teacher' is meaningless for
AUC)."* It is recorded only to show where the target could plausibly come from.

## What would make this analysis wrong

- If label noise is instance-dependent rather than class-conditional, a better
  teacher may not translate into better image-only ranking at all. B15 is the
  warning: +0.167 on teacher agreement, -0.008 on expert truth.
- If spatial pooling is not the constraint, Tier 2 returns nothing; B8's loss is
  weak evidence that it might not.
- If the long-series tail is not where ACL and MCL fail, Ceiling 4 is a story
  rather than a mechanism.

## The standing measurement problem

Every recent experiment has landed inside the 58-study surface's resolution of
roughly 0.03, and that surface is biased low by about 0.033 — a figure resting
on two points, which `CURRENT_STATUS.md` correctly says cannot establish it.

The 499-study PV2 surface has three times less noise and an offset of 0.052
measured once, but it is **parser-derived**, and large weak-surface gains have
twice failed to transfer (B15 +0.167 → -0.008; B25X +0.058 → +0.002 on eleven
targets). It is safe across very similar models and unsafe across changes of
label source or representation.

For Tier 1 in particular, the hidden test is the only trustworthy ruler.

## Two gaps worth closing regardless

- **B31 has no recorded paired interval against B20.** It is the highest number
  on the expert surface and the architecture the shipping interface targets, and
  `B31_REUSED_GOLD_RESULT.md` does not exist while B29, B30, B32 and B33 all
  have one.
- **Compressed-DICOM decoding was marked `GO before hidden submission` and there
  is no record it was verified.** Every training header is Explicit VR Little
  Endian; three submissions have since been made without the check.
