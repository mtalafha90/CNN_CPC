# B50 result — adapting the study hierarchy is supported

**Date:** 2026-08-29
**Status:** COMPLETED / SUPPORTED under the rule frozen before any result was seen.

Protocol: [`B50_REDESIGN_UNFREEZE_STUDY_HIERARCHY.md`](B50_REDESIGN_UNFREEZE_STUDY_HIERARCHY.md).

## Result

Primary endpoint: combined macro ROC AUC on the fresh scanner-grouped
`validation_unseen_scanners` surface, 548 studies, all 12 targets.

```text
surface     control     candidate     delta
combined    0.763117    0.774336    +0.011219
base        0.762566    0.774243    +0.011676
local       0.743541    0.753820    +0.010278

targets improved              12 / 12
discordant pair fraction      0.030651
max possible |delta|          0.030651   against a +0.010 threshold
headroom used                 37%
```

Every clause of the frozen rule passes: the delta clears `+0.010`, the paired
bootstrap interval sits above zero, at least seven targets improve, every
leave-one-target-out macro delta stays positive, and the seen-scanner delta is
within tolerance.

## Why this one is believable where B48 and B49 were not

**The measurement could have failed.** The two arms order 3.07% of study pairs
differently, which caps any AUC difference at `0.030651`. The observed effect
uses 37% of that headroom. B48 and B49 were judged against the same `+0.010`
threshold with ceilings of `0.0015` and `0.0024` — neither could have passed
whatever their mechanisms did. B50 had roughly 2.7x the headroom it needed.

**The effect is where the mechanism says it should be.** B50 changes the study
hierarchy, which produces the base logits. The base delta (`+0.011676`) is
*larger* than the combined delta (`+0.011219`); the local branch, admitted
through a gate of about 0.009, dilutes it slightly. A change that improved the
score for some unrelated reason would not land precisely there.

**Nothing is carried by one finding.** All twelve improve, and every
leave-one-target-out delta stays positive. That is the failure mode that misled
this project twice — B25X's macro gain was 96.4% Synovitis, and Phase-9 v2's
sign flipped when Contusion was removed.

**The gate agreed during training, before any score was computed.** The
candidate's `|tanh(g)|` settled at roughly half the control's (0.00886 against
0.01621) at both epochs. When the hierarchy can adapt, the optimiser leans on
the local correction less — which is what a genuinely improved base looks like
from the head's point of view, and it was visible before evaluation existed.

## For scale

```text
B26-B34 architecture ladder, eight experiments, total   0.006
B48 candidate effect                                    0.0000749
B49 candidate effect                                    0.0005468
B50 candidate effect                                    0.011219
B37's hidden jump, 0.694 -> 0.714                       0.020
```

## What this does not establish

**The labels here are report-derived.** This surface measures agreement with the
weak label process, not with expert truth. B15 gained `+0.167` on teacher
agreement and lost `0.008` on expert truth; B25X gained `+0.058` on the weak
surface and `+0.002` on eleven targets. Large weak-surface gains have twice
failed to transfer, and that warning applies in full here.

**The absolute numbers are not comparable to 0.714.** B50 trained on 1,447
studies — the fresh gate excludes every row B48/B49 validation spent, then holds
out a fifth of what remains. Different population, different labels, different
surface. `0.774` is not a hidden-test score and must never be quoted as one.

**One seed.** The retrospective's item 10 asks for a small predeclared seed
replication before a hidden submission is spent on any new mechanism.

## Cost, recorded for the first time in this project

```text
training, both arms       5.61 h    (~2.8 h per arm, ~85 min per epoch)
evaluation, both arms     under an hour
```

No earlier experiment in this line recorded its wall clock. Every future run on
this family can now be budgeted.

## Governance

B50's frozen rule is satisfied. That authorises nothing beyond recording the
result. Do not tune the hierarchy learning rate, epoch count, seed, geometry,
target subset or endpoint from it, and do not submit from these checkpoints —
they were trained on a third of the available studies specifically so the gate
could be clean.

## What the result implies for the line

The project's own measurements now read:

```text
224 base -> 448 encoder + tail          0.119   pair-order movement
B41 -> B42 geometry                     0.036
B37 -> B42 geometry                     0.0145
entire local sparse MIL branch          0.013
B48 / B49 candidate mechanisms          0.0015 / 0.0024
```

Nine experiments were spent inside the bottom four rows, on a branch reaching
the score at about 2% strength through a zero-start gate. B50 is the first to
train the part of the model that produces the prediction, and it is the first
powered positive since B37.

The natural next questions, in order:

1. **Does it hold on expert labels?** Score both arms on the 58 official gold
   studies as a post-training audit. They were never in any B50 gradient. This
   is minutes of compute and it is the closest available proxy for the hidden
   test's expert reference standard.
2. **Does it hold across seeds?** Two more pairs, mean and dispersion reported,
   no selection.
3. **Does it hold at full scale?** A separately declared protocol retraining the
   adapted hierarchy on the full report-only population, which is the only
   configuration a submission could come from.

Step 1 is the cheap one and it is the one that could stop the sequence.
