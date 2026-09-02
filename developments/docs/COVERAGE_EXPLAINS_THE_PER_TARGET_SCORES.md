# The report surface ranks how often a finding is mentioned, not how well it is found

## Status

**COMPLETE. CLOSES THE SYNOVITIS QUESTION AND RETRACTS THE ADVICE THAT FOLLOWED IT.**

## The number

How often the teacher answers a target at all, against the model's AUC on that
target on the 548-study report validation surface:

```text
target              cells   coverage   report AUC
Synovitis             728      16.7%       0.9954
Lateral OA          1,277      29.4%       0.8940
Medial OA           1,323      30.4%       0.9315
PF OA               1,566      36.0%       0.8798
Contusion           1,685      38.7%       0.8588
Baker's             1,839      42.3%       0.8605
Fracture            2,267      52.1%       0.7708
Effusion            2,749      63.2%       0.7973
Medial Meniscus     2,914      67.0%       0.8087
Lateral Meniscus    3,185      73.2%       0.7644
MCL                 3,246      74.6%       0.7005
ACL                 3,423      78.7%       0.7582

coverage vs report AUC     Pearson -0.931     Spearman -0.951
```

**The less often a report mentions a finding, the higher the model scores on
it**, and the relationship is close to deterministic.

## Why

A report mentions a finding when it is worth mentioning. So the studies where
Synovitis is answered at all are the ones where synovitis was remarkable enough
to write down, in either direction — the clear cases at both ends. Separating
those is easy.

ACL is mentioned in 78.7% of reports, including every unremarkable and equivocal
knee, because a knee MRI report says something about the cruciates almost
regardless. Separating that population is hard.

The AUC is computed over supervised cells only. So each target is scored on a
differently selected subpopulation, and the selection is exactly what the
correlation measures.

## Synovitis is explained, and there is nothing to fix

```text
supervised on          728 cells, 16.7% coverage — the lowest of the twelve
validation studies     roughly 92 of the 548 carry a Synovitis cell
best sibling           Effusion, 0.9148 against a null of 0.537
best metadata column   none — no acquisition column beat a sibling anywhere
```

The scanner-shortcut hypothesis is dead: across all twelve targets, not one
acquisition column out-predicted a sibling finding. The label is not riding
manufacturer, field strength, 3D-ness or slice count.

What is left is selection. Synovitis scores 0.9954 because it is scored on the
92 studies where synovitis was worth a sentence. That is a real number about a
narrow population, not a defect, and not something to repair.

**The epoch-selection worry is retired.** Coverage is a property of the teacher
and is identical across epochs of a run, so the macro is a consistent yardstick
for choosing an epoch even though its twelve components are not comparable to
each other.

## What this retracts

Last conclusion said MCL at 0.7005 was "genuinely the weakest thing on a surface
that can measure it" and should be the next target. That was wrong. MCL has the
second-highest coverage of the twelve, and its 0.7005 is what the hardest,
least-selected population produces. It is not evidence of weakness on MCL.

Ranking targets by report AUC ranks them by how rarely radiologists mention
them. Nothing more.

## What each surface can and cannot do

```text
                       report, 548 studies        experts, 58 studies

per-target ranking     NO   confounded by         NO   intervals 0.1-0.2 wide
                            coverage, r = -0.93

per-target change      YES  coverage is fixed     NO   too small to move
across model versions       across runs

macro                  YES  consistent within     YES  as a veto, once per
                            a run                       experiment
```

The one thing the report surface does support is the comparison that matters
for development: **the same target, same teacher, two model versions.** Coverage
cancels, and a change of 0.02 is real. It is comparison *between* targets that
is meaningless.

## What it does not explain

The universal expert gap is not this.

```text
coverage vs REPORT auc     Pearson -0.931     Spearman -0.951
coverage vs EXPERT auc     Pearson -0.266     Spearman -0.308
coverage vs the gap        Pearson -0.392     Spearman -0.441
```

The expert surface labels all twelve findings on all 58 studies, so it carries no
mention-selection at all — which is why coverage barely predicts it. The mean
`+0.157` gap between the two surfaces therefore still stands as what the previous
conclusion said it was: the teacher and the experts disagree about what these
findings are, plus the report surface being an easier, self-selected population.

Both of those point the same way. The report-derived route measures something
narrower than the competition scores, and it flatters itself most exactly where
radiologists speak least.
