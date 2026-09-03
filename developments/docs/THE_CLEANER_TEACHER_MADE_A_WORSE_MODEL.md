# The cleaner teacher made a worse model

## Status

**COMPLETE / VETOED. DO NOT SUBMIT.**

The negated-only teacher was the best teacher this project has measured: 18.2%
wrong against the 58 experts, against 24.2% for the merge it replaced. Trained
on it, the model is `-0.0399` worse on the only surface that predicts the hidden
score.

## The two surfaces disagree about the sign

```text
report labels, 548 studies    0.862671  vs  0.834998    +0.0277
expert labels, 58 studies     0.638317  vs  0.678247    -0.0399
```

The trainer printed the first of those itself, as "against B52 on the same
split". The split is the same; the surface is not. This teacher supervises
25,524 cells where B52's supervises 34,010, coverage falls from 65.2% to 48.9%,
and coverage predicts report AUC at Pearson `-0.931`. A smaller, cleaner surface
raises an AUC without the model improving.

The expert surface carries no such selection: all twelve findings are labelled
on all 58 studies, whatever the report happened to mention.

## The veto

`expert_audit` declares its threshold before any run:

```text
VETO_DELTA = -0.020
observed     -0.0399        twice the threshold
```

Per target, with the usual caution that individual rows carry intervals far too
wide to read alone:

```text
Medial OA            0.7907   0.6744   -0.1163
Effusion             0.8981   0.7913   -0.1068
Synovitis            0.7778   0.6977   -0.0801
Baker's              0.8152   0.7355   -0.0797
Contusion            0.5735   0.4966   -0.0769
ACL                  0.5478   0.4792   -0.0686
Medial Meniscus      0.7188   0.6575   -0.0613
Fracture             0.6139   0.5542   -0.0597
MCL                  0.5011   0.4490   -0.0521
Lateral Meniscus     0.6783   0.7106   +0.0323
PF OA                0.6512   0.6847   +0.0335
Lateral OA           0.5725   0.7292   +0.1567
```

Worse on 9 of 12. No single row is evidence; nine of twelve moving one way is.

## What each teacher actually holds

Counted directly from each export's `structured_labels.csv`, gold rows excluded,
by the same rule training uses -- `positive` or `negated` at confidence 0.75 or
above:

```text
teacher                 studies    cells      pos      neg   pos %    err
B6 v1.2.1 alone           4,349   14,123    6,871    7,252   48.7%  21.9%
B52: fill both            4,349   34,010   15,357   18,653   45.2%  24.2%
091: fill negated         4,349   25,524    6,871   18,653   26.9%  18.2%
092: 091 + rescue         4,349   26,202    7,530   18,672   28.7%  18.2%
```

Three identities confirm the merge behaved as designed, and were checked rather
than assumed: B6 and 091 carry the same 6,871 positives, because the negated-only
rule never touches a parser cell; B52 and 091 carry the same 18,653 negatives,
because both take the same negated fills; and B52's 34,010 matches
`EXPECTED_BASE_CELLS`, so this is the surface B52 actually trained on rather than
a later rebuild.

## Why, most likely

The negated-only rule was adopted for a good measured reason. Filled positive
cells were 62.8% correct where filled negated cells were 97.8%, so dropping the
positives raised label accuracy from 24.2% wrong to 18.2%.

The LLM had filled 19,887 cells: 11,401 negated and 8,486 positive. The rule
kept every negated one and discarded every positive one.

```text
positives   15,357  ->  6,871      a fall of 55%
```

**The model lost more than half of every example of what a finding looks like,
and gained accuracy about what one does not.** 092 restores 659 of those 8,486,
which is 7.8% of what was removed.

That is a hypothesis rather than a demonstration; the run changed one thing, but
that one thing changed both label accuracy and positive count together, and this
result cannot say which mattered. What it does establish is that the two do not
point the same way.

## What this overturns

Two conclusions from this week were built on the assumption that a cleaner
teacher is a better teacher. That assumption is now contradicted by direct
experiment:

```text
point 2   the 18.2% error rate    chased teacher accuracy
point 3   the confidence column   chased teacher accuracy
```

Both were closed as dead ends for their own reasons, which stands. But the axis
they were on turns out to be the wrong axis entirely, and that is worth more
than either conclusion. Two independent measurements already pointed here and
were not followed:

```text
teacher label error vs model AUC, per target     r = +0.09     no relationship
teacher coverage vs model AUC, per target        r = -0.93     near-deterministic
```

Accuracy does not predict performance. Coverage does.

## What it changes for the rescue merge

`runs/092_rescued_negated_fill` adds 678 cells, 659 of them positive, to exactly
this teacher. It was built expecting a small effect and holding a mild worry
about pushing a teacher further towards positives when its measured fault was
false positives.

Under this result that worry inverts. The change adds positive coverage, which
is the axis that correlates with model performance, to a teacher that has just
been shown to be short of it. It is now the more promising of the two pending
runs rather than the marginal one.

## The pattern this is the fourth instance of

```text
B40   improved every training loss          Expert-58 fell
B50   +0.011221 on 548 report studies       -0.002432 on Expert-58
B51   production version of B50             -0.011785 on Expert-58
this  +0.027673 on 548 report studies       -0.039930 on Expert-58
```

Four times a report-derived gain has failed to reach expert truth, and this is
the largest. The report surface is not a weaker version of the hidden test. It
measures a different thing, and it can move in the opposite direction.
