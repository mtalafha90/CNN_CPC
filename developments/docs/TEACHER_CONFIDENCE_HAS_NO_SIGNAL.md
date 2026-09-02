# The confidence column cannot be rescued

## Status

**COMPLETE. NEGATIVE RESULT. NO CHANGE MADE.**

Three candidate replacements for the constant confidence column were measured
against the 58 expert studies. Two are dead. The third is real, already acted on
as far as the merge allows, and turning it into a number would cost the only
expert truth the project has.

## What was wrong to begin with

Every committed cell carries exactly `0.90`, every silent one `0.0`. Training
reads the column once:

```python
positive = (state == "positive") & (conf >= MIN_CONFIDENCE)
w[positive, j] = POSITIVE_WEIGHT      # flat 0.50, never scaled by conf
```

So `conf >= 0.75` asks "is this cell answered" and nothing else. The column
looks like a quality score and is not one.

## The measurement

`runs/092_rescued_negated_fill` against the 58 experts, `b23_full` as a second
labeller. 313 cells scored; 57 disagree with the expert.

```text
                            cells   wrong     err    accuracy
all cells answered            313      57   18.2%   0.818 +/- 0.022

who answered the cell
  base                        251      55   21.9%   0.781 +/- 0.026
  filled                       62       2    3.2%   0.968 +/- 0.022

what was said
  negated                     145       5    3.4%   0.966 +/- 0.015
  positive                    168      52   31.0%   0.690 +/- 0.036

what the second labeller said
  contradicted                  2       2  100.0%   0.000
  corroborated                295      49   16.6%   0.834 +/- 0.022
  unwitnessed                  16       6   37.5%   0.625 +/- 0.121
```

## Signal 1: the filler's own confidence — dead

```text
AUC separating its right answers from its wrong ones    0.283
```

Below `0.5`, which would mean higher self-reported confidence predicts a
*worse* cell. It does not mean that. It is computed over 62 filled cells
containing 2 errors, so the whole estimate rests on where those two sit. It is
noise, and the earlier `0.61` estimate — measured on a larger, denser export —
is the better one. Either way it is far too weak to gate on.

There is a second reason it can never matter here: the filled cells are 96.8%
correct. Filtering them can remove at most 2 of the teacher's 57 errors.

## Signal 2: agreement between the two labellers — dead

```text
corroborated    295
unwitnessed      16
contradicted      2
```

The signal has no variance. Where `b23_full` speaks about a cell the parser
answered, it agrees 99.3% of the time. Dropping every contradicted cell moves
the teacher from 18.2% wrong to 17.7% — two cells.

This was the one lever nobody had pulled, and it deserved measuring precisely
because *removing* a contradicted cell is not *overriding* it, and only
overriding had been tested (B23 lost specificity; B24X put replacement at
nothing, 95% CI [-0.0100, +0.0035]). The answer is that there is almost nothing
to remove.

## Signal 3: what was said — real, large, and already spent

```text
negated     3.4% wrong    0.966 +/- 0.015
positive   31.0% wrong    0.690 +/- 0.036
```

A 27.6-point gap either side of standard errors of 0.015 and 0.036. This is not
a marginal finding; it is the largest structure in the teacher. **52 of the 57
errors are positive calls.**

It is also the finding that produced the negated-only fill rule, and that rule
has already extracted what the merge step can extract. The remaining positives
are the frozen parser's own, and they cannot be dropped:

- overriding them is B23, measured and refused;
- removing them wholesale leaves 145 positive-free cells and no classifier.

## Where the parser's errors actually are

```text
base    251 cells    55 wrong    21.9%
filled   62 cells     2 wrong     3.2%
```

96.5% of the teacher's errors are B6's own cells, and the parser has no
confidence of its own. That is *why* the column is a constant: there was never
anything to put in it.

## The one live option, and why it is not taken

Precision is the Bayes-optimal soft target for a noisy label. Measured:

```text
                    in use    measured
positive target      0.85       0.690
negative target      0.05       0.034
```

The negative target is near enough already. The positive target is 0.16 too
confident, and the machinery to change it exists and is plumbed:
`label_confidence_positive_target` in `config/training.yaml`, recorded in the
checkpoint as a distinct supervision policy.

It is not taken because `config/training.yaml` says, in its own header:

> Do not tune these values after Expert-58.

Reading 0.690 off the 58 studies and feeding it into training is exactly that.
The estimate is 0.690 +/- 0.036, so the honest interval is roughly 0.62 to 0.76,
and choosing a point inside it spends the only expert-truth proxy the project
has on a number that will not reproduce.

The asymmetry is in any case **already encoded**, qualitatively and from before
this measurement:

```text
POSITIVE_WEIGHT   0.50
NEGATIVE_WEIGHT   1.00
```

Somebody already decided positives are less trustworthy and halved them. This
audit confirms the direction was right. Whether 0.50 should be 0.35 is precisely
the kind of question 58 studies cannot settle.

## What this closes

Point 3 is answered: the confidence column has nothing to hold. Two candidate
signals are measured and dead, so nobody need look at them again. The third is
real and already acted upon.

The teacher's remaining fault is concentrated, named, and outside what any
confidence scheme reaches: **the frozen parser calls 31% of its positives
wrong.** Fixing that means a better parser or a better labeller, not a better
weight.
