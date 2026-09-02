# The teacher, measured against expert truth

**Date:** 2026-09-02
**Status:** COMPLETED MEASUREMENT. No gate, no model change. A number that was
missing and is now recorded.

## Why this was run

Five architectures span `0.694` to `0.716` on the hidden test: about `0.02`
between the worst and the best. Every one of them is trained on labels derived
from radiology reports, and every one is scored against expert labels. Nobody
had measured how far apart those two things are for the export that actually
trained them.

## Result

All three exports, scored against the same 58 expert studies and the same 696
expert cells, at the default `min_confidence` of 0.75:

```text
                  b6_v121    b23_full   fill_merged
coverage           0.3606     0.6351     0.6580
precision          0.6736     0.6647     0.6527
sensitivity        0.9721     0.9869     0.9768
specificity        0.5752     0.4919     0.4988
balanced accuracy  0.7736     0.7394     0.7378
cells called      251/696    442/696    458/696
disagreements          55        103        111
```

`fill_merged` is the export that trained B42, B51 and B52.

## What it says

**The teacher in use is worse than the rule-based teacher it extended, on every
quality measure. It wins only on coverage.**

```text
balanced accuracy   0.7736  ->  0.7378    -0.036
specificity         0.5752  ->  0.4988    -0.076
precision           0.6736  ->  0.6527    -0.021
coverage            0.3606  ->  0.6580    +0.297
```

The merge nearly doubled the number of cells answered and doubled the number of
wrong ones. Per cell:

```text
B6 alone                       251 cells,  55 wrong,  21.9% error
the cells the merge added      207 cells,  56 wrong,  27.1% error
merged export                  458 cells, 111 wrong,  24.2% error
```

**`base_cells_overridden: 0` protects B6's calls, not the export's quality.**
The merge does exactly what it promises -- every B6 call survives, only silent
cells are filled -- and the pooled quality still falls, because the filled cells
are worse than the ones B6 already had. That guarantee is about reproducibility.
It was never a guarantee about accuracy, and it should not be read as one.

## The pilot did not hold

`B23_LLM_REPORT_LABELS.md` records a pilot in which B23 improved state-only
macro AUC from `0.7025` to `0.8125`. That is a different metric on a different
subset, and the pilot's own status line warns that it is "descriptive/post-hoc,
not confirmatory validation, because aggregate information from this reused
expert surface influenced the prompt design".

On the full corpus, by balanced accuracy, B23 is **below** B6: `0.7394` against
`0.7736`. A pilot result that does not reproduce at full scale is the expected
failure mode of a prompt tuned with sight of the surface it is scored on, and
this is that failure mode.

The formal gate B23 failed was specificity. The full run fails it by more than
the pilot did: `0.4919` against B6's `0.5752`, where the pilot showed `0.5678`.

## Specificity is the number to look at

```text
sensitivity   0.9768    when the expert says yes, the teacher says yes
specificity   0.4988    when the expert says NO, the teacher agrees half the time
```

The teacher almost never misses a finding and calls roughly half of the true
negatives positive anyway. A model trained on it inherits that: it learns to
say yes. AUC is a ranking measure, so a uniform bias would be survivable -- but
this is not uniform. It is noise concentrated entirely on the negative cells,
which is precisely the noise that destroys a ranking.

That is a concrete, measured explanation for a hidden score near `0.71` from a
model that scores `0.835` against the labels it was trained on.

## One discrepancy to resolve

The recorded B6 v1.2.1 reference in `b23_llm_labels.py` does not match what B6
v1.2.1 measures here:

```text
                measured   recorded
coverage          0.3606     0.3606     matches exactly
precision         0.6736     0.6905
sensitivity       0.9721     0.9749
specificity       0.5752     0.6061
```

Coverage agrees to four decimals and the other three do not, on the same export
and the same studies. The likely cause is a different `min_confidence` or a
difference between `b6_gold_audit`'s state-only computation and this module's.
It is small and it does not change any conclusion above -- the merged export is
below B6 on either set of B6 numbers -- but a reference that is quoted and does
not reproduce should be either explained or replaced.

## Where the errors are, measured

`mismatches.csv` for the merged export, 111 disagreements:

```text
expert says NO, report says YES   106
expert says YES, report says NO     5
```

Spread evenly across all twelve findings -- 18, 15, 12, 9, 9, 9, 9, 7, 7, 7, 5,
4 -- so this is systemic over-calling, not one broken target. Per-target repair
will not help.

Split by what the teacher says, on B23's 442 committed gold cells:

```text
negated cells      137    97.8% correct
positive cells     305    67.2% correct
```

**The teacher is reliable when it says no and unreliable when it says yes.**

## The confidence column is a constant

`b23_llm_labels.py` stamps `B23_DEFINITE_STATE_CONFIDENCE = 0.90` on every
definite call, right or wrong. The column that gates supervision at `>= 0.75`,
and that `confidence_weighted_bce` weights by, therefore takes exactly two
values: `0.90` and `0.0`. **No threshold on it can separate anything**, and
every supervised cell currently carries identical weight.

The LLM's own confidence is recorded as `__model_confidence`, deliberately
unused, and never reaches `training_targets.csv`. Measured against being right:

```text
correct calls   mean 0.974   sd 0.039
wrong calls     mean 0.957   sd 0.055
```

About `0.61` AUC at detecting its own errors. Thresholding it buys little:

```text
conf >= 0.99    255 cells (57.7%)   80.0% correct
conf >= 1.00    242 cells (54.8%)   80.2% correct
baseline        442 cells           76.7% correct
```

The comment in the source calls it "an uncalibrated self-report" and declines to
threshold on it. That judgement is confirmed.

## Corroboration beats self-confidence

Keep every `negated` cell; accept a `positive` only where the frozen B6 parser
independently says positive too:

```text
agreement rule  296 cells (67.0%)   83.4% correct
conf >= 0.99    255 cells (57.7%)   80.0% correct
```

**It dominates**: more cells kept and more of them right. It also has no free
parameter, which matters more than the margin -- a threshold chosen by looking
at these 58 studies is the same move that produced B23's pilot result and lost
it again at full scale.

## The ceiling this exposes

```text
B23 positive, B6 also positive    159 cells    71.1% correct
B23 positive, B6 silent           145 cells    62.8% correct
```

**Two independent extractors agreeing on a positive is still wrong 29% of the
time.** The disagreement is therefore not mostly between the parsers and the
report. It is between the *report* and the *expert* -- the radiologist writing
the report and the radiologist labelling for this task are not calling the same
knees positive.

That bounds what better extraction can buy. A perfect reader of these reports
would still inherit that 29%.

## The cost of the agreement rule, stated plainly

```text
dropping the uncorroborated positives loses 91 correct labels
to remove 54 wrong ones
```

Better label purity, less supervision. Whether that trade helps a trained model
is not settled by any number here: purity on 58 studies is a proxy, and the only
test that resolves it is a retrain and a hidden submission.

## Recommended: agreement as a weight, not a gate

The loss already supports per-cell weighting and that mechanism is currently
inert, because every cell carries the same `0.90`. Giving it something real to
carry keeps the 91 good labels instead of discarding them:

```text
negated                     full weight    97.8% correct
positive, corroborated      full weight    71.1% correct
positive, uncorroborated    half weight    62.8% correct
```

Half is chosen as "trust it half as much", not by sweeping for the best number
on 58 studies. The one free parameter is soft and monotone rather than a
hard cut-off, which is the safer shape when the surface you can measure on is
this small.

## Result: filling negatives only beats every other teacher measured

```text
                 cover    prec    sens    spec  bal acc   cells  wrong     err
b6_v121         0.3606  0.6736  0.9721  0.5752   0.7736     251     55   21.9%
b23_full        0.6351  0.6647  0.9869  0.4919   0.7394     442    103   23.3%
fill_merged     0.6580  0.6527  0.9768  0.4988   0.7378     458    111   24.2%
negated_fill    0.4497  0.6736  0.9568  0.6518   0.8043     313     57   18.2%
```

`negated_fill` is the best on balanced accuracy and specificity, and it beats
B6 -- the teacher whose specificity the whole fill-only argument was built to
protect -- on both, while carrying 25% more coverage and a **lower** error rate.

Against B6, cell for cell:

```text
what negated_fill adds to B6      62 cells,  2 wrong   ->  96.8% correct
B6's own cells, for comparison   251 cells, 55 wrong   ->  78.1% correct
```

**The added cells are cleaner than the ones already there.** That is the
opposite of what filling both states did, where the added cells ran at 27.1%
error against B6's 21.9%.

The trade is visible and it is the right one for this failure:

```text
specificity   0.5752 -> 0.6518   +0.077
sensitivity   0.9721 -> 0.9568   -0.015
```

Sensitivity falls because some added `negated` cells are wrong where the expert
says yes. That costs little: the measured failure was 106 false positives
against 5 false negatives, so buying specificity with sensitivity is spending
the currency in surplus.

On the full corpus the export carries 25,524 usable cells across 4,349 studies,
48.9% coverage, with every parser call preserved and zero positives added.

## Why this evidence is stronger than the margin suggests

`+0.031` balanced accuracy sits right at the `+/-0.03` that 58 studies resolve.
Taken alone that is one study short of silence.

What makes it worth acting on is that **the rule was chosen from the mechanism
before the number existed**. The prediction written down in advance was that
filling only negatives would raise specificity, because specificity is precisely
the rate at which true negatives are identified and the measured failure was
over-calling positives. The prediction was conservative -- it said specificity
would return *near* B6's -- and the measurement exceeded it.

That is a different kind of evidence from a threshold swept until a number came
out well, which is what produced B23's pilot and lost it again at full scale.
There is no free parameter here to have overfitted.

## What it does not settle

This is a **labeller** audit. It says the labels are better. It does not say a
model trained on them scores better, and nothing here should be quoted as
though it did. Fewer positive labels also means a different class balance,
which `target_balance_multipliers` will absorb but not for free.

The only ruler that settles it is a retrain and a hidden submission.

## What follows

Nothing here changes a model, and nothing here is a gate. What it changes is
where the next effort goes.

```text
five architectures, hidden          0.694 -> 0.716     0.022 of spread
the teacher, balanced accuracy      0.7378             what everything sits under
```

The obvious first probe costs no GPU: `--min-confidence` is 0.75 by default, and
raising it trades coverage for specificity. If specificity recovers towards B6's
while coverage stays above B6's, there is a better teacher available from the
exports that already exist, with no new labelling at all.

That is a measurement, not a promise. It has not been run.
