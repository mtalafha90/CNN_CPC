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
