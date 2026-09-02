# The two repairs that were never combined

## Status

**MEASUREMENT TOOL READY. THE MEASUREMENT HAS NOT BEEN RUN.**

Nothing here changes a label. It asks one question whose answer decides whether
a merge is worth building at all.

## The situation

Two independent repairs were built on top of frozen B6 v1.2.1:

```text
B6 v1.2.1
  |
  +-- B23 fill        the LLM answers cells B6 left silent
  |                   negated-only variant is the best teacher measured
  |                   18.2% wrong on the 58 experts, 0.8043 balanced accuracy
  |                   THIS IS WHAT THE RUNS TRAIN ON
  |
  +-- Phase 7 / 8     translate the report, re-run the unchanged B6 parser
                      1053 of 1229 unreadable studies rescued
                      3901 cells recovered, +27.62% over B6
                      COMPLETE, FROZEN, NEVER MERGED INTO THE TRAINED TEACHER
```

Phase 7 ran. Phase 8 froze the merged artefact. Phase 9 v2 trained a matched
pair on it. None of that reached the teacher the current runs use, because the
fill merge was built from B6 directly.

## Why not simply merge them

Because the overlap is unmeasured and could be nearly total. The LLM filler ran
over the whole report-only population, including all 1,229 studies B6 could not
read. If it already answered most of them, the rescue adds a handful of cells
and the cost is a fresh teacher with a new provenance chain. Merging first and
counting afterwards gets that backwards.

## The one thing to measure first

```bash
PYTHONPATH=developments/src \
python -m rsna_knee.teacher_coverage_audit \
  --export runs/091_negated_fill_merge/structured_labels.csv \
  --phase7-root runs/report_translation_rescue_full \
  --out-json runs/091_negated_fill_merge/coverage.json
```

It prints how many studies the current teacher still leaves entirely blank, and
splits the rescued cells into two piles that are **not** the same decision:

```text
study is still wholly silent here
    -> filling it is exactly the frozen Phase-8 policy

study was silent under B6, but the filler has since answered part of it
    -> filling the rest is a NEW policy, never frozen, never measured
```

Phase 6 froze "no translated cell may enter a B6-active study". These studies
were not B6-active; the filler made them active. Whether that clause reaches
them is a judgement, so the audit reports the two totals apart and makes no
choice.

## Two warnings that apply whatever the count says

**The expert audit cannot see this change.** Not one of the 58 gold studies is
in the 1,229 -- gold and report-only are disjoint by construction. A rescued
teacher scores *identically* on that surface, to the last decimal. The ruler
that settled the negated-only question is blind here.

**The rescue skews positive, and false positives are the known fault.** It adds
2,719 positive cells against 1,182 negative. The teacher's measured errors on
the 58 experts are 106 false positives against 5 false negatives. Some targets
are worse than the aggregate:

```text
Synovitis      35 positive     0 negative
Medial OA     199 positive     7 negative
Lateral OA    132 positive     5 negative
PF OA         253 positive    23 negative
```

Synovitis is the standing B26 warning restated: its scarce negatives were
reporting habit, not a missing-label defect, and filling them made expert
Synovitis AUC fall.

Phase 7's governance forbids fixing this by dropping targets after seeing the
result, and that clause still binds. The honest options are one global policy,
or a state-restricted global policy declared before the numbers are read --
negated-only, matching the rule the fill merge already adopted for the same
reason.

## What is already known about whether it helps

Phase 9 v2 trained a matched pair, control against B6 + rescue, same studies,
same series, same architecture:

```text
weighted BCE      -0.0099   95% CI [-0.0199, +0.0001]   P(better) 0.974
macro ROC AUC     +0.0032   95% CI [-0.0085, +0.0151]   P(better) 0.690
```

Directionally favourable, statistically inconclusive on both. That is the whole
evidence base. It was measured against B6 alone, not against the negated-only
fill merge, so it does not answer whether the rescue still adds anything once
the filler has run.

## Order of work

```text
1  measure the overlap                      the command above, minutes
2  decide the policy before merging         global, or global negated-only
3  build the merged teacher                 b23_fill_merge, chained
4  confirm the expert score is unchanged    it must be, exactly
5  train, and let the hidden test judge     the only ruler that can
```

Step 4 is not a formality. If the expert-58 score moves at all, a rescued cell
has reached a gold study and the merge is wrong.
