# The two repairs that were never combined

## Status

**COMPLETE. MERGED TEACHER BUILT AT `runs/092_rescued_negated_fill`, UNTRAINED.**

The measurement was run, the frozen half of the rescue was merged, and the
expert surface confirmed unchanged. No model has been trained on it.

## Result

The overlap was most of it. The LLM filler had already reached three quarters of
the studies B6 could not read:

```text
studies B6 could not read at all        1,229
studies the filler still leaves blank     321
```

Of the 3,901 rescued cells, 1,176 were already answered. The remaining 2,725
split into two piles that are not the same decision:

```text
into wholly silent studies      678 cells    659 pos    19 neg    228 studies
into studies the filler reached 2,047 cells 1,973 pos   74 neg    741 studies
```

Both piles are about 97% positive, because the negated-only filler had already
taken every negated cell the rescue had to offer. What was left was positives.

**The frozen pile was taken. The 2,047-cell pile was refused.** Taking both
would have raised positive labels by 38% and negative labels by 0.5%, in a
teacher whose measured fault is 106 false positives against 5 false negatives.

```text
before    25,524 cells    48.9% coverage    321 studies blank
after     26,202 cells    50.1% coverage     93 studies blank
```

The 93 remaining are unreachable: not by B6, not by the LLM, not by translation.
That is 2.1% of the population and no built mechanism is left to try.

### The B26 shape is absent

B26 failed by adding *negatives* to Synovitis. This adds 19 Synovitis
*positives* — the opposite operation, and 2.8% of the pile. Nothing is
concentrated; the 678 cells spread across all twelve findings roughly in
proportion to how common each is, the largest single block being ACL at 102.

These are also not LLM positives. They are the frozen B6 parser's own positives,
read off a translated report, and B6 positives are accepted everywhere else in
this teacher. Taking them is consistent rather than a new leniency.

### The expert surface is unchanged, as required

No gold study is in the rescued population, so the audit had to come back
identical, and did:

```text
                  rescued_negated_fill    negated_fill
coverage                        0.4497          0.4497
precision                       0.6736          0.6736
sensitivity                     0.9568          0.9568
specificity                     0.6518          0.6518
balanced accuracy               0.8043          0.8043
cells called                       313             313
disagreeing with the expert         57              57
```

That is the check, not a formality: a single moved digit would have meant a
rescued cell had reached a gold study.

### What it does not do

It does not fix ACL. 102 new ACL positives against 3,361 existing is a 3%
increase, and ACL sits at 0.5478 against expert truth. That remains its own
problem, and a larger one — ACL and MCL reaching 0.70 is worth +0.029 macro.

The change is 2.6% of the surface and cannot be scored locally. It does not
justify a training run of its own; it should ride along with one already
planned.

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
  --phase7-root runs/056_Experiment_AUDIT_P7_full_translation_rescue/\
report_translation_rescue_full \
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
1  measure the overlap                      DONE   678 of 2,725 eligible
2  decide the policy before merging         DONE   frozen pile only
3  build the merged teacher                 DONE   runs/092_rescued_negated_fill
4  confirm the expert score is unchanged    DONE   identical to four decimals
5  train, and let the hidden test judge     PENDING
```

Step 5 needs `--expected-cells 26202`: the surface guard is pinned to the base
checkpoint's own count, and a deliberate teacher change must state its number
rather than have the guard relaxed for everyone.

The merge command, for the record:

```bash
PYTHONPATH=developments/src \
python -m rsna_knee.b23_fill_merge \
  --base   runs/091_negated_fill_merge/structured_labels.csv \
  --filler runs/057_Experiment_AUDIT_P8_merged_translation_supervision/\
translation_rescue_supervision_v1/training_targets.csv \
  --only-silent-studies \
  --fill-states both \
  --out-root runs/092_rescued_negated_fill
```

The Phase-8 file is B6 plus the rescue, and the base already holds every B6
cell, so anything the filler can add is by construction a rescued cell. No
conversion from the long-form `recovered_cells.csv` was needed.
