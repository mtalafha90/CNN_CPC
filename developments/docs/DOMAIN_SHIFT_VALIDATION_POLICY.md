# Domain-shift validation policy — a surface held out by scanner

**Date:** 2026-08-25

**Status:** IMPLEMENTED / NOT RUN. The split has not been created on the real
data, so every number in this document is a description of the method, not a
result.

This closes item 9 of the post-B45 plateau retrospective and step 5 of its
recommended sequence, which asks for a site/scanner-grouped validation surface
to be frozen in parallel with B46. It is deliberately independent of B46: it
adds no GPU work, touches no frozen contract, and reads no B46 output.

## The gap it fills

Every split this project has frozen was drawn by hashing the study UID. Hashing
is honest — it cannot be steered by peeking at labels — but it scatters each
scanner across both sides. The consequence:

```text
current splits      every scanner appears in training AND in validation
so a model that     "this is what a Siemens 1.5T knee looks like"
learns              scores well on both sides
and nothing         measures what happens on a machine it has never met
```

The challenge data is multicentre by construction. A model can therefore look
strong while leaning on acquisition regularities that sit on both sides of every
split this project owns.

## The group key, and what it is not

There is no institution column in `train.csv`, and DICOM institution tags in
this dataset are frequently blank or anonymised. The group key is therefore
built from what is reliably present:

```text
manufacturer_family | manufacturer_model | field_strength_bin
```

Field strength is bucketed, not compared exactly, because scanners report values
such as `1.494` and `2.8936`; an equality test would shatter one machine into
several groups.

**This is a scanner proxy, not a site.** Two hospitals running the same model
share a group. One hospital running two scanners is split across two groups. The
resulting gap is a **lower bound** on domain sensitivity: real centre shift also
carries differences in population, protocol and reporting habit that this key
cannot see.

A study whose series disagree — a stitched or re-imported series — is assigned
the profile most of its series carry, because it has to land on exactly one side.
The count of such studies is reported rather than hidden.

## The three groups

```text
train                       model trains on these only
validation_seen_scanners    machines the model DID train on
holdout_unseen_scanners     machines the model has NEVER met
```

The third group is the point. The second exists because a score on unseen
scanners means very little alone: it cannot distinguish "this model does not
travel" from "this model is not very good". The comparator is drawn by UID hash
from the training-side scanners and sized to match the holdout, so both scores
carry similar noise.

```text
domain gap = score(validation_seen_scanners) - score(holdout_unseen_scanners)
```

taken over `comparable_targets` only — the findings that have both a positive
and a negative case on **both** sides. A target that cannot be scored is named,
never averaged in as a missing value.

The 58 official gold studies are excluded entirely. They are the only clean
labels in the project and B46 is using them.

## How the split is chosen

Whole profiles move together. Among them the choice is a deterministic greedy
walk: at each step the profile that brings the holdout's per-target positive rate
closest to the whole population's is added, and SHA-256 over a frozen salt breaks
exact ties — the same construction B46 uses for its gold folds.

Balancing prevalence is not cosmetic. Taking profiles in hash order alone can
easily produce a holdout in which a rare finding has no positive cases at all,
and an AUC needs both classes to exist.

```text
salt        CNN_CPC|domain-split|scanner-grouped|2026-08-25
default     20% of report-only studies held out
refuses     any split where a profile lands on both sides
refuses     a comparator drawn from a scanner training never sees
records     SHA-256 of the frozen split
```

## What it can and cannot tell you

**It can** estimate how much of the model's performance is acquisition-specific,
and identify which findings degrade most on unfamiliar machines.

**It cannot** tell you whether the model is right. The labels here are still
report-derived, so this surface measures domain generalisation and nothing else.
A model that copies the parser's mistakes consistently across all scanners will
show a small gap and still be wrong.

**It is not** independent hidden evidence, and a small gap does not authorise a
submission.

## Reading the result, declared in advance

```text
gap <= 0.01     acquisition shift is not a material bottleneck at this scale;
                stop spending effort here
0.01 - 0.03     present but second-order; note it and carry on
gap > 0.03      acquisition shift is comparable to the entire B37-to-B46
                improvement budget, and belongs in the next experiment's design
```

Also record, regardless of the gap:

- whether one profile holds more than 35% of studies, in which case a grouped
  split cannot separate it and the whole estimate is weak;
- which targets fell out of `comparable_targets`, since a shrinking comparison
  set makes the macro gap less trustworthy;
- the per-target gaps, because an aggregate driven by one finding has misled this
  project twice before.

Do not select checkpoints, thresholds or architectures from this surface without
declaring that in advance.

## How to run it

It needs no GPU. It does read DICOM headers, so **do not run the header audit
while B46 is training** — they will compete for the same disk.

The header audit, if `header_by_series.csv` does not already exist:

```bash
python -m rsna_knee.dataset_header_audit \
  --data-root "$DATA_ROOT" \
  --out-root runs/dataset_header_audit
```

Then the split itself, which is fast:

```bash
python -m rsna_knee.domain_shift_split \
  --data-root "$DATA_ROOT" \
  --header-csv runs/dataset_header_audit/header_by_series.csv \
  --labels-root "$LABELS_ROOT" \
  --out-root runs/domain_shift_split
```

It writes `domain_split.json`, `domain_split_by_study.csv` and
`domain_split.sha256`. Archive the SHA before any model is scored here.

## Implementation

```text
developments/src/rsna_knee/domain_shift_split.py
developments/tests/test_domain_shift_split.py
```
