# B50 — ordered slice-sequence sparse MIL

## Status

**PROTOCOL AND FRESH SELECTION GATE PREPARED / NOT IMPLEMENTED / NOT RUN.**

B50 is the first proposed experiment after B49. It is not an approved training
run until the fresh selection split is created once, its SHA-256 is recorded,
and its coverage report confirms that all twelve weak targets are measurable.

## Why this is different

B42/B48/B49 represent local evidence at individual 2.5-D slice centres. They
provide each token with an absolute slice coordinate, but they do not let one
slice feature update another according to their ordered position through the
series. B48/B49's global-query compatibility residual is also not a substitute:
the query is detached and it adds a target-wise cosine score to each local token
independently.

B50 asks one new question:

> Does a lightweight, position-aware within-series sequence residual improve
> local sparse evidence over an otherwise matched position-hidden control?

This is not another crop, tile, resolution, centre-count, top-k, query-rank,
plane-router, duration, calibration, blend, or gold-weight experiment.

## Frozen data and validation boundary

The B48/B49 scanner split has been inspected and is therefore spent for new
architecture selection. B50's fresh gate is made only from its former `train`
rows:

```text
B48/B49 parent split
  parent train rows only
    -> B50 train
    -> B50 validation_seen_scanners
    -> B50 validation_unseen_scanners  (primary endpoint)

  parent validation_seen_scanners + holdout_unseen_scanners
    -> B50 excluded_prior_surface
```

The fresh B50 unseen profiles are disjoint from B50 training profiles. The seen
comparator contains only profiles that remain represented in B50 training. All
former B48/B49 validation rows are excluded from B50 entirely. This is fresh as
a B50 **model-selection surface**, not an external cohort: its candidate rows
were training data in the completed B48/B49 experiments.

The builder is deterministic and write-once. The launcher also verifies that
both B49 arms completed, that the parent split hash matches, and that it is not
about to overwrite a B50 gate:

```bash
B50_SELECTION_ROOT="$PWD/runs/083_Experiment_B50_ordered_slice_sequence_mil/b50_ordered_slice_selection_split" \
  bash developments/scripts/prepare_b50_ordered_slice_gate.sh
```

It uses a new frozen salt, a 20% whole-scanner-profile unseen holdout from the
parent training pool, and a same-size UID-hash selected seen-scanner comparator.
It refuses to write if a B50 split already exists, if the B48 parent SHA fails,
if headers differ from the parent profile assignment, or if any target cannot
be measured on the fresh unseen surface.

## Intended matched architecture

Both arms will inherit B42's fixed data and optimisation contract:

```text
native 90% crop -> one constant-area aspect-preserving resize -> ragged encoder
32 deterministic 2.5-D centres per series
frozen B34 global hierarchy and base logits
6x6 local cells per slice, TopK=8 sparse MIL
two epochs, effective batch 2, report-only weak labels
one seed (2026), no checkpoint selection
```

The B42 base-logit path must remain exactly reconstructible. B50 can only add a
zero-start residual to the local evidence path.

| Arm | Same sequence block | Difference |
|---|---|---|
| `position_hidden_control` | two-layer bidirectional slice-context block over all 32 slice features | receives zero positional basis, so it can use the set of slices but not their order/spacing |
| `ordered_position_candidate` | identical width, layers, heads, loss, optimiser, seed and data | receives the real normalized slice positions after stable physical-position sorting |

The candidate must stable-sort each series' 32 centres by physical position
before contextualization, because B42's first 16 historical centres and extra
16 dense centres are intentionally stored in a nested—not depth-sorted—order.
Its context residual is scattered back to the original token order before the
unchanged B36 sparse head. Duplicate positions in short series retain stable
original-index order.

The exact block dimensions, position-basis projection, parameter count, and
zero-initialisation checks will be committed only **after** the B50 split hash
exists. They may not be selected from a score on that split.

## Predeclared endpoint

The primary result will be candidate minus control macro ROC AUC on
`validation_unseen_scanners`, using all 12 targets. The seen-scanner score is a
domain-gap safety comparison, not a second selection opportunity.

The candidate will be supported only if all of the following are true:

```text
unseen delta >= +0.010
paired 95% CI lower bound > 0
P(candidate > control) >= 0.95
at least 7 of 12 targets improve
every leave-one-target-out macro delta > 0
seen-scanner delta >= -0.005
candidate domain-gap increase <= +0.005
```

Otherwise B50 is `no_support` or `inconclusive` under this frozen rule. No
Kaggle submission, B50 seed sweep, architecture-width sweep, calibration, or
blend is authorised unless the gate supports the candidate.
