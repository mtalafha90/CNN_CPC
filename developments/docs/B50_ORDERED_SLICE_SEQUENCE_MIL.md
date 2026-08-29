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
about to overwrite a B50 gate. In a clean B50 worktree, point `HEADER_CSV` at
the completed B49 header audit rather than recreating that audit:

```bash
export HEADER_CSV="/media/talafha/Disk_1/CNN_CPC_b49_run/runs/dataset_header_audit/header_by_series.csv"
export B49_ROOT="/media/talafha/Disk_1/CNN_CPC_b49_run/runs/082_Experiment_B49_native_tiled_multiscale_mil/b49_native_tiled_multiscale_mil"
export B50_SELECTION_ROOT="$PWD/runs/083_Experiment_B50_ordered_slice_sequence_mil/b50_ordered_slice_selection_split"

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

## Amendment: why the B48/B49 endpoint could not have answered this question

**Added 2026-08-29, before any B50 implementation, from measurements on the
completed B48 and B49 runs alone. No new training was involved.**

Every model in this family scores

```text
z = z_B34 + tanh(g) * z_local
```

with `z_B34` frozen and identical in both arms of a matched pair. The completed
runs recorded the gate once per epoch. At their fixed two-epoch endpoint:

```text
                 |tanh(g)| mean    max      epoch 1 -> 2
B48 candidate       0.025348    0.044643    0.00895 -> 0.02535
B48 control         0.025128    0.045040
B49 candidate       0.021934    0.051947    0.00793 -> 0.02193
B49 control         0.021788    0.051565
```

Two facts follow. The local branch reaches the score at roughly two per cent
strength. And **both arms independently learned the same gate**, to within one
per cent — so the mechanism under test did not change how much the model trusted
local evidence, and the differing component entered the score attenuated about
fiftyfold.

Comparing the two arms' saved predictions directly confirms the consequence.
Because the base cancels, what remains is the local branch:

```text
                          spearman   discordant pairs   max possible |dAUC|
B48 unseen scanners        0.99997        0.001532            0.0015
B49 unseen scanners        0.99993        0.002382            0.0024
```

An ROC AUC is the share of positive/negative pairs ordered correctly, so two
models' AUCs differ only on pairs they order differently. The two arms ordered
99.8% of pairs identically.

**Both experiments predeclared a `+0.010` threshold that exceeded the largest
value their measurement could produce** — by 6.5x for B48 and 4.2x for B49.
Neither could have been supported regardless of merit. Their recorded verdicts
of "no support" are therefore statements about the endpoint, not about global
conditioning or native tiling.

Normalised by what was measurable, the two are also not equivalent nulls:

```text
        measured      ceiling    share of the available headroom used
B48     0.0000749     0.0015                  5%
B49     0.0005468     0.0024                 23%
```

B49's mechanism reordered roughly four to five times more of what it was able to
reorder. That comparison is not evidence for promoting B49, and no B49 tuning is
authorised by it; it is recorded because the original analysis discarded it.

B50 as originally written repeats the same design — a zero-start residual inside
the local path, scored on combined logits, against a `+0.010` bar. It would
return `no_support` whatever ordered slice reasoning is worth.

## Predeclared endpoint

**Primary.** Candidate minus control macro ROC AUC of the **local evidence path**
on `validation_unseen_scanners`, all 12 targets. This is the quantity B50's
question is about. `local_logits` is already computed by the existing evaluation
and discarded; B50 must persist it.

**Secondary, reported always, never a selector.** The same delta on the combined
prediction, which is what a submission would use, plus the gate values at the
endpoint. A local-path gain that does not reach the combined score is a real
result about the gate rather than about the sequence block, and must be recorded
as such rather than reported as a win.

The evaluation must also persist `global_prediction` alongside the combined
prediction, so the base and local contributions stay separable afterwards
without re-running inference.

**Mandatory power check, before any verdict is written.** Compute the discordant
pair fraction between the two arms on the primary surface and report it beside
the threshold:

```text
if ceiling < threshold        the endpoint could not have detected the effect;
                              the run is `endpoint_underpowered`, NOT `no_support`
```

`no_support` may only be recorded when the measurement was capable of producing
a result that would have passed. This clause exists because B48 and B49 were
both filed as `no_support` under thresholds their measurements could not reach.

The seen-scanner score remains a domain-gap safety comparison, not a second
selection opportunity.

The candidate will be supported only if all of the following are true:

```text
local-path unseen delta >= +0.010
paired 95% CI lower bound > 0
P(candidate > control) >= 0.95
at least 7 of 12 targets improve
every leave-one-target-out macro delta > 0
seen-scanner delta >= -0.005
candidate domain-gap increase <= +0.005
discordant-pair ceiling >= +0.010 on the primary surface
```

Otherwise B50 is `no_support`, `endpoint_underpowered`, or `inconclusive` under
this frozen rule. No Kaggle submission, B50 seed sweep, architecture-width
sweep, calibration, or blend is authorised unless the gate supports the
candidate.

## What this amendment deliberately does not do

It does not touch the gate. Letting `tanh(g)` train faster, initialising it away
from zero, or extending past two epochs would all raise the local branch's
influence, and one of them may well be the single highest-value change available
to this project — the gate was still nearly tripling between epoch 1 and epoch 2
when training stopped, so it has never been allowed to settle.

But that is a different experiment with a different question, and folding it
into B50 would confound "does ordered slice reasoning help" with "does the local
branch matter when it is actually connected". It should be declared separately.

The zero-start gate is also a governance feature, not an oversight: it makes the
residual a no-op at initialisation so the parent endpoint is exactly
reconstructible. Any change to it needs its own protocol.

Recorded observation for whoever writes that experiment: at the B48/B49
endpoint the gate is widest for Effusion, Contusion, Medial Meniscus and PF OA,
and is `+0.000044` for ACL — the weakest target in the project, and the one B45's
plane routing was designed to repair. The local branch contributes essentially
nothing to ACL, which is a mechanical explanation for why B45 could not move it.
