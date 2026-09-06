# B54: one run, four fixes

## Status

**BUILT, WIRED AND TESTED. NOT RUN.**

The teacher is on disk at `runs/085_B54/teacher_final`. The model is wired into
B52's trainer behind `--spacing-geometry-csv`, with resume, a preflight, an
optimiser check and a post-run backstop. Nothing has been trained.

## What is in it

```text
1  spacing conditioning   the model is told how much knee each input holds
2  rebuilt teacher        B52's fill rule on a v1.3.1 base, not 091's
3  B6 v1.3                an OA vocabulary, and the list-negation guard
4  B47 native grid        the head stops pooling 196 cells onto 36
```

## The one thing to understand before starting

Four changes in one run means the score cannot be attributed. If it rises you
will not know which change earned it; if it falls you will not know which to
remove.

**One of the four is exempt.** The spacing conditioning is zero-initialised and
switchable, so a single trained checkpoint yields both arms: evaluate once
normally, once with `set_spacing_enabled(model, False)`, and the difference is
the spacing effect exactly. The teacher change and the grid change remain
confounded with each other. That is the accepted cost of one run.

## Why the teacher is rebuilt rather than reused

`runs/092_rescued_negated_fill` is the rescue applied on top of **091**, the
negated-only teacher. 091's teacher is the one already measured: the teacher
swap scored 0.638317 against B52's 0.678247, a loss of 0.0399 — twice the veto
threshold, worse on 9 of 12 targets.

Step 2 below rebuilds it with B52's own fill rule on a v1.3.1 parser base. The
rescued cells were then measured against that teacher and turned out to be
spent — see step 3.

---

## Step 0 — get the code

```bash
cd /media/talafha/Disk_1/CNN_CPC
git pull origin main
PYTHONPATH=developments/src python -m pytest developments/tests -q
```

Expect 1,925 passed, 1 skipped.

## Step 1 — B6 v1.3 report labels

```bash
PYTHONPATH=developments/src python -m rsna_knee.b6_v13_report_labels \
  --train-csv /media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection/train.csv \
  --out-root runs/085_B54/b6_v13
```

**Gate.** Read `runs/085_B54/b6_v13/v13_changes.json` before going on:

```text
cells_newly_answered                      1,814 at v1.3.0
fallback_cells_now_quoted                 2,093 at v1.3.0
cells_flipped_by_list_negation_guard         26 at v1.3.0
cells_silenced                            must be 0
cells_weakened_by_new_vocabulary          must be 0
cells_weakened_after_list_negation_guard  expect a handful
```

The first three should barely move at v1.3.1: the fix only suppresses answers
the vocabulary could not commit to, so placements and re-quotings stay.

`cells_silenced` above zero means v1.3 is removing calls, which it is not
allowed to do.

`cells_weakened_by_new_vocabulary` above zero is a hard stop. These are broad
anatomy words, validated for *placing* a cell and never for arbitrating one,
so they answer only with a committed state. A bare anatomy word must not turn
a confident call into `uncertain`, which drops its confidence from 0.90 to
0.25 on exactly the three targets with the worst coverage already.

**This gate has already fired once, on the real corpus, at 360 cells** — 338
of them PF OA. v1.3.0 protected calls the *aliases* made and not calls the
*fallback* made, and the OA targets live almost entirely on the fallback.
v1.3.1 fixes it. The counter stays so the fix is checked rather than trusted.

`cells_weakened_after_list_negation_guard` is a different thing and a small
number is expected. When the guard correctly reads "ACL: intact" as a negation
and the report also describes a tear, the contradiction is real; v1.2.1 only
looked confident there because it misread the list entry. It appears only on
targets that also show guard flips.

## Step 2 — fill the silent cells, on the v1.3 base

Locate the LLM fill export first; the numbered directory differs per machine:

```bash
find runs -name structured_labels.csv -path '*LLM_FILL*' | head
```

Then:

```bash
PYTHONPATH=developments/src python -m rsna_knee.b23_fill_merge \
  --base runs/085_B54/b6_v13 \
  --filler <the LLM fill export from the find above> \
  --fill-states both \
  --out-root runs/085_B54/teacher_step2
```

`--fill-states both` is B52's rule and is deliberate. `negated` is the rule
that produced the −0.0399 teacher; do not use it.

## Step 3 — add the rescued cells: MEASURED AND DROPPED

Do not run this. It was measured against the step-2 teacher and offers **one
cell in one study**; blank studies would go 48 to 47. Its other 203 cells need
the policy Phase 8 refused — filling studies the filler already reached.

`translation_rescue_supervision_merge` also refuses a v1.3.1 base by design
(`REQUIRED_B6_VERSION = "1.2.1"`, `EXPECTED_ORIGINAL_USABLE = 14123`), and the
measurement shows that refusal cost nothing.

**`runs/085_B54/teacher_step2` is the final teacher.** Promote it so the paths
below hold:

```bash
cp -r runs/085_B54/teacher_step2 runs/085_B54/teacher_final
```

See `THE_B54_TEACHER_AND_A_SPENT_RESCUE.md` for the full numbers.

### What step 2 produced, against B52

```text
                             B52 teacher   B54 teacher    change
cells answered                    34,010        34,842      +832
studies with no answer                57            48        -9
parser, clause recorded           11,491        15,004    +3,513
parser, no clause                  2,632           896    -1,736
no clause at all                   66.2%         56.9%
```

Every gate passed. The evidence-free osteoarthritis calls fell by two thirds,
which is what B6 v1.3 was built for.

## Step 4 — the geometry table

Already produced. If it is missing:

```bash
PYTHONPATH=developments/src python -m rsna_knee.slice_geometry_scan \
  --data-root /media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection \
  --workers 8 \
  --out-csv runs/slice_geometry_scan/series_geometry.csv \
  --out-json runs/slice_geometry_scan/summary.json
```

## Step 5 — preflight

`b54_spacing_run.preflight` refuses two ways this run could be silently
pointless: the spacing failing to resolve for most series, and the conditioning
never being installed. Call it from the training entry point before the first
epoch, with the assembled series records and the constructed model, and abort
on `passed: False`.

Expect `resolved_fraction` at 1.000 — the scan measured a usable spacing for
all 24,371 series — and `conditioning_sites` at 1 or 2 depending on whether the
head alone or the base as well is conditioned.

## Step 6 — train

The five edits are made. `b52_competition_training` now takes
`--spacing-geometry-csv`, and **every B54 branch is behind it** — omit the flag
and B52 runs exactly as it always has. 26 tests hold that gate shut.

```bash
cd /media/talafha/Disk_1/CNN_CPC

PYTHONPATH=developments/src python -m rsna_knee.b52_competition_training \
  --data-root /media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection \
  --labels-root runs/085_B54/teacher_final \
  --series-policy runs/020_Experiment_B12_variable_series/b12_variable_series/audit/series_policy.json \
  --base-checkpoint runs/067_Experiment_LLM_FILL_ALL_b6_preserved_llm_fill_all_targets/b6_plus_llm_fill_all_ft1/train/llm-filled/model.pt \
  --domain-split runs/083_Experiment_B50_selection_gate/b50_ordered_slice_selection_split \
  --spacing-geometry-csv runs/slice_geometry_scan/series_geometry.csv \
  --expected-supervision-cells 34842 \
  --epochs 6 \
  --out-root runs/085_B54/train \
  2>&1 | tee runs/085_B54/b54_train.log
```

### Two arguments that are not optional

**`--expected-supervision-cells 34842`.** `_report_only_surface` asserts the
teacher has exactly `B35_EXPECTED_CELLS` = 34,010 usable cells — the surface
every run from B35 to B52 trained on. Our teacher has 34,842, so the run stops
before the first epoch with "B48 weak supervision surface changed". That guard
is right: it exists to catch a teacher that moved without anyone meaning it to.
Declaring the new count satisfies it without weakening it — the check is still
a hard equality, so a wrong `--labels-root` or a half-finished merge fails just
as it did. The number used is written into the checkpoint under
`supervision.expected_usable_cells`.

**`--epochs 6`.** The default is 12; B52 ran 6 (`epochs_planned` in its
checkpoint, selecting epoch 5). Leaving the default would double the GPU time
*and* change the recipe being compared.

Expect 1,925 passed, 1 skipped.

### The four paths, and why they were nearly lost

B52 recorded fingerprints, not paths, and its log did not keep the command
line. Recovering these took an hour of archaeology, so they are written down
here — and the `tee` above means this run will not repeat the problem.

```text
--series-policy    runs/020_Experiment_B12_variable_series/
                     b12_variable_series/audit/series_policy.json
                   matched by its series_signature_sha256, 5c4bb1c5...

--base-checkpoint  runs/067_Experiment_LLM_FILL_ALL_.../
                     b6_plus_llm_fill_all_ft1/train/llm-filled/model.pt
                   the only path B52's checkpoint stored verbatim

--domain-split     runs/083_Experiment_B50_selection_gate/
                     b50_ordered_slice_selection_split
                   sha256 fa8eb88f... of its b50_selection_split.json
```

**`--domain-split` does not take a `domain_shift_split` output.** It takes a
B50 *selection gate*, whose file is `b50_selection_split.json`, loaded by
`load_b50_selection_gate` — "the fresh B50 gate, not the split B48 and B49
already spent". The gate is built only from the parent split's former `train`
rows with every B48/B49 validation row excluded, which is how 4,349 report
studies become 1,447 training and 548 validation. Searching for
`domain_split.json` finds nothing, and regenerating one produces a valid but
completely different artefact that the loader would reject.

Related, and worth knowing separately: `domain_shift_split` no longer
reproduces B48/B49's split either. Commit `8bd5f80`, "Fix B48 seen-scanner
comparator allocation", changed which studies land in the seen-scanner group.
The old artefacts are therefore not regenerable, only preserved.

### What changed for the second run

The first B54 run trained the conditioning to 0.07% of the sum it was added to,
so its ablation measured nothing. Two corrections, both declared before the
re-run and neither chosen by looking at a score:

**The term is scaled to where it can matter.** `install_spacing_conditioning`
now sets `scale = reference_scale(metadata_norm(module))`, the size at which a
weight of Frobenius norm 1 produces a p05-to-p95 spread equal to the
`plane + fluid + fat` sum. The weight no longer has to travel four orders of
magnitude to be relevant, and its learned norm now reads directly as the
fraction of the metadata scale it reached. Zero initialisation survives, so the
model still starts numerically identical to B52.

**It gets its own parameter group at the head rate.** It was in
`study_hierarchy` at 5e-6 — the rate that exists because those weights are
pretrained and a large step destroys them. The conditioning is fresh, like the
sparse head, which gets 1e-4 for exactly that reason. Watch for a fourth group
in the log:

```text
[B52]   spacing_conditioning  lr=1.000e-04  params=...
```

The scale used is printed and written into the checkpoint as
`spacing.conditioning_scale`, so this run can be reproduced and the first one
still means what it meant: a checkpoint with no recorded scale is read back at
1.0, which is what it trained with.

### Three guards, in the order they fire

**Preflight, before the first epoch.** Refuses a spacing that failed to resolve
for most series, and a conditioning that was never installed. Expect
`resolved_fraction` 1.000 — the scan measured a usable spacing for all 24,371.

**The optimiser check, once the groups are built.** This is the one that
matters. `b52_parameter_groups` builds three groups from the encoder,
`hierarchy_parameters()` and the head, and B50 fixes its hierarchy names in
`__init__` — before the conditioning exists. Left alone the conditioning would
reach no group, never update, stay at zero all run, and the ablation would
report no effect from a model never trained to use the spacing.
`B54SpacingConditionedMIL.hierarchy_parameters` includes it and
`assert_conditioning_will_train` checks the finished optimiser rather than
trusting that.

**The backstop, before the payload is written.** The conditioning starts at
exactly zero, so if it is still zero the run raises rather than reporting a
number that means nothing. Mirrors B49's encoder-fingerprint check.

### Resume

`recovery_latest.pt` is written to `--out-root` after every epoch — weights,
optimiser, schedule, loss scale and every generator, atomically. Re-running the
same command picks up at the next epoch and says so. It refuses a checkpoint
written by a different version, so move it aside if you deliberately restart.

## Step 7 — evaluate twice, from one checkpoint

```bash
cd /media/talafha/Disk_1/CNN_CPC

PYTHONPATH=developments/src python -m rsna_knee.b54_expert58_eval \
  --data-root /media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection \
  --checkpoint runs/085_B54/train/b52_best_model.pt \
  --base-checkpoint runs/067_Experiment_LLM_FILL_ALL_b6_preserved_llm_fill_all_targets/b6_plus_llm_fill_all_ft1/train/llm-filled/model.pt \
  --spacing-geometry-csv runs/slice_geometry_scan/series_geometry.csv \
  --out-root runs/085_B54/expert58 \
  2>&1 | tee runs/085_B54/b54_expert58.log
```

One load, two scoring passes: `spacing_on` and `spacing_off`. Same weights,
same 58 studies, same crops, same three centre offsets — differing only in
whether one learned vector is added to each series' metadata. That makes the
delta between them far tighter than any two-run comparison on this surface.

### The loading order inverts here

At training the conditioning is installed **after** the checkpoint load, because
the pretrained checkpoint does not carry its key. At evaluation it must be
installed **before**, because now the checkpoint does. Both are correct and they
are opposite; a test pins each.

### Three refusals before it scores anything

- a checkpoint whose `spacing.enabled` is false — there would be no ablation
- a conditioning still at exactly zero — it would compare a thing to itself
- any expert-surface series whose spacing did not resolve — the ablation would
  be partly silent

## Step 8 — read it against the right threshold

`VETO_DELTA = -0.020` on the Expert-58 macro. B52 sits at **0.678247**.

And the standing caution, which has been earned three times this project: at 58
studies the surface cannot resolve small per-target differences. Read the macro,
not the twelve.
