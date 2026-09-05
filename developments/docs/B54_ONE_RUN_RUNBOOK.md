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

Expect 1,847 passed, 1 skipped.

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
  --series-policy <the series policy B52 used> \
  --base-checkpoint <the Phase-9 llm_fill checkpoint B52 used> \
  --domain-split <the split B52 used> \
  --spacing-geometry-csv runs/slice_geometry_scan/series_geometry.csv \
  --out-root runs/085_B54/train
```

Recover the three unchanged paths from B52's own audit:

```bash
python -c "
import json; a=json.load(open('runs/086_Experiment_B52_competition_full_finetune/training_audit.json'))
print({k:a.get(k) for k in ('series_policy','base_checkpoint','domain_split')})"
```

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

```python
set_spacing_enabled(model, True)    # the arm
set_spacing_enabled(model, False)   # its own control
```

Run the Expert-58 audit on both. The difference is the spacing effect, free of
any second training run.

## Step 8 — read it against the right threshold

`VETO_DELTA = -0.020` on the Expert-58 macro. B52 sits at **0.678247**.

And the standing caution, which has been earned three times this project: at 58
studies the surface cannot resolve small per-target differences. Read the macro,
not the twelve.
