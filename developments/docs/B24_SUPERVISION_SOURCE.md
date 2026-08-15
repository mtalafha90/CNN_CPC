# B24 — formal supervision-source experiment

> **Status — 2026-08-15:** **FORMAL B24 HAS NOT BEEN RUN.** B23-v1 failed its predeclared formal labeller gate because specificity (`0.5678`) was below frozen B6 (`0.6061`). No canonical B23 holdout was frozen, so formal B24 is blocked. B20 remains the active working model.
>
> The completed **B24X** work is a separate exploratory pilot with no gold evaluation and no promotion path. It must not be confused with formal B24. See [`B24X_EXPLORATORY_SUPERVISION.md`](B24X_EXPLORATORY_SUPERVISION.md).

## Formal question

B24 is the experiment B23 was designed to enable:

> If a report labeller first passes its own formal quality gate, does replacing frozen B6 regex supervision with that labeller improve the downstream MRI model when everything else is held fixed?

The formal arms are:

```text
b6_control      B6 v1.2.1 regex supervision
b23_candidate   a formally accepted B23-version local-LLM supervision
```

B24 changes the labels and **nothing else**.

## Why formal B24 is currently blocked

B23-v1 audit:

```text
                         B6                 B23-v1
state-only macro AUC     0.7024597743       0.8125164416
coverage                 0.3606             0.6365
specificity              0.6061             0.5678
```

The frozen B23 gate required all conditions to pass, including:

```text
B23 specificity > B6 specificity
```

That condition failed. Therefore:

```text
formal B23 gate                  FAILED
canonical B23 holdout            not frozen
formal B24 control training      not run
formal B24 candidate training    not run
formal B24 cross-labeller eval   not run
formal B24 gold acceptance       not consumed
```

This is the current formal state even though the separate B24X exploratory pilot produced favorable weak-v2 evidence.

## Frozen formal recipe

If a future B23 version passes its gate and a valid holdout is frozen prospectively, formal B24 uses the same matched downstream recipe for both arms:

```text
encoder                  frozen weak-v2-safe B16-v2
encoder LR               0
spatial geometry         B20 post-resize centered 90% crop
input resolution         224
batch size               2
head LR                  1e-4
scheduler horizon        5 epochs
training endpoint        fixed E2
TTA                      [-1,0,1]
seed                     2026
checkpoint selection     none
```

`require_b24_contract` is intended to refuse protocol drift.

## Matched studies, not matched cells

Formal B24 uses the intersection of the two accepted supervision sources' active-study sets after excluding all frozen holdouts and all gold studies.

Both arms therefore see:

```text
same studies
same study order
same MRI series
same batch count
same encoder initialization
same augmentation policy
same optimizer trajectory
same fixed E2 endpoint
```

What differs is which target cells inside those studies carry supervision and what state each source assigns.

This is the single-variable guarantee.

## Formal cross-labeller development evidence

A supervision-generated holdout favors the labeller that generated it. Formal B24 therefore evaluates both arms on both frozen weak surfaces:

| Surface | Labels | Structural advantage |
|---|---|---|
| weak-v2 | B6 | B6 control |
| future B23 holdout | accepted B23 version | B23 candidate |

The informative asymmetry is whether the B23-supervised model also wins on **B6's own weak-v2 surface**.

Even then, neither weak surface is expert truth. B15 and B21 already showed that weak-surface gains need not transfer to expert-gold ranking.

## Formal decision rule

Only after the labeller gate passes, the B23 holdout is frozen, both matched arms are trained, and the cross-labeller development evidence is frozen may formal B24 consume its one predeclared expert-surface look.

Comparator:

```text
canonical B20
macro AUC = 0.667159355531343
```

Formal promotion rule:

```text
paired median(B24 - B20) > 0
AND
P(B24 > B20) >= 0.95
```

A B20 replay sanity check must first reproduce the canonical score within the frozen tolerance.

The acceptance artifact is one-look only. Re-running or adjusting the model after seeing that result would invalidate the protocol.

## Formal gates

```text
B23 labeller gate passed          required
B23 frozen holdout exists         required
gold excluded from gradients      required
weak-v2 excluded from gradients   required
B23 holdout excluded from gradients required
same training study order         required
same encoder initialization       required
fixed E2                          required
one expert-surface look           required
```

As of 2026-08-15 the first two conditions are not satisfied for B23-v1, so formal B24 cannot proceed.

## API compatibility fixes discovered during B24X

The exploratory execution exposed stale API calls in the previously unexecuted formal B24 code. These were compatibility bugs, not protocol changes. The GitHub code has been updated to the current live APIs:

```text
make_b7_dataset_config(config, root, train=True/False, tta_offsets=...)
b12_1_model_spec(config, normalize_input=True)
RuntimeBudget(max_hours=..., reserve_minutes=...)
batch["target"] / batch["weight"]
remaining_work_seconds instead of removed exhausted()
```

Equivalent dataset API fixes were also applied to the formal B24 weak evaluator and gold evaluator.

These changes do not make B23-v1 eligible for formal B24; the failed labeller gate remains binding.

## B24X is not formal B24

After the B23-v1 gate failed, an explicitly exploratory pilot was run to investigate whether the denser supervision nevertheless contains useful MRI-learning signal.

B24X differs procedurally from formal B24 because:

```text
B23-v1 gate remains failed
no B23 formal holdout exists
pilot uses only the available B23-labelled subset
no B23-side formal holdout evaluation exists
no gold evaluation is allowed
no promotion is allowed
checkpoints are retagged exploratory
```

B24X weak-v2 result:

```text
B6 control       0.6148488366
B23/Qwen         0.7116126450
raw delta       +0.0967638083
paired 95% CI   [+0.0612014772,+0.1316174812]
P(B23 > B6)      1.0000
```

This result is scientifically interesting but does not satisfy or replace the formal B24 path.

See [`B24X_EXPLORATORY_SUPERVISION.md`](B24X_EXPLORATORY_SUPERVISION.md).

## What must happen before formal B24 can resume

A future B23 version must:

1. be versioned separately from B23-v1;
2. use new prompt/model provenance and incompatible cache keys where appropriate;
3. preserve the frozen 12 target semantics;
4. run a new labeller audit;
5. pass the same formal gate, including specificity;
6. freeze a prospective B23 development holdout only after passing;
7. then run the matched formal B24 arms without consulting gold.

Only then does the one-look B24 gold acceptance become available.

## Explicitly prohibited

```text
pretending B23-v1 passed its gate
using B24X checkpoints with rsna-knee-b24-accept
running a formal gold look on B24X/Density
relaxing the B23 specificity gate after seeing the audit
freezing a canonical B23-v1 holdout after the failed gate
using B24X per-target results to build a target-wise hybrid
calling weak-v2 expert truth
changing more than the supervision source in formal matched arms
```

## Related records

- [`CURRENT_STATUS.md`](CURRENT_STATUS.md)
- [`WORKING_MODEL.md`](WORKING_MODEL.md)
- [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md)
- [`B23_LLM_REPORT_LABELS.md`](B23_LLM_REPORT_LABELS.md)
- [`B24X_EXPLORATORY_SUPERVISION.md`](B24X_EXPLORATORY_SUPERVISION.md)
- [`VALIDATION.md`](VALIDATION.md)
