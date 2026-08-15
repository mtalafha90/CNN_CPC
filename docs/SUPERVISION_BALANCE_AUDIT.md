# Supervision balance audit — full frozen B6 surface

> **Status — 2026-08-15:** COMPLETE. This is a training-label-only diagnostic performed after the B25X mechanism analysis. **B20 remains the active working model.** No model was trained, no prediction was read, no expert-gold label was used, and no weak-v2 score was consulted by the audit itself.

## Purpose

B25X showed that its 12-target macro improvement was overwhelmingly driven by Synovitis and that the matched 2,497-study B6 training surface contained only `13` negative Synovitis cells versus `322` positive cells. The repository therefore added a general supervision-balance audit that operates on training-label counts only and applies the same rule to every target.

The audit is intended to distinguish a **structural supervision-coverage problem** from outcome-driven target selection. It does not hard-code Synovitis: whichever target violates the frozen class-balance rule is flagged.

## Audit rule

The current declared thresholds are:

```text
maximum majority-class share     90%
minimum minority-class cells      30
```

A target is flagged when either:

```text
majority_share >= 0.90
OR
minority_cells < 30
```

Important chronology: these thresholds were introduced **after** the B25X Synovitis diagnosis. They therefore do not retroactively make B25X prospective. They are frozen from this point forward as a general training-label policy for future B20-family development unless a separately versioned protocol changes them before looking at new model outcomes.

## Command

```bash
python -m rsna_knee.supervision_balance \
  --config configs/b24_supervision.yaml \
  --data-root "$DATA_ROOT" \
  --export-root "$B6_ROOT" \
  --labeller b6 \
  --out runs/supervision_balance/b6_balance.json
```

Implementation:

```text
src/rsna_knee/supervision_balance.py
```

Tests:

```text
tests/test_supervision_balance.py
```

## Full frozen B6 result

```text
Supervision balance audit (b6)
  threshold 90% one class | minimum 30 minority cells

  target              usable     pos     neg   major  flag
  Synovitis              416     399      17  95.9%  NEEDS FILL
  MCL                   1360     271    1089  80.1%
  Fracture               755     203     552  73.1%
  Lateral Meniscus      1630     448    1182  72.5%
  Medial Meniscus       1662    1126     536  67.7%
  ACL                   1661     572    1089  65.6%
  PF OA                 1054     682     372  64.7%
  Effusion              2095    1338     757  63.9%
  Medial OA              818     484     334  59.2%
  Contusion              855     389     466  54.5%
  Baker's               1033     557     476  53.9%
  Lateral OA             784     402     382  51.3%

  targets needing fill: 1 of 12
    Synovitis
```

Only **Synovitis** fails the frozen rule. The next-highest majority share is MCL at `80.1%`, well below the `90%` threshold, and all other targets also have at least 30 minority-class cells.

## Relationship to the B25X matched surface

The full frozen B6 audit independently reproduces the same class-coverage pathology seen inside B25X:

```text
surface                         positive   negative   majority share
B25X matched 2,497 studies          322         13       96.1%
Full frozen B6 supervision          399         17       95.9%
```

The close agreement shows that the severe Synovitis imbalance is not an artifact of the B25X matched subset. It is present in the broader frozen B6 supervision source.

## Scientific interpretation

This audit supports a narrower and more defensible conclusion than selecting a target from the weak-v2 performance table:

> The frozen B6 supervision source has one target with severe minority-class scarcity under the declared general balance rule: Synovitis.

The audit itself does not establish that adding labels will improve expert performance, nor does it promote any model. It identifies a training-surface defect before future training under the now-frozen rule.

The wording **severely class-imbalanced / insufficient minority-class support** is preferred over claiming that the target is mathematically "unlearnable." The earlier B25X result showed a practical failure under the existing recipe, not an impossibility theorem for binary classification.

## Development implication

The next controlled B20-family experiment may use the audit output rather than a hard-coded target name:

```text
for every target:
    if the frozen balance audit passes:
        preserve the existing B6 supervision exactly

    if the frozen balance audit fails:
        preserve every existing B6 committed cell
        allow fill supervision only on B6-silent cells
```

With the current full-B6 audit, this rule selects only:

```text
Synovitis
```

A future implementation should consume the audit result (`targets_needing_fill`) and must not contain a target-specific rule such as `if target == "Synovitis"`.

This preserves the lesson of B24X/B25X—**fill missing supervision rather than replace B6 decisions**—while avoiding broad Hybrid additions to targets that do not show the same training-label defect.

## Governance

```text
B20 active working model                 unchanged
balance audit uses model predictions     no
balance audit uses weak-v2 outcomes      no
balance audit uses expert gold           no
balance thresholds tuned prospectively   no; defined after B25X diagnosis
balance thresholds frozen going forward  yes
current targets flagged                  Synovitis only
model promotion from this audit           prohibited
```

The result is a development diagnostic, not an evaluation result. Hidden competition evaluation remains the independent predictive signal, and the repeatedly reused 58-study expert set remains a post-hoc/development surface.

## Related records

- [`CURRENT_STATUS.md`](CURRENT_STATUS.md)
- [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md)
- [`WORKING_MODEL.md`](WORKING_MODEL.md)
- [`B24X_EXPLORATORY_SUPERVISION.md`](B24X_EXPLORATORY_SUPERVISION.md)
- [`B25X_HYBRID_SUPERVISION.md`](B25X_HYBRID_SUPERVISION.md)
- [`BEST_MODEL_REFERENCES.md`](BEST_MODEL_REFERENCES.md)
