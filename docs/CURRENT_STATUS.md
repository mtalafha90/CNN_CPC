# Current project status

**Snapshot:** 2026-08-15  
**Package:** `0.29.0`  
**Active working model:** `B20_crop_only_joint_focus`  
**Canonical B20 checkpoint:** `runs/b20_crop_focus/b20_model.pt`  
**Canonical B20 epoch:** `2`  
**Primary metric:** macro ROC AUC across 12 targets

> **B20 remains the active working model.** B23/B24X/B25X results are development evidence about supervision quality and mechanism. None of them has a valid promotion path.

## Current headline

- **B20 remains active and unchanged.** Canonical reused-expert macro AUC: `0.6671593555` at epoch 2.
- **B23-v1 formal gate FAILED** because specificity `0.5678` did not exceed B6 `0.6061`; no canonical B23 holdout exists and formal B24 remains blocked/not run.
- **B24X pilot completed:** B23/Qwen supervision improved frozen weak-v2 from `0.6148488366` to `0.7116126450` on the matched 692-study pilot.
- **B24X-Density completed:** preserving all B6 cells and adding only B23-silent-cell supervision reached `0.7147994969`, slightly above full B23 `0.7116126450`; the B23-vs-Density interval crossed zero. This showed that almost all of the B24X gain came from supervision recovery rather than replacing B6 decisions.
- **B25X completed on the full 2,497-study weak-v2 training surface** using the ChatGPT-created hybrid cache as an explicitly mixed/unknown-provenance exploratory source.
- B25X frozen weak-v2: `B6=0.6723718048`, `Hybrid=0.7268784872`, `B6+Hybrid-fill=0.7308472686`.
- The B25X macro gain is **overwhelmingly a Synovitis effect**. Excluding Synovitis, the 11-target macro changes are `Hybrid-B6=-0.0028167951` and `Fill-B6=+0.0023982627`.
- Synovitis diagnosis: B6 training had only `13` negatives versus `322` positives; Hybrid-fill added `136` negatives and `66` positives. The Synovitis weak-v2 AUC rose from `0.2370` to `0.9123` for Fill and remained `0.8874–0.9610` under leave-one-negative-out checks.
- The current plan is **not** to move to DINOv2 or soft-dense labels. The next phase will develop the current B20-family working model, using the B25X class-coverage finding as a diagnostic.

## Experiment state

| ID | Purpose | Current result | Status |
|---|---|---|---|
| **B20** | active historical knee-focused model | expert macro AUC `0.6671593555` | **ACTIVE WORKING MODEL** |
| B21 | pre-resize crop correction | weak-v2 improved; expert acceptance failed | closed / not promoted |
| B22 | B21 duration audit E1-E5 | E2 best; longer training did not rescue | closed |
| **B23-v1** | local Qwen report labeller | state-only AUC `0.8125164416`; specificity `0.5678` | **formal gate FAILED** |
| **B24 formal** | matched B6-vs-B23 supervision | not run | **blocked by B23 gate** |
| **B24X** | matched B6-vs-B23 pilot | `0.6148488366 -> 0.7116126450` weak-v2 | exploratory complete |
| **B24X-Density** | B6 preserved + B23-only missing cells | `0.7147994969` weak-v2 | exploratory complete |
| **B25X** | full matched B6 / Hybrid / B6+Hybrid-fill | `0.6723718048 / 0.7268784872 / 0.7308472686` | exploratory complete; mechanism diagnosed |

## B24X-Density final result

```text
B6 control       0.6148488366
Density          0.7147994969
Full B23         0.7116126450

Density - B6       +0.0999506603
Full B23 - B6      +0.0967638083
Full B23 - Density -0.0031868519

B6 -> Density
median             +0.0998800219
95% CI             [+0.0642300469,+0.1348991590]
P(Density > B6)     1.0000

B23 - Density
median             -0.0031277652
95% CI             [-0.0099855349,+0.0034718378]
P(B23 > Density)    0.1799
```

Interpretation: the point-estimate gain is explained almost entirely by **adding supervision where B6 was silent**. There is no evidence that B23 replacements/drops are required.

## B25X full matched hybrid experiment

### Training surface

```text
shared studies              2497
possible cells             29964
B6 usable cells            11248  (37.5%)

Pure Hybrid
usable cells               20001  (66.8%)
added                       9542
dropped                      789
disagreements               1120 / 10459 = 10.7%

B6 + Hybrid-fill
usable cells               20790  (69.4%)
B6 cells preserved         11248
Hybrid-only cells added     9542
B6 cells dropped               0
B6 cells overridden            0
```

All arms used the same 2,497 studies/order, frozen weak-v2-safe B16-v2 encoder, B20 post-resize crop and fixed E2 endpoint. The 623-study weak-v2 holdout and all 58 expert-gold studies were excluded from gradients.

### Training

```text
Control: E1 0.7601064120 | E2 0.6592396402 | 45m50s
Fill:    E1 0.6799390770 | E2 0.5913315904 | 46m43s
Hybrid:  E1 0.6557762888 | E2 0.5718413529 | 46m03s
```

### Frozen weak-v2 result

```text
B6 control          0.6723718048
Pure Hybrid         0.7268784872
B6 + Hybrid-fill    0.7308472686

Hybrid - B6
raw                 +0.0545066824
median              +0.0557034913
95% CI              [+0.0269870416,+0.0750180195]
P(>0)                1.0000

Fill - B6
raw                 +0.0584754637
median              +0.0591551676
95% CI              [+0.0301804537,+0.0814020218]
P(>0)                1.0000

Hybrid - Fill
raw                 -0.0039687813
median              -0.0039491728
95% CI              [-0.0137571379,+0.0058102163]
P(Hybrid > Fill)     0.2037
```

Fill has the best 12-target point estimate while preserving every B6 decision. The Hybrid-vs-Fill interval crosses zero.

## B25X Synovitis diagnosis

Per-target analysis showed that the full macro gain is dominated by Synovitis:

```text
Synovitis AUC
B6       0.2370
Hybrid   0.9221
Fill     0.9123
```

Excluding Synovitis:

```text
11-target macro
B6       0.7119498792
Hybrid   0.7091330840
Fill     0.7143481419

Hybrid - B6   -0.0028167951
Fill - B6     +0.0023982627
```

Training-label audit:

```text
B6 Synovitis                  322 positive / 13 negative
Hybrid-only additions          66 positive / 136 negative
Final Fill                    388 positive / 149 negative
```

Frozen weak-v2 Synovitis has only `77` positives and `4` negatives. Nevertheless, the effect is not controlled by one lucky negative:

```text
leave-one-negative-out AUC range
B6       0.177489 -- 0.259740
Hybrid   0.900433 -- 0.978355
Fill     0.887446 -- 0.961039
```

The current interpretation is therefore narrow: **Hybrid-fill repaired a severe negative-class coverage failure for Synovitis. Across the other eleven targets, the net effect is approximately neutral.**

## Current scientific position

1. More downstream epochs are not the next lever under the current recipe.
2. B23-v1 remains formally rejected by its own specificity gate.
3. B24X/B24X-Density show that recovering B6-silent supervision can be useful, and that replacing B6 decisions is unnecessary on the measured development surface.
4. B25X scales that finding to 2,497 studies but shows that the aggregate gain is highly target-specific, dominated by Synovitis class coverage.
5. B20 remains the active working model. The next phase should improve the current B20-family model rather than switching architecture wholesale.
6. No DINOv2 or soft-dense-label branch is currently planned.

## Governance

```text
B20 remains ACTIVE WORKING MODEL
B23-v1 formal gate FAILED
formal B24 has NOT been run
B24X/B24X-Density/B25X are exploratory only
no B24X/B25X gold acceptance
no B24X/B25X promotion
weak-v2 measures B6 teacher agreement, not expert truth
58-study expert surface is reused/post-hoc, not independent validation
hidden competition evaluation remains the independent predictive signal
```

## Canonical records

- [`WORKING_MODEL.md`](WORKING_MODEL.md) — active working model and governance.
- [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md) — experiment ledger.
- [`B23_LLM_REPORT_LABELS.md`](B23_LLM_REPORT_LABELS.md) — B23 protocol/background.
- [`B24_SUPERVISION_SOURCE.md`](B24_SUPERVISION_SOURCE.md) — formal B24 protocol.
- [`B24X_EXPLORATORY_SUPERVISION.md`](B24X_EXPLORATORY_SUPERVISION.md) — B24X/B24X-Density record.
- [`B25X_HYBRID_SUPERVISION.md`](B25X_HYBRID_SUPERVISION.md) — full hybrid experiment and Synovitis mechanism audit.
- [`VALIDATION.md`](VALIDATION.md) — validation governance.
