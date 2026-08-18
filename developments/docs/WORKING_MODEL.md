# Active working model — decision record, 2026-08-15

> **SUPERSEDED as a status document.** This is the frozen record of the
> decision taken on 2026-08-15 and the evidence available at that time. It is
> preserved unrevised. It is **not** the current position.
>
> For the model the code actually builds and runs, see the top-level
> [`docs/WORKING_MODEL.md`](../../docs/WORKING_MODEL.md).
> For where the project stands, see [`CURRENT_STATUS.md`](CURRENT_STATUS.md).
>
> What changed since: B26 through B34 all ran and none was promoted; the
> top-level interface now targets the B34/B31 architecture as an interface
> choice rather than a promotion; the 58-study expert surface was retired as a
> design surface in favour of the prospective weak splits; and the dataset
> contract audit found that the reports are multilingual and the frozen parser
> reads Latin script only.

> **Decision — 2026-08-15:** **B20 remains the active working model.** B25X has completed as an exploratory supervision experiment and provides a useful diagnosis of a Synovitis class-coverage failure, but it has no gold/promotion path and does not replace B20.

## Active model

```text
model                  B20_crop_only_joint_focus
checkpoint             runs/b20_crop_focus/b20_model.pt
canonical epoch        2
implemented geometry   native MRI -> resize 224 -> center crop 90% -> resize 224
cosine/vignette mask   no
encoder                frozen historical B16 report-aligned encoder
canonical gold score   0.667159355531343
status                 ACTIVE WORKING MODEL
```

Historical B20 is preserved unchanged.

## Why B20 still remains active

B20 remains active because none of the later experiments has produced a valid independent promotion result.

- **B21** passed weak-v2 development but failed reused-gold acceptance.
- **B22** showed that extending training beyond E2 does not rescue the B21 formulation.
- **B23-v1** improved report-label coverage/ranking but failed its frozen specificity gate (`0.5678 < 0.6061`).
- **Formal B24** is therefore blocked/not run.
- **B24X/B24X-Density** are exploratory only and cannot use gold for promotion.
- **B25X** is also exploratory only because its ChatGPT hybrid source has mixed/unknown original LLM provenance.

## B24X-Density mechanism result

B24X-Density preserved all B6 labels and added B23 only where B6 was silent:

```text
B6 control       0.6148488366
Density          0.7147994969
Full B23         0.7116126450

Density - B6       +0.0999506603
Full B23 - B6      +0.0967638083
Full B23 - Density -0.0031868519
```

The paired B23-vs-Density interval crossed zero. The main B24X benefit therefore came from **recovering missing supervision**, not from replacing B6 decisions.

## B25X full hybrid experiment

B25X scaled the supervision-density question to the full leakage-safe 2,497-study weak-v2 training surface.

### Matched surfaces

```text
B6 usable cells                  11248  (37.5%)
Pure Hybrid usable              20001  (66.8%)
B6 + Hybrid-fill usable         20790  (69.4%)
Hybrid-only additions            9542
Fill B6 drops                       0
Fill B6 overrides                   0
```

All arms used the same study order, frozen weak-v2-safe encoder, B20 crop geometry and fixed E2 endpoint. The 623-study weak-v2 holdout and all expert-gold studies were excluded from gradients.

### Frozen weak-v2 result

```text
B6 control          0.6723718048
Pure Hybrid         0.7268784872
B6 + Hybrid-fill    0.7308472686

Hybrid - B6         +0.0545066824
Fill - B6           +0.0584754637
Hybrid - Fill       -0.0039687813
```

Paired intervals:

```text
Hybrid - B6   [+0.0269870416,+0.0750180195]
Fill - B6     [+0.0301804537,+0.0814020218]
Hybrid - Fill [-0.0137571379,+0.0058102163]
```

Fill has the best point estimate while preserving every B6 committed cell. There is no paired evidence that replacing/dropping B6 decisions improves over Fill.

## B25X Synovitis diagnosis

The aggregate B25X gain is overwhelmingly driven by Synovitis:

```text
Synovitis weak-v2 AUC
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

The training-label distribution explains the mechanism:

```text
B6 Synovitis                  322 positive / 13 negative
Hybrid-only additions          66 positive / 136 negative
Final Fill                    388 positive / 149 negative
```

Thus B6 had an extreme negative-class deficit. Hybrid-fill repaired that deficit without changing any existing B6 labels.

The weak-v2 Synovitis holdout contains `77` positives and only `4` negatives, so the absolute AUC remains statistically fragile. However, the effect is not controlled by one single negative case:

```text
leave-one-negative-out AUC range
B6       0.177489 -- 0.259740
Hybrid   0.900433 -- 0.978355
Fill     0.887446 -- 0.961039
```

This makes the mechanism credible as a class-coverage repair while still limiting the breadth of the claim.

## Current model roles

```text
B17  frozen fixed-epoch reference
B18  frozen full-FOV comparator
B19  rejected spatial formulation
B20  ACTIVE WORKING MODEL
B21  weak-v2 passed; gold acceptance failed; CLOSED
B22  duration audit; E2 best; CLOSED
B23-v1  formal labeller gate FAILED
B24 formal  BLOCKED / NOT RUN
B24X  exploratory pilot; COMPLETE; NO GOLD / NO PROMOTION
B24X-Density  exploratory density ablation; COMPLETE; NO GOLD / NO PROMOTION
B25X  exploratory full hybrid/fill experiment; COMPLETE; NO GOLD / NO PROMOTION
```

## Current development direction

The next phase will **develop the existing B20-family working model**, not replace it with DINOv2 or a soft-dense-label branch.

The useful lesson from B25X is specific:

> supervision coverage and within-target class balance can be a bottleneck even when the model architecture is unchanged.

This finding should guide future B20-family experiments, but each new intervention must remain controlled and separately identifiable.

Current priorities:

```text
1. preserve B20 architecture/checkpoint as the reference;
2. use B25X to identify supervision/class-balance weaknesses target by target;
3. test improvements as one-variable B20-family experiments;
4. keep fixed development splits and avoid reused-gold tuning;
5. retain hidden competition evaluation as the independent predictive signal.
```

Not currently planned:

```text
DINOv2 replacement
soft-dense uncertain/unmentioned labels
more-epoch rescue
post-hoc target-wise model mixing from weak-v2 tables
B25X gold acceptance
```

## Governance

- B20 remains the working checkpoint and must stay unchanged as the reference.
- B23-v1 formal gate remains failed.
- Formal B24 remains blocked/not run.
- B24X, B24X-Density and B25X are exploratory only.
- Do not run gold acceptance on B24X/B25X checkpoints.
- Weak-v2 measures agreement with the B6 teacher, not expert truth.
- The 58 expert studies remain a repeatedly reused development/post-hoc surface, not pristine independent validation.
- Hidden competition evaluation remains the independent predictive-performance signal.

## Canonical records

- [`CURRENT_STATUS.md`](CURRENT_STATUS.md)
- [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md)
- [`B20_CROP_ONLY_FOCUS.md`](B20_CROP_ONLY_FOCUS.md)
- [`B21_PRERESIZE_CROP.md`](B21_PRERESIZE_CROP.md)
- [`B22_DURATION_AUDIT.md`](B22_DURATION_AUDIT.md)
- [`B23_LLM_REPORT_LABELS.md`](B23_LLM_REPORT_LABELS.md)
- [`B24_SUPERVISION_SOURCE.md`](B24_SUPERVISION_SOURCE.md)
- [`B24X_EXPLORATORY_SUPERVISION.md`](B24X_EXPLORATORY_SUPERVISION.md)
- [`B25X_HYBRID_SUPERVISION.md`](B25X_HYBRID_SUPERVISION.md)
- [`VALIDATION.md`](VALIDATION.md)
