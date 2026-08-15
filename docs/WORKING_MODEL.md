# Active working model

> **Decision — 2026-08-15:** **B20 remains the active working model.** B23-v1 has now been run and audited, but its formal labeller gate failed on specificity. Formal B24 therefore remains blocked. The separate B24X exploratory pilot produced strong weak-v2 evidence for B23 supervision, but no gold evaluation or promotion is allowed. B24X-Density has completed training and awaits weak-v2 evaluation.

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

B20 remains active for governance reasons, not because the newer exploratory results are uninteresting.

### B21-v1

B21 corrected the crop ordering to:

```text
B20 historical: native MRI -> resize 224 -> crop 90% -> resize 224
B21-v1:         native MRI -> crop 90% -> percentile normalization -> resize 224
```

Weak-v2 favored B21:

```text
B20-v2 control macro AUC        0.7298727911
B21-v1 macro AUC                0.7410090411
raw B21 - control              +0.0111362500
paired 95% CI        [+0.0001624070,+0.0226346590]
P(B21 > control)                0.9758888435
```

But the predeclared reused-gold acceptance comparison did not:

```text
B20 replay macro AUC            0.6674066371
B21 macro AUC                   0.6573196516
B21 - B20 replay               -0.0100869854
paired 95% CI        [-0.0328814731,+0.0117052345]
P(B21 > B20)                    0.1812
```

B21 is closed and not promoted.

### B22

B22 tested whether B21 was simply undertrained. It was not:

```text
Epoch   training loss   expert macro AUC
E1      0.7388751291    0.6135270850
E2      0.6381611442    0.6574269018  <- best
E3      0.6087977977    0.6387456622
E4      0.5890809184    0.6136783995
E5      0.5680555741    0.6282683534
```

The training objective continued to improve after E2 while expert ranking deteriorated. Longer downstream training therefore does not rescue the pre-resize formulation.

## B23-v1 formal status

B23-v1 replaces the B6 regex parser with a local `qwen3:14b` Ollama labeller while keeping the downstream target semantics fixed.

The pilot labeller audit showed a large gain in state-only ranking and coverage:

```text
                         B6                 B23
state-only macro AUC     0.7024597743       0.8125164416
sensitivity              0.9748             0.9855
specificity              0.6061             0.5678
coverage                 0.3606             0.6365
```

Paired state-only AUC difference:

```text
raw B23 - B6             +0.1100566673
paired median            +0.1095402088
paired 95% CI            [+0.0680786389,+0.1531882641]
P(B23 > B6)              1.0000
```

However, the formal B23 rule required specificity to exceed B6. It did not:

```text
B23 specificity          0.5678
B6 specificity           0.6061
formal gate              FAILED
```

Therefore:

```text
B23 formally adopted          no
canonical B23 holdout         not frozen
formal B24                    blocked / not run
```

The audit remains descriptive/post-hoc because the repeatedly reused 58-study expert surface has influenced development.

## B24X exploratory result

B24X was explicitly separated from formal B24. It preserves the failed B23 gate and prohibits gold evaluation and promotion.

Matched surface:

```text
shared studies                         692
possible cells                        8304
B6 usable cells                       3045
B23 usable cells                      5697
B23-only added cells                  2844
B6 cells dropped by B23                192
both committed                        2853
disagreements                           70  (2.5%)
```

Both arms used the same study order, same MRI exposure, same weak-v2-safe frozen B16-v2 encoder, same B20 post-resize crop, same optimizer/scheduler and fixed E2 endpoint.

Frozen 623-study B6 weak-v2 evaluation, with zero train/holdout overlap:

```text
B6 control       0.6148488366  [0.5856757959,0.6451316589]
B23/Qwen         0.7116126450  [0.6785972089,0.7435358854]
raw B23 - B6    +0.0967638083
paired median   +0.0963512743
paired 95% CI   [+0.0612014772,+0.1316174812]
P(B23 > B6)      1.0000
```

This is strong exploratory cross-teacher evidence because B23 wins on a surface labelled by B6. It is still **not expert truth** and therefore cannot replace B20.

Per-target weak-v2 deltas:

```text
Synovitis          +0.3344
PF OA              +0.2804
Lateral Meniscus   +0.2172
ACL                +0.1678
Contusion          +0.1497
Medial Meniscus    +0.0724
MCL                +0.0427
Medial OA          +0.0091
Lateral OA         -0.0137
Effusion           -0.0142
Baker's            -0.0184
Fracture           -0.0663
```

The heterogeneous target pattern means the gain should not be attributed to supervision count alone without an ablation.

## B24X-Density

B24X-Density isolates label density by preserving every B6 committed cell exactly and adding B23 only on cells where B6 is silent.

```text
shared studies                 692
B6 cells preserved            3045
B23-only cells added           2844
final supervised cells         5889
B6 cells dropped                  0
B6 labels overridden              0
```

Training completed at fixed E2:

```text
E1 loss  0.7647414911
E2 loss  0.6197285242
checkpoint  runs/b24x_density/density/b24x_density_model.pt
```

**Current status:** frozen weak-v2 evaluation pending.

The intended comparison is:

```text
B6       = 0.6148488366
Density  = pending
Full B23 = 0.7116126450
```

This will test how much of the B23 gain comes from filling B6-silent cells rather than replacing/dropping existing B6 supervision.

## Current model roles

```text
B17  frozen fixed-epoch reference
B18  frozen full-FOV comparator; nested audit complete
B19  rejected spatial formulation
B20  ACTIVE WORKING MODEL
B21  pre-resize crop candidate; weak-v2 passed, gold acceptance failed; CLOSED
B22  duration audit; E2 best, no rescue; CLOSED
B23-v1  local Qwen report labeller; FORMAL GATE FAILED; NOT ADOPTED
B24 formal  BLOCKED / NOT RUN
B24X  exploratory matched B6-vs-B23 pilot; WEAK-V2 COMPLETE; NO GOLD / NO PROMOTION
B24X-Density  exploratory density ablation; TRAINING COMPLETE; WEAK-V2 PENDING
```

## Current scientific position

The evidence now points to the supervision/development signal as a genuine research lever, but with an important distinction between **useful exploratory signal** and **formal model promotion**.

1. B21/B22 show that more optimization is not the main bottleneck.
2. B23-v1 is not formally acceptable because its specificity gate failed.
3. B24X nevertheless shows that B23 supervision can improve MRI learning strongly on B6's own weak surface.
4. The mechanism of that gain is not yet isolated; B24X-Density is the next diagnostic.
5. No result so far justifies replacing B20 without an independent expert/competition signal.

## Current optimization priority

Do **not** spend the next experiment on:

```text
more epochs
crop-order retry
crop-fraction sweep under the current formulation
another reused-gold epoch search
target-wise B6/B23 model mixing from the B24X weak table
B24X gold acceptance
```

Priority:

```text
1. complete B24X-Density weak-v2 evaluation;
2. determine whether the B24X gain is mainly density or changed B23 semantics;
3. revise B23 only as a new version with new provenance/cache if semantic errors are corrected;
4. require a future B23 version to pass the formal labeller gate before formal B24 resumes;
5. preserve hidden competition evaluation as the independent predictive signal.
```

## Governance

- Keep historical B20 unchanged as the working checkpoint.
- B21 and B22 remain closed.
- B23-v1 formal gate failed; do not reinterpret it as passed.
- No canonical B23 holdout exists.
- Formal B24 has not been run.
- B24X and B24X-Density are exploratory only.
- Do not run formal `b24-accept` on B24X checkpoints.
- Do not use gold to select among B24X variants.
- Do not build target-wise hybrids from the weak-v2 per-target result.
- Weak-v2 measures B6 teacher agreement, not expert truth.
- The 58 expert studies remain a repeatedly reused development/post-hoc surface, not pristine independent validation.
- Hidden competition evaluation remains the independent predictive-performance signal.

## Canonical records

- [`CURRENT_STATUS.md`](CURRENT_STATUS.md)
- [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md)
- [`B20_CROP_ONLY_FOCUS.md`](B20_CROP_ONLY_FOCUS.md)
- [`B21_PRERESIZE_CROP.md`](B21_PRERESIZE_CROP.md)
- [`B21_FULL_ACCEPTANCE.md`](B21_FULL_ACCEPTANCE.md)
- [`B22_DURATION_AUDIT.md`](B22_DURATION_AUDIT.md)
- [`B23_LLM_REPORT_LABELS.md`](B23_LLM_REPORT_LABELS.md)
- [`B24_SUPERVISION_SOURCE.md`](B24_SUPERVISION_SOURCE.md)
- [`B24X_EXPLORATORY_SUPERVISION.md`](B24X_EXPLORATORY_SUPERVISION.md)
- [`VALIDATION.md`](VALIDATION.md)
