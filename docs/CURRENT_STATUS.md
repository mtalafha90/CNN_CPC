# Current project status

**Snapshot:** 2026-08-15  
**Package:** `0.29.0`  
**Active working model:** `B20_crop_only_joint_focus`  
**Canonical B20 checkpoint:** `runs/b20_crop_focus/b20_model.pt`  
**Canonical B20 epoch:** `2`  
**Primary metric:** macro ROC AUC across 12 targets

> **B20 remains the active working model.** The new B23/B24X results are scientifically important, but they are exploratory and do not promote or replace B20.

## Current headline

- **B20 remains active and unchanged.** Canonical expert macro AUC: `0.6671593555` at epoch 2.
- **B23-v1 has now been run on a pilot and audited.** It substantially improved state-only macro AUC and coverage relative to B6, but **failed its frozen formal gate because specificity was lower than B6**.
- Because the B23 formal gate failed, **no canonical B23 development holdout was frozen** and **formal B24 is blocked/not run**.
- A separate **B24X exploratory matched pilot** was therefore run with no gold evaluation and no promotion path.
- On the frozen 623-study B6 weak-v2 holdout, the B23/Qwen-supervised B24X arm strongly outperformed the B6-supervised matched control: `0.7116126450` versus `0.6148488366`, paired delta `+0.0967638083`, 95% CI `[+0.0612014772,+0.1316174812]`, `P(B23>B6)=1.0000`.
- This is **teacher-agreement evidence, not expert truth**. It is especially interesting because the evaluation surface is labelled by B6 and therefore structurally favors the B6-supervised control.
- A **B24X-Density** ablation has also completed training. It preserves all 3,045 B6 cells and adds only the 2,844 B23 cells where B6 was silent, for 5,889 supervised cells total, with zero B6 overrides and zero B6 drops. **Its frozen weak-v2 evaluation is pending.**

## Experiment state

| ID | Purpose | Current result | Status |
|---|---|---|---|
| **B20** | active historical knee-focused model | expert macro AUC `0.6671593555` | **ACTIVE WORKING MODEL** |
| B21 | pre-resize crop correction | weak-v2 improved; expert acceptance failed | closed / not promoted |
| B22 | B21 duration audit E1-E5 | E2 best; longer training did not rescue | closed |
| **B23-v1** | local LLM report labeller | state-only AUC `0.8125164416`; coverage `0.6365`; specificity `0.5678` | **formal gate FAILED** |
| **B24 formal** | matched B6-vs-B23 supervision experiment | not run | **blocked by failed B23 gate** |
| **B24X** | exploratory matched B6-vs-B23 pilot | weak-v2 `0.6148488366 -> 0.7116126450` | completed exploratory evidence; no gold/no promotion |
| **B24X-Density** | isolate added-label density from changed B23 decisions | trained on 5,889 cells | training complete; weak-v2 evaluation pending |

## B23-v1 pilot and formal labeller audit

The pilot export contained:

```text
structured rows                 1290
training/non-gold rows          1232
gold studies available            58 / 58
gold rows used for training         0
usable pilot cells              9321 / 14784 = 63.0%
```

The 58-study labeller audit was descriptive/post-hoc because the expert surface has already influenced development. It therefore does not restore independence, but it does enforce the predeclared B23 gate.

```text
                         B6 v1.2.1          B23/Qwen
state-only macro AUC     0.7024597743        0.8125164416
sensitivity              0.9748              0.9855
specificity              0.6061              0.5678
PPV                      0.6905              0.6667
NPV                      0.9639              0.9781
coverage                 0.3606              0.6365
usable gold cells        251                 443
```

Paired state-only macro-AUC difference:

```text
raw B23 - B6             +0.1100566673
paired median            +0.1095402088
paired 95% CI            [+0.0680786389,+0.1531882641]
P(B23 > B6)              1.0000
```

The formal gate nevertheless failed because:

```text
B23 specificity          0.5678
required > B6            0.6061
```

Consequences:

```text
B23 adopted formally                  no
canonical B23 holdout frozen          no
formal B24 allowed                    no
B20 replaced                          no
```

## B24X exploratory matched pilot

B24X was created only to answer a narrower exploratory question after the formal B23 gate failed: **does the denser B23 report supervision produce a better MRI learner when downstream exposure is matched?**

The two arms used the same 692 studies in exactly the same order, the same frozen weak-v2-safe B16-v2 encoder, the same 90% post-resize crop, the same architecture, optimizer, scheduler horizon, augmentations and fixed epoch-2 endpoint. Gold studies and the frozen weak-v2 holdout were excluded from gradients.

Matched supervision surface:

```text
shared studies                         692
possible cells                        8304
B6 usable cells                       3045  (36.7%)
B23 usable cells                      5697  (68.6%)
B23-only added cells                  2844
B6 cells dropped by B23                192
cells where both committed           2853
disagreements where both committed     70  (2.5%)
```

Training completed with exact coverage:

```text
B6 control
E1 loss 0.8581187165
E2 loss 0.7132374823

B23/Qwen candidate
E1 loss 0.7599072829
E2 loss 0.6096711156
```

The loss values are not directly comparable because the supervision masks differ; frozen holdout AUC is the meaningful comparison.

### Frozen weak-v2 evaluation

Leakage check:

```text
B24X training studies        692
weak-v2 holdout studies      623
train/holdout overlap          0
```

Result:

```text
B6 control macro AUC       0.6148488366  [0.5856757959,0.6451316589]
B23/Qwen macro AUC         0.7116126450  [0.6785972089,0.7435358854]
raw B23 - B6              +0.0967638083
paired median             +0.0963512743
paired 95% CI             [+0.0612014772,+0.1316174812]
P(B23 > B6)                1.0000
valid bootstrap            4913 / 5000
```

Interpretation: this is strong **exploratory cross-teacher evidence** that richer B23 supervision improved MRI learning on this matched pilot. However, weak-v2 is labelled by B6, not by experts. It is a development/teacher-agreement surface and cannot promote a model.

### Per-target weak-v2 AUC

| Target | B6 | B23 | Delta |
|---|---:|---:|---:|
| Synovitis | 0.5292 | 0.8636 | +0.3344 |
| PF OA | 0.4492 | 0.7297 | +0.2804 |
| Lateral Meniscus | 0.4658 | 0.6830 | +0.2172 |
| ACL | 0.4840 | 0.6519 | +0.1678 |
| Contusion | 0.4596 | 0.6093 | +0.1497 |
| Medial Meniscus | 0.6399 | 0.7123 | +0.0724 |
| MCL | 0.6182 | 0.6610 | +0.0427 |
| Medial OA | 0.7170 | 0.7260 | +0.0091 |
| Lateral OA | 0.7082 | 0.6946 | -0.0137 |
| Effusion | 0.7736 | 0.7594 | -0.0142 |
| Baker's | 0.7829 | 0.7645 | -0.0184 |
| Fracture | 0.7504 | 0.6842 | -0.0663 |

The target pattern shows that simple supervision count alone cannot explain every effect. For example, Fracture gained many B23 labels but lost weak-v2 AUC. This motivates the density-only ablation rather than target-wise tuning.

## B24X-Density ablation

The density arm tests only the contribution of **filling B6-silent cells**:

```text
if B6 is committed:   keep B6 target and weight exactly
if B6 is silent:      use B23 target/weight if B23 is committed
otherwise:            remain unsupervised
```

Frozen surface:

```text
shared studies                 692
B6 cells preserved            3045
B23-only cells added           2844
final supervised cells         5889
B6 cells dropped                  0
B6 labels overridden              0
```

Training completed with exact coverage:

```text
E1 loss 0.7647414911
E2 loss 0.6197285242
checkpoint  runs/b24x_density/density/b24x_density_model.pt
```

**Current status:** density training is complete; frozen weak-v2 evaluation is the next step. No gold evaluation is permitted for this exploratory arm.

## Scientific position

The campaign now supports four distinct conclusions:

1. **More downstream epochs are not the answer.** B22 showed that weak-training loss can continue falling while expert ranking worsens.
2. **B23-v1 is not formally admissible under its own predeclared gate.** Its specificity is lower than B6, so formal B24 remains blocked.
3. **Nevertheless, B23 supervision contains useful learning signal.** In the exploratory matched B24X pilot it improved B6-teacher agreement by almost `0.10` macro AUC despite the evaluation surface favoring B6.
4. **We still do not know whether the gain is mainly label density or changed semantics.** B24X-Density is designed to answer exactly that question without using gold.

## Governance

```text
B20 remains ACTIVE WORKING MODEL
B23-v1 formal gate FAILED
no B23 formal development split exists
formal B24 has NOT been run
B24X is exploratory only
B24X gold acceptance is PROHIBITED
B24X model promotion is PROHIBITED
B24X-Density gold evaluation is PROHIBITED
weak-v2 measures B6 teacher agreement, not expert truth
58-study expert surface is reused/post-hoc, not independent validation
hidden competition evaluation remains the independent predictive signal
no target-wise model mixing from the B24X per-target table
```

## Immediate next step

Evaluate the trained B24X-Density checkpoint on the same frozen 623-study weak-v2 holdout while reusing the already saved B6 and full-B23 predictions. Compare:

```text
B6       = 0.6148488366
Density  = pending
Full B23 = 0.7116126450
```

This will estimate how much of the full B23 point-estimate gain is captured by added supervision density alone.

## Canonical records

- [`WORKING_MODEL.md`](WORKING_MODEL.md) — active working model and governance.
- [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md) — experiment ledger.
- [`B23_LLM_REPORT_LABELS.md`](B23_LLM_REPORT_LABELS.md) — B23 protocol/background.
- [`B24_SUPERVISION_SOURCE.md`](B24_SUPERVISION_SOURCE.md) — formal B24 protocol.
- [`B24X_EXPLORATORY_SUPERVISION.md`](B24X_EXPLORATORY_SUPERVISION.md) — exploratory B24X/B24X-Density record.
- [`VALIDATION.md`](VALIDATION.md) — validation governance.
