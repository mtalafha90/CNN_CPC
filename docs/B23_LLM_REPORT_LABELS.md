# B23 — local-LLM report labels

> **Status — 2026-08-15:** B23-v1 has been run on a pilot and audited. It substantially improves state-only macro AUC and coverage relative to B6, but **fails the predeclared formal gate because specificity is lower than B6**. B23-v1 is not formally adopted, no canonical B23 holdout has been frozen, and formal B24 remains blocked. B20 remains the active working model.

## Purpose

B23 changes the **source of weak supervision**, not the MRI model. It replaces the frozen B6 v1.2.1 regex report parser with a local open-weights LLM while preserving the same 12 target meanings and the same downstream MRI recipe.

The implemented competition path uses:

```text
backend          Ollama
model            qwen3:14b
quantisation     Q4_K_M
decoding         greedy
thinking         disabled
input            Report text only
output           strict structured JSON for all 12 targets
states           positive / negated / uncertain / unmentioned
```

B6 remains frozen as the historical weak-label source for B7-B22.

## Frozen target semantics

The 12 targets continue to mean **abnormality of the named structure/pathology**, not only tear or severe disease. B23 is not allowed to redefine the task to improve its audit.

Examples that remain positive under the frozen semantics include:

```text
ACL: grade 1 sprain is seen with intact fibers.
Mucoid degeneration of the ACL without evidence of tear.
Myxoid degeneration of the posterior horn of the medial meniscus but no definite tear.
```

Hedged findings should be `uncertain`, not positive. Silence is not negation. Generic/degenerative marrow edema should not automatically become traumatic `Contusion`.

## Why B23 was attempted

B6 v1.2.1 is high-sensitivity but low-coverage report supervision. On the full 4,349 non-gold report corpus:

```text
possible cells       52,188
usable B6 cells      14,123
usable fraction       27.1%
active B6 studies      3,120 / 4,349
```

On the reused 58-study expert surface, B6's state-only audit is approximately:

```text
macro AUC          0.7024597743
sensitivity        0.9748
specificity        0.6061
PPV                0.6905
NPV                0.9639
coverage           0.3606
usable cells       251 / 696
```

The hypothesis was that a multilingual local LLM could read more of the report while preserving the frozen pathology semantics.

## B23-v1 pilot export

The completed pilot contained:

```text
structured rows                 1290
training/non-gold rows          1232
gold studies available            58 / 58
gold rows used for training         0
usable non-gold cells           9321 / 14784
pilot cell coverage             63.0%
```

The extraction cache is provenance-checked and resumable. Gold studies are available to the audit but excluded from `training_targets.csv`.

## Formal labeller audit — completed

The labeller audit compares B23-v1 and frozen B6 on the 58 expert studies. This audit is **descriptive/post-hoc, not confirmatory validation**, because aggregate information from this reused expert surface influenced the prompt design. The formal gate nevertheless remains binding.

### Aggregate result

```text
                         B6 v1.2.1          B23-v1 / Qwen
state-only macro AUC     0.7024597743        0.8125164416
sensitivity              0.9748              0.9855
specificity              0.6061              0.5678
PPV                      0.6905              0.6667
NPV                      0.9639              0.9781
coverage                 0.3606              0.6365
usable gold cells        251                 443
```

B23-v1 confusion summary:

```text
TP  204
FP  102
TN  134
FN    3
balanced accuracy  0.77665
```

B6 confusion summary on its usable cells:

```text
TP  116
FP   52
TN   80
FN    3
balanced accuracy  0.79043
```

### Paired state-only macro-AUC difference

```text
raw B23 - B6             +0.1100566673
paired median            +0.1095402088
paired 95% CI            [+0.0680786389,+0.1531882641]
P(B23 > B6)              1.0000
```

The gain in ranking and coverage is large, but that alone was not the frozen adoption rule.

## Predeclared formal gate

B23-v1 was to be adopted only if all of the following held:

```text
state-only macro AUC > B6 baseline
paired 95% CI excludes zero
coverage > B6 coverage
specificity > B6 specificity
```

Observed specificity:

```text
B6 specificity          0.6061
B23 specificity         0.5678
```

Therefore:

```text
formal B23 gate         FAILED
```

Consequences:

```text
B23-v1 formally adopted             no
canonical B23 development split     not frozen
formal B24 training                 blocked
formal B24 gold acceptance          unavailable
B20 replaced                        no
```

`rsna-knee-b23-split` correctly refused to freeze a canonical B23 holdout after the failed gate.

## What the failed gate means

The failure should not be reduced to "B23 is worse." The actual result is more specific:

- B23-v1 reads substantially more report cells.
- It improves state-only macro AUC on the reused expert surface.
- It has slightly higher sensitivity and NPV.
- It has lower specificity and PPV than frozen B6.
- The predeclared formal protocol required B23 to improve specificity as well, so B23-v1 is formally rejected.

The gate must not be relaxed after seeing the result.

## Error interpretation

Review of the audit showed two different phenomena mixed inside B23 false positives.

### Report/gold disagreement under broad target semantics

Some apparent false positives are explicit low-grade findings present in the report but absent in the reused expert label, for example small effusions, low-grade ligament sprains, or meniscal degeneration. These cases should not automatically be "fixed" by teaching the LLM to ignore explicit report abnormalities merely to fit the 58 reused expert studies.

### Genuine B23 semantic/extraction errors

The more credible B23-v1 failure modes include:

```text
generic/degenerative marrow edema -> Contusion overcall
hedged/suspected findings -> positive instead of uncertain
periligamentous fluid with intact MCL -> MCL overcall
```

Qwen's self-reported confidence does not separate TP from FP well enough to justify a post-hoc confidence threshold.

## Why an exploratory B24X pilot was still informative

Although B23-v1 failed the formal gate, it produced much denser supervision. A separate experiment, **B24X**, was therefore created to answer an exploratory question without pretending that the formal gate passed.

B24X:

```text
keeps formal B23 gate recorded as FAILED
uses no canonical B23 holdout
excludes gold and frozen weak-v2 holdout from gradients
uses matched studies and MRI exposure
uses no gold evaluation
allows no model promotion
```

On the matched 692-study pilot:

```text
B6 usable cells                       3045
B23 usable cells                      5697
B23-only added cells                  2844
B6 cells dropped by B23                192
cells both committed                 2853
disagreements                           70  (2.5%)
```

Frozen B6 weak-v2 result:

```text
B6 control       0.6148488366
B23/Qwen         0.7116126450
raw delta       +0.0967638083
paired 95% CI   [+0.0612014772,+0.1316174812]
P(B23 > B6)      1.0000
```

This is strong exploratory evidence that B23 supervision contains useful MRI-learning signal, but weak-v2 measures B6 teacher agreement rather than expert truth. It does **not** reverse the failed B23 gate.

See [`B24X_EXPLORATORY_SUPERVISION.md`](B24X_EXPLORATORY_SUPERVISION.md).

## B23.1 / future-version rule

If B23 is revised, it must be a **new labeller version**, not a hidden retune of B23-v1.

A future version should:

```text
use a new prompt/version identifier
use new provenance/cache keys
retain frozen target semantics
correct genuine semantic errors rather than fit reused gold labels
run a new labeller audit
pass the same formal gate before a development split is frozen
```

Particular semantic corrections worth testing independently include:

```text
Contusion: require traumatic marrow-injury context rather than generic degenerative edema
Hedging: suspected/possible/likely-no -> uncertain where appropriate
Ligaments: adjacent/periligamentous fluid does not imply intrinsic ligament abnormality
```

These corrections must be justified by target semantics and radiology language, not by target-wise optimization against the reused 58-study labels.

## Reproducibility and provenance

The intended B23 export records the local model provenance, including model identifier/revision, quantization, decoding policy, prompt hash and weight/blob identity. Cached extraction is keyed so incompatible prompt/model versions cannot be mixed silently.

The competition path remains local/open-weights so a third party can reproduce the report-label generation without depending on a mutable hosted service.

## Governance

```text
B6 v1.2.1 remains frozen
B23-v1 formal gate FAILED
B23-v1 not adopted
no canonical B23 holdout exists
formal B24 blocked
B24X is exploratory only
B24X does not repair or override the failed B23 gate
no B23-v1 confidence-threshold retuning after audit
no prompt edits under the same B23-v1 provenance
no gold-driven target-semantic changes
58-study audit is descriptive/post-hoc, not independent validation
```

## Related records

- [`CURRENT_STATUS.md`](CURRENT_STATUS.md)
- [`WORKING_MODEL.md`](WORKING_MODEL.md)
- [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md)
- [`B24_SUPERVISION_SOURCE.md`](B24_SUPERVISION_SOURCE.md)
- [`B24X_EXPLORATORY_SUPERVISION.md`](B24X_EXPLORATORY_SUPERVISION.md)
- [`VALIDATION.md`](VALIDATION.md)
