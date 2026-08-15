# B26.2 — deterministic evidence gate

> **Status — 2026-08-16:** IMPLEMENTED / FRESH MANUAL QUALITY GATE PASSED / ELIGIBLE FOR CONTROLLED FIXED-E2 TRAINING. **B20 remains the active working model; B26.2 is not promoted.**

## Why B26.2 exists

B26.1 applied a strict second LLM pass to the 631 exact B20-surface B26 proposals. It retained 281 same-polarity candidates:

```text
raw B26 candidates             631
raw positive                   113
raw negated                    518

B26.1 accepted positive         83
B26.1 accepted negated         198
B26.1 accepted total           281
polarity flips rejected          2
```

If used directly, B26.1 would produce:

```text
B6 Synovitis                  399 positive / 17 negative
B26.1 additions               83 positive / 198 negative
final                         482 positive / 215 negative
```

However, the required fresh B26.1 manual audit did not pass for negative precision:

```text
                    correct    reviewed    observed precision
positive               19         20            95.0%
negated                36         60            60.0%
overall                55         80            68.8%
```

Therefore B26.1 is not approved for training.

## B26.2 policy

B26.2 is a deterministic, precision-first whitelist over the frozen B26.1 output. It does not call an LLM and does not inspect model predictions, weak-v2, or gold outcomes.

It can only **remove** B26.1 proposals:

```text
B26.1 accepted candidate
        |
        +-- positive -> retain only explicit Synovitis or qualifying synovial abnormality
        |
        +-- negated  -> retain only explicit target-specific negation
        |              OR a vetted whole-exam global-normal conclusion
        |
        `-- otherwise -> unmentioned / no added supervision
```

It never creates a label, flips polarity, changes an existing B6 cell, drops an existing B6 cell, or reads weak-v2/gold outcomes.

### Positive whitelist

Accepted only when the B26.1 quoted evidence is verbatim in the original report and contains a direct qualifying concept such as synovitis/sinovitis/sinovit, Reizsynovialitis, synovial thickening, synovial hypertrophy, synovial proliferation, pannus, or frozen multilingual equivalents observed in the reviewed corpus.

Generic synovial findings such as a small synovial recess/outpouching or synovial-fluid leakage are not sufficient.

### Negative whitelist

Accepted when either:

1. the quoted evidence is verbatim and explicitly negates Synovitis or a qualifying synovial abnormality; or
2. the original report contains a vetted unqualified whole-exam conclusion such as `Conclusion: Normal`, `Normal study`, `No significant abnormality identified`, `Normal MR examination of the ... knee`, `Geen afwijkingen aangetoond`, and equivalent frozen forms.

The following remain insufficient by themselves: no/trace effusion, normal bone marrow, normal meniscus/ligament/cartilage/soft tissue, no intra-articular body or injury, normal/non-thickened capsule, or no perimeniscal inflammation.

## Completed B26.2 run

The completed deterministic run used the exact 631-candidate B26.1 surface and accepted 171 cells:

```text
input B26 candidates             631
B26.1 accepted input             281
B26.2 accepted positive           76
B26.2 accepted negated            95
B26.2 accepted total             171
B26.1 calls removed              110
```

Acceptance mechanism:

```text
explicit positive synovial evidence      76
global-normal report conclusion          90
explicit negative synovial evidence       5
```

The resulting exact B20 Synovitis supervision surface would be:

```text
B6 baseline                 399 positive / 17 negative
B26.2 additions              76 positive / 95 negative
final                       475 positive / 112 negative
final usable                587
majority share              80.92%
```

Across all 12 targets the total active supervision becomes:

```text
B6 usable cells             14123
B26.2 added cells             171
final usable cells          14294
```

No B6 cell is dropped or overridden.

With the unchanged B20 weak-supervision weights, the Synovitis effective loss mass is:

```text
positive: 475 x 0.50 = 237.5
negative: 112 x 1.00 = 112.0
```

B26.2 therefore repairs the original 399:17 class-coverage failure without the severe negative overcorrection of raw B26.

## Fresh manual quality audit

A third manual semantic review was performed on the fresh B26.2 accepted-call set after excluding all 160 UIDs from the two prior review sets.

Only 70 fresh rows were available under the requested 60-negative/20-positive design:

```text
fresh reviewed positive       20
fresh reviewed negated        50
fresh reviewed total          70
prior reviewed UIDs excluded 160
```

Under the frozen B26.2 target semantics and whitelist policy, all 70 reviewed calls were supported by the original report:

```text
                    correct    reviewed    observed precision
positive               20         20           100.0%
negated                50         50           100.0%
overall                70         70           100.0%
```

The exact-binomial 95% lower confidence bounds are approximately 83.2% for 20/20 positives, 92.9% for 50/50 negations, and 94.9% for 70/70 overall. These intervals are descriptive quality-control statistics; the review is not independent expert-gold validation.

One positive example described synovial proliferation in association with a Baker cyst. It is retained under the frozen B26.2 semantics because the report explicitly states a qualifying synovial proliferation. This remains a semantic-scope choice, not external expert adjudication.

### Quality decision

```text
B26 raw v1.0 quality gate      FAILED
B26.1 fresh quality gate       FAILED
B26.2 fresh quality gate       PASSED
```

This passage authorizes a **controlled exploratory fixed-E2 training experiment only**. It does not promote B26.2, does not convert the manual audit into independent validation, and does not authorize outcome-driven retuning of the whitelist.

## Implementation and provenance

```text
developments/src/rsna_knee/b26_2_deterministic_gate.py
developments/tests/test_b26_2_deterministic_gate.py
```

The completed audit recorded:

```text
B26.1 candidate SHA-256
35ead84a7ba127d7f844af339b71cf699831dbbda99f1305e58d7307da133522

balance-audit SHA-256
bcc6fadbfb3a0d0b8cdeb1ea3fc116ad1236242d3b699d06d421d237a032f3f2
```

## Next experiment

The next run is a fixed-E2 B20-family training experiment in which the **only supervision change** is the 171 B26.2-approved B6-silent Synovitis cells.

Frozen elements:

```text
training studies             3120
historical B16 encoder       frozen
B20 90% post-resize crop     unchanged
architecture                 unchanged
optimizer                    unchanged
augmentation                 unchanged
scheduler horizon            5 epochs
training endpoint            fixed E2
expert labels in gradients   0
B6 cells dropped             0
B6 cells overridden          0
```

No weak-v2 or reused-gold result was inspected in defining B26.2. Any subsequent reused-gold evaluation remains post-hoc development evidence, and B20 remains active unless a separately stated promotion rule is satisfied.
