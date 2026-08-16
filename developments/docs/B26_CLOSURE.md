# B26 supervision-repair campaign — closure

> **Status — 2026-08-16:** CLOSED / NOT PROMOTED. **B20 remains the active working model.** No B26.3 training run is authorized from the reused expert surface.

## Completed lineage

```text
B26 raw targeted fill       quality gate failed
B26.1 LLM evidence gate     quality gate failed
B26.2 deterministic gate    fresh manual quality gate passed
B26.2 fixed-E2 training     completed exact frozen B20 surface
B26.2 reused expert result  no macro improvement; Synovitis lower
B26.2 mechanism audit       completed
```

B26.2 added 171 B6-silent Synovitis cells (76 positive, 95 negated) while preserving every existing B6 cell. The trained surface contained 3,120 studies, 17,475 eligible series and 14,294 supervised cells.

## Reused expert result

```text
B20 macro AUC        0.6674066371
B26.2 macro AUC      0.6662972442
delta               -0.0011093928
paired 95% CI       [-0.0156579503, +0.0142502372]
P(B26.2 > B20)       0.4442

Synovitis
B20                  0.8375
B26.2                0.7826
delta               -0.0550
```

The 58-study expert surface is already reused development data. These numbers are post-hoc diagnostic evidence, not independent validation.

## Mechanism audit

The audit shows that the intervention did **not** increase Synovitis' total target-level loss share. Because the B20 loss uses target-balance multipliers, each target remained at approximately one twelfth of normalized loss mass:

```text
normalized Synovitis total loss share
B6       0.0833333337
B26.2    0.0833333316
```

What changed strongly was the **within-target class composition**:

```text
B6 Synovitis
positive cells                 399
negative cells                  17
negative effective-mass frac   7.852%
target multiplier              4.11374

B26.2 Synovitis
positive cells                 475
negative cells                 112
negative effective-mass frac  32.046%
target multiplier              2.58000
```

Thus B26.2 shifted the Synovitis loss from about 92:8 positive:negative effective mass to about 68:32 while leaving total target share essentially fixed.

## Co-occurrence shift

B26.2 also changed the joint-label geometry of Synovitis with related report targets.

### Synovitis × Effusion

```text
phi
B6 supervision          0.5532
B26.2 supervision       0.8479
reused expert           0.4032
```

The B26.2 defined-cell table was:

```text
Syn+ Eff+   375
Syn+ Eff-     2
Syn- Eff+    22
Syn- Eff-    81
```

The reused expert table was:

```text
Syn+ Eff+    22
Syn+ Eff-     5
Syn- Eff+    13
Syn- Eff-    18
```

### Synovitis × Baker's

```text
phi
B6 supervision          0.0627
B26.2 supervision       0.4198
reused expert           0.2060
```

B26.2 therefore made Synovitis more tightly coupled to companion abnormalities on the report-supervision surface than on the reused expert surface.

## Interpretation

The original imbalance flag (399 positive / 17 negated) was real as a **report-supervision count imbalance**, but the B26 campaign shows that such imbalance is not automatically a missing-label defect that should be repaired by filling negative cells.

The evidence is consistent with report missingness being non-random: Synovitis is often mentioned when present, while target-specific negative statements are uncommon. The B26.2 negative pool was dominated by globally normal studies rather than explicit target-specific Synovitis negations. Adding those cells produced many easy, jointly normal negatives and altered cross-target correlations.

This is a mechanism interpretation, not a proof of the causal explanation. It is nevertheless sufficient to reject further outcome-driven B26 loss/weight tuning on the already-reused expert set.

## Decision

```text
B26.2 labels themselves       not declared invalid
B26.2 model                   not promoted
B26.3 loss-mass retuning      not authorized
B26 family                    closed
active working model          B20
```

A B26.3 weight ablation could always be made to explore different class-mass settings, but choosing that setting after observing the reused expert Synovitis decline would be post-hoc outcome-driven tuning with no independent selection surface. It is therefore not a scientifically clean path for promotion.

## General supervision rule learned from B26

For report-derived weak labels, a target should **not** proceed from class-imbalance flag directly to negative-label filling. Future supervision-repair work must first distinguish:

```text
true lack of class support
vs.
mention/silence selection bias in clinical reports
```

A balance flag remains a useful diagnostic, but is no longer sufficient by itself to authorize filling silent cells.

## Next direction

Return to the B20 model family and seek an imaging-side improvement that does not alter weak-label semantics. The preferred next candidate should make one controlled architectural change and retain the frozen B20 supervision, encoder, crop geometry and evaluation governance.
