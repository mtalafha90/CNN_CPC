# B33 reused 58-study expert development result

> **Status — 2026-08-16:** B33 is a successful simplification of B29 on the reused expert development surface, but it is **not independently validated and not promoted**. B20 remains the active reference. B31 remains the highest reused-58 development candidate. No B33.1 or target-wise blend is permitted from this result.

## Frozen comparison

The evaluator compared the exact frozen checkpoints for B20, B29, B31 and B33 on the same 58-study reused expert development surface. This surface is not independent: B20 was historically selected on it and B29/B31 had already been inspected on it before B33 evaluation.

```text
B20 macro AUC  0.6674066371
B29 macro AUC  0.6768879224
B31 macro AUC  0.6822797439
B33 macro AUC  0.6764460785
```

Raw differences:

```text
B33 - B20  +0.0090394414
B33 - B29  -0.0004418439
B33 - B31  -0.0058336654
```

Paired bootstrap results (5,000 valid replicates):

```text
B33 vs B20
median delta   +0.0091897746
95% CI         [-0.0206538625, +0.0375563121]
P(B33 > B20)   0.7372

B33 vs B29
median delta   -0.0004963547
95% CI         [-0.0233115476, +0.0220359251]
P(B33 > B29)   0.4800

B33 vs B31
median delta   -0.0054013714
95% CI         [-0.0281115176, +0.0144539599]
P(B33 > B31)   0.2990
```

## Interpretation

B33 removes B29's learned complementary query and uses an exact uniform mean of the 16 slice tokens behind one zero-init 768-D gate. Its macro AUC is only 0.000442 below B29, with a paired probability of 0.48 that B33 is better. On this reused surface, B33 and B29 are therefore practically indistinguishable at the available resolution.

This supports the mechanistic interpretation that most of B29's observed development gain may come from adding a second broad global series representation rather than from learned slice selection. B33 does this with half of B29's new parameters (768 vs 1,536).

B31 remains numerically highest on the reused 58 studies, but its prospective attention audit showed almost no actual redistribution of the complementary weights. The extra B31 gain therefore remains mechanistically uncertain and is not independent validation.

## Per-target caution

B33 improves some targets and worsens others. These target-wise outcomes must not be used to construct target-specific switches or blends after inspection. In particular, no B33.1, target-selective B31/B33 ensemble, or gate/mean retuning is allowed from this reused result.

## Governance decision

The 58-study expert surface has now been repeatedly used for architecture decisions. Further architecture design should no longer optimize against that surface. The next step is a prospectively frozen weak-label validation framework using an untouched 20% StudyInstanceUID partition of the 3,120 active B6 training studies. B20, B31 and B33 are to be retrained as matched fixed-E2 controls on the remaining 80% before any B34 design is evaluated.
