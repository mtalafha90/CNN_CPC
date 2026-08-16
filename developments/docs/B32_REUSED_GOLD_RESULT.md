# B32 reused-expert development result

> **Decision:** B32 is **NOT PROMOTED**. The weighted-dispersion formulation is closed. B20 remains the active reference; B31 remains the leading frozen development candidate.

## Surface and governance

The comparison used the same 58-study expert development surface used repeatedly in the B20-family development history. It is **not independent validation**. B20 was historically selected on this surface, and B29/B31 had already been inspected before B32 was evaluated. B32 itself was frozen at fixed E2 before this comparison.

Weak-v2 was not used because B32 trains on all 3,120 historical B20 weak-supervision studies, so that historical partial-surface partition is not a holdout.

## Macro AUC

```text
B20   0.6674066371
B29   0.6768879224
B31   0.6822797439
B32   0.6686993213
```

Raw differences:

```text
B32 - B20   +0.0012926842
B32 - B29   -0.0081886011
B32 - B31   -0.0135804226
```

Paired bootstrap, 5,000 valid replicates:

```text
B32 vs B20
median delta   +0.0011332456
95% CI         [-0.0317855361, +0.0345607918]
P(B32>B20)      0.5296

B32 vs B29
median delta   -0.0081690205
95% CI         [-0.0357886251, +0.0201083846]
P(B32>B29)      0.2658

B32 vs B31
median delta   -0.0131666526
95% CI         [-0.0373309378, +0.0066109897]
P(B32>B31)      0.0946
```

## Per-target pattern

B32 improved strongly on some targets, including Baker's cyst and Effusion, but lost materially on ACL, MCL, Lateral Meniscus, Lateral OA and Fracture. These target-wise outcomes are descriptive only and must not be used to construct a selective B32.1, per-target routing rule, or blend.

## Mechanism interpretation

The prospective B32 mechanism audit showed that the dispersion branch was active and non-redundant with the B29 mean-like residual. At E2 the mean-dispersion residual cosine was approximately 0.074 and the dispersion residual was about 0.39% of the B20 series-token norm on average. Nevertheless, this new second-order information did not improve macro AUC consistently.

The conclusion is therefore narrow:

> A non-redundant weighted feature-dispersion summary is **not sufficient** to improve this B20/B29 family under the frozen fixed-E2 recipe.

This does not establish that all second-order statistics are useless. It only closes this exact same-weight weighted-standard-deviation formulation.

## Frozen decision

```text
B20   active reference
B29   frozen promising candidate
B31   frozen leading development candidate
B32   NOT PROMOTED — formulation closed
B32.1 do not create from this reused outcome
```

Do not tune the dispersion statistic, variance epsilon, gates, endpoint, target-specific behavior, or blending from this result.
