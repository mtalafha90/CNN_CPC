# Completed PV2 B34 mechanism result

> **Result recorded after the frozen PV2 protocol was completed.** This document does not change the predeclared split, comparisons, equivalence margin, or decision rule in `PROSPECTIVE_WEAK_V2_B34.md`.

## Governance and evaluation surface

PV2 used the frozen nested split:

```text
PV2 split SHA256
b53331ce314b2d2ccc68aea1737427c01bd0d916997e78fbefe88fec5cc95855

parent PV1 split SHA256
a0032307abb1ab99724eb39fac25332ce131c575f64d823083bb37f5ec20d1e6

training studies per control   1997
validation studies              499
validation MRI series          2775
locked original PV1 validation  624
```

The locked 624-study PV1 validation partition was not used. Expert labels were not read. PV2 remains an internal weak-label mechanism surface with historical downstream exposure; it is not independent clinical validation and does not promote a model to active status.

## Predeclared models

```text
PV2-B29  learned complementary query; no local-context scaffold
PV2-B31  local context active during training and retained at inference
PV2-B34  same train-time local-context scaffold; exact context bypass at inference
```

Primary metric: macro per-target B6-weighted soft-label BCE, lower is better.

Secondary metric: macro ROC AUC over B6 positive/negated states where both classes are defined.

## Global results

| Model | Macro weighted soft BCE ↓ | Macro AUC ↑ |
|---|---:|---:|
| B29 | 0.5992237910 | 0.7471635770 |
| B31 | **0.5909689396** | **0.7528002817** |
| B34 | **0.5909695511** | **0.7527943588** |

The B31 and B34 point estimates are essentially identical, whereas both improve over the matched B29 no-scaffold control.

## Primary training-scaffold test: B34 − B29

Difference is candidate macro weighted BCE minus reference macro weighted BCE, so negative favors B34.

```text
median difference             -0.0083070252
95% CI                        [-0.0125729456, -0.0039875203]
P(B34 better)                  0.9998
bootstrap replicates           5000
```

The entire predeclared 95% interval is below zero. Therefore:

**training-scaffold benefit supported.**

## Inference-bypass replication: B34 − B31

The absolute equivalence margin was frozen before evaluation at ±0.001 macro weighted BCE.

```text
median difference             +0.0000005969
95% CI                        [-0.0000031323, +0.0000046725]
P(B34 better)                  0.3768
bootstrap replicates           5000
predeclared equivalence band   [-0.001, +0.001]
```

The full confidence interval lies far inside the predeclared equivalence band. Therefore:

**B34 and B31 are equivalent at the predeclared metric resolution, and retaining the learned local-context branch at inference is unnecessary in this matched experiment.**

## Reference context-training comparison: B31 − B29

```text
median difference             -0.0082841699
95% CI                        [-0.0126808867, -0.0038284057]
P(B31 better)                  1.0000
bootstrap replicates           5000
```

The B31-vs-B29 advantage therefore replicates on PV2.

## B34 scaffold state at evaluation

B34's evaluation contract was verified directly:

```text
training_context_active        false
eval_context_exact_bypass      true
inference_context_parameters   0
```

The learned train-time scaffold itself was nonzero:

```text
context parameters             2304
kernel                         depthwise Conv1d k=3
weight max |.|                 0.0016619507
weight mean |.|                0.0000712489
weight L2                      0.0061931191
```

The complementary branch remained active at inference:

```text
query parameters               768
gate parameters                768
query L2                       0.5934500098
query cosine to primary       -0.1064639390
effective gate max |.|         0.0203592628
effective gate mean |.|        0.0036702447
```

## Mechanistic conclusion

The combined predeclared decision rule required both:

```text
B34 - B29 95% CI entirely < 0
and
B34 - B31 95% CI entirely within [-0.001,+0.001]
```

Both conditions passed. The frozen result is therefore:

```text
training_scaffold_interpretation            training_scaffold_benefit_supported
b34_b31_equivalent_within_margin            true
b34_mechanism_success                       true
```

Together with the earlier trained-B31 context-zero counterfactual, PV2 supports the global mechanism that the small local-context pathway changes the optimization/training trajectory while being materially unnecessary in the final inference function. This is a mechanism statement under the frozen weak-label/fixed-encoder contract; it is not evidence that B34 is independently clinically superior.

## Status after PV2

```text
B20  active historical/predictive checkpoint
B31  PV1-selected downstream development architecture
B34  frozen successful training-scaffold simplification/mechanistic architecture
B29  frozen no-scaffold mechanistic comparator
B33  frozen simple complementary-mean comparator
```

No B34.1 or target-specific scaffold mask, switch, blend, kernel retune, or threshold adaptation is authorized from PV2 outcomes. Further architecture invention against PV1 or PV2 is paused. Independent hidden competition or new external expert-labelled evidence remains required before active-model promotion.
