# Completed PV1 B31 local-context counterfactual result

> **Post-result mechanism audit.** This records the already-frozen inference-only intervention defined in `PV1_B31_CONTEXT_COUNTERFACTUAL.md`. It is not a prospective model-selection result and does not change the original PV1 ranking.

## Surface and intervention

```text
PV1 validation studies                624
PV1 validation MRI series            3544
PV1 split SHA256
a0032307abb1ab99724eb39fac25332ce131c575f64d823083bb37f5ec20d1e6

intervention
same trained B31 checkpoint with local_context.weight set to exact zero at inference only
```

Expert labels were not read. All other checkpoint parameters were unchanged.

The trained local-context branch was nonzero before the intervention:

```text
parameters          2304
kernel              depthwise Conv1d k=3
weight max |.|      0.0009498816
weight mean |.|     0.0000572350
weight L2           0.0046429192
```

After intervention, max absolute weight, mean absolute weight, and L2 norm were all exactly zero.

## Prediction perturbation

```text
mean absolute probability change       0.0000669380
RMS probability change                 0.0001171025
maximum absolute probability change    0.0008941889
```

The direct inference perturbation from the trained local-context branch was therefore very small.

## Primary result

```text
normal B31 macro weighted soft BCE      0.5743065510
context-zero B31 BCE                    0.5743052782

context-zero - normal B31
median difference                      -0.0000012547
95% CI                                 [-0.0000110175, +0.0000086129]
P(context-zero better)                  0.5974
bootstrap replicates                    5000
```

The confidence interval tightly includes zero. Under the predeclared interpretation, a direct inference contribution is unresolved and an optimization/training-path effect remains plausible.

The secondary macro AUC changed from `0.7567308761` for normal B31 to `0.7572869665` for context-zero B31. This secondary change is descriptive only.

## Mechanism placement

Context-zero B31 still clearly improved over the frozen B29 and B33 reference predictions:

```text
context-zero B31 - B29
median     -0.0213725084
95% CI     [-0.0321308565, -0.0098262813]
P better    1.0000

context-zero B31 - B33
median     -0.0107608640
95% CI     [-0.0199149107, -0.0022454949]
P better    0.9930
```

The result therefore did not support the hypothesis that B31's advantage required the learned context operation in the deployed inference function. It motivated the separately frozen B34 training-scaffold experiment, which was later tested successfully on PV2.

## Governance

The original PV1 architecture-selection ranking remains:

```text
B31 > B33 > B20
```

This counterfactual does not create a new independently validated model. No target-wise context masks, switches, blends, kernel retunes, B29.1, or B31.1 are authorized from this result.
