# B28 reused-expert development result

> **Status — 2026-08-16:** B28 NOT PROMOTED. B20 remains the active working model. The zero-gated max-evidence residual hypothesis is closed in its present formulation.

## Evaluation surface

B28 was compared with the canonical historical B20 checkpoint on the same 58-study expert development surface using 5,000 paired bootstrap replicates.

This surface is **not independent validation**. Historical B20 was selected using these expert studies. B28 used a fixed E2 endpoint and did not use expert labels for training or checkpoint selection. The historical 623-study weak-v2 partition is not a valid holdout because B28 trained on all 3,120 B20 weak-supervision studies.

## Macro result

```text
B20 macro AUC       0.6674066371
B28 macro AUC       0.6383456190
raw delta          -0.0290610180
paired median      -0.0286355485
paired 95% CI      [-0.0656236673, +0.0071218972]
P(B28 > B20)        0.0586
```

The paired confidence interval still spans zero, but the point estimate is materially negative and only 5.86% of paired bootstrap replicates favor B28. This does not support promotion.

## Per-target result

```text
Target                B20       B28        Delta
ACL                  0.5270     0.4877    -0.0392
MCL                  0.4626     0.5306    +0.0680
Medial Meniscus      0.6779     0.7212    +0.0433
Lateral Meniscus     0.7441     0.6758    -0.0683
Medial OA            0.6946     0.6946    +0.0000
Lateral OA           0.6712     0.5919    -0.0793
PF OA                0.6744     0.6062    -0.0682
Effusion             0.8646     0.7652    -0.0994
Synovitis            0.8375     0.6953    -0.1422
Baker's              0.7120     0.7645    +0.0525
Contusion            0.5209     0.4966    -0.0243
Fracture             0.6222     0.6306    +0.0083
```

Four targets improved, one was unchanged, and seven declined. The largest gains were MCL (+0.0680), Baker's (+0.0525), and Medial Meniscus (+0.0433). The largest losses were Synovitis (-0.1422), Effusion (-0.0994), Lateral OA (-0.0793), Lateral Meniscus (-0.0683), and PF OA (-0.0682).

## Mechanistic interpretation

B28 itself trained cleanly: its 768-dimensional residual gate started exactly at zero, remained far from tanh saturation, and the frozen encoder fingerprint was unchanged. Therefore this result is not a numerical-failure signal.

Instead, the result argues against adding an element-wise max-across-slice residual globally to every series token under the current B20 objective. Sparse/extreme slice features can help some targets, but the global residual also appears to inject harmful extreme evidence for several pathologies, especially Synovitis and Effusion.

The pattern is consistent with max pooling being highly sensitive to isolated activations. That can be useful when pathology is focal, but it can also amplify nuisance/extreme feature responses that B20's learned attention pooling had already suppressed. This is a hypothesis-level interpretation only; the reused 58-study expert surface cannot establish causality.

No post-hoc target-specific gate masking, target-selective ensembling, gate clipping, endpoint retuning, or selective max-residual use is authorized from this result because those changes would be driven by heavily reused development outcomes.

## Decision

```text
B28                 valid fixed-E2 experiment, NOT PROMOTED
max-residual family  CLOSED in this global featurewise formulation
active model         B20
weak-v2 evaluation   NOT VALID for B28
```

The next experiment should start from a new independently motivated imaging hypothesis rather than tuning B28 against these 58 expert studies.