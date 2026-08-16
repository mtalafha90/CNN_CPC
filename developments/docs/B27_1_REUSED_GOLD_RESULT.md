# B27.1 reused-expert development result

> **Status — 2026-08-16:** B27.1 NOT PROMOTED. B20 remains the active working model. The B27 routing family is closed after this result unless a new hypothesis is defined independently of the reused expert outcome.

## Evaluation surface

B27.1 was compared with the canonical B20 checkpoint on the same 58-study expert development surface using 5,000 paired bootstrap replicates.

This surface is **not independent validation**. Historical B20 was selected using these expert studies, while B27.1 used a fixed E2 endpoint and did not use expert labels for training or checkpoint selection. The historical 623-study weak-v2 set is not a valid holdout because B27.1 trained on the full 3,120-study B20 weak-supervision surface.

## Macro result

```text
B20 macro AUC       0.6674066371
B27.1 macro AUC     0.6599232994
raw delta          -0.0074833377
paired median      -0.0070466019
paired 95% CI      [-0.0347249174, +0.0191823000]
P(B27.1 > B20)      0.2918
```

The paired interval spans zero, but the point estimate and bootstrap probability do not support promotion. This is post-hoc development evidence only and must not be described as independent validation.

## Per-target result

```text
Target                B20       B27.1      Delta
ACL                  0.5270     0.4926    -0.0343
MCL                  0.4626     0.4671    +0.0045
Medial Meniscus      0.6779     0.7404    +0.0625
Lateral Meniscus     0.7441     0.7006    -0.0435
Medial OA            0.6946     0.7163    +0.0217
Lateral OA           0.6712     0.6132    -0.0580
PF OA                0.6744     0.6577    -0.0167
Effusion             0.8646     0.8273    -0.0373
Synovitis            0.8375     0.8411    +0.0036
Baker's              0.7120     0.7953    +0.0833
Contusion            0.5209     0.4953    -0.0256
Fracture             0.6222     0.5722    -0.0500
```

Five targets improved and seven declined. The largest gains were Baker's (+0.0833) and Medial Meniscus (+0.0625). The largest declines were Lateral OA (-0.0580), Fracture (-0.0500), Lateral Meniscus (-0.0435), Effusion (-0.0373), and ACL (-0.0343).

## Interpretation

B27.1 successfully removed the pre-outcome collinearity defect of B27 and produced small, clinically interpretable pathology-specific routing biases. The audit-only Ollama review found no obvious anatomical or imaging-principle violation. Nevertheless, the routing intervention did not improve the 12-target macro on the reused expert surface.

The mixed per-target pattern suggests that a single global metadata-routing mechanism can help some acquisition-sensitive pathologies while perturbing others. In particular, the large Medial Meniscus and Baker's gains do not compensate for losses across Lateral OA, Fracture, Lateral Meniscus, Effusion, ACL, Contusion, and PF OA.

No post-hoc target-specific route tuning, route masking, loss reweighting, endpoint change, or selective ensembling is authorized from this 58-study result. Such changes would be outcome-driven on a heavily reused development surface.

## Decision

```text
B27                 structurally superseded pre-outcome by B27.1
B27.1               valid experiment, NOT PROMOTED
B27 routing family  CLOSED for this formulation
active model         B20
weak-v2 evaluation  NOT VALID for B27.1
```

The useful scientific lesson is retained: pathology-specific acquisition metadata can create target-specific gains, but the present additive routing formulation does not provide a robust macro improvement. Any future experiment should start from a new, independently motivated imaging hypothesis rather than tuning B27.1 against these 58 expert studies.
