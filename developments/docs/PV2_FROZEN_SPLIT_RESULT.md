# PV2 frozen split result

> **Frozen before any PV2-B29, PV2-B31, or PV2-B34 matched training result was observed.**

PV2 is a nested, study-level weak-label metric surface drawn only from the 2,496 studies in the already-frozen PV1 training partition. The original 624-study PV1 validation partition remains locked and is excluded from both PV2 training and PV2 validation.

## Frozen fingerprints

```text
PV2 version
prospective_weak_nested_pv1train_hash_80_20_v1

parent PV1 split SHA256
a0032307abb1ab99724eb39fac25332ce131c575f64d823083bb37f5ec20d1e6

PV2 split SHA256
b53331ce314b2d2ccc68aea1737427c01bd0d916997e78fbefe88fec5cc95855

source UID SHA256
cea2e6da128d74c4ae62568296b7a9179d2e200298eff5ac5eca57cccf032e0b

PV2 training UID SHA256
21b7dcfa25d3ebc7b9ff905ddb0862628bae3a63e1be0208a0f4f3d7a0bee4b7

PV2 validation UID SHA256
6e6124fcdc20a3f2f8cae8a77b62316096bac0980f5f7108a4ce3bc27a55795c

locked parent PV1 validation UID SHA256
878a3cd83e1b023b9cea2c8918d4896e2326f98ab6140dbfcb8558aa3a2bee3d
```

The source UID fingerprint is exactly the frozen PV1 training UID fingerprint, and the locked-parent fingerprint is exactly the frozen PV1 validation UID fingerprint.

## Assignment contract

Membership is assigned only by StudyInstanceUID:

```text
sort the frozen PV1-training StudyInstanceUIDs by
SHA256("CNN_CPC|prospective-weak-v2|parent-pv1-train|2026-08-17" + NUL + uid)

first 499 ranked UIDs -> PV2 validation
remaining 1,997 UIDs -> PV2 training
```

No B6 label state, confidence, expert label, model output, PV1 result, B29 addendum result, or B31 counterfactual result enters split membership.

## Exact separation audit

```text
source studies                         2496
PV2 training studies                   1997
PV2 validation studies                  499
locked original PV1 validation          624
PV2 training/validation overlap            0
PV2 training/locked-PV1 overlap            0
PV2 validation/locked-PV1 overlap          0
```

The generated UID lists are unique, lexicographically stored, reproduce the frozen UID-hash assignment exactly, and their recorded SHA-256 fingerprints recompute exactly. The complete split-core JSON recomputes to the recorded PV2 split SHA-256 above.

## Descriptive weak-supervision composition

Weak labels were inspected only after membership was fixed.

```text
partition                       cells    positive   negative
PV2 training                     9005       4380       4625
PV2 validation                   2298       1179       1119
locked parent PV1 validation     2820       1312       1508
```

PV2 validation per-target composition:

```text
Target               usable   positive   negative
ACL                     269        109        160
MCL                     220         53        167
Medial Meniscus         285        193         92
Lateral Meniscus        268         81        187
Medial OA               140         81         59
Lateral OA              127         67         60
PF OA                   169        109         60
Effusion                340        233        107
Synovitis                74         68          6
Baker's                 163         91         72
Contusion               130         63         67
Fracture                113         31         82
```

The primary PV2 metric remains macro per-target B6-weighted soft-label BCE. Secondary hard-state AUC remains descriptive; in particular, Synovitis is still highly imbalanced despite having six negative cells on PV2 validation.

## Exposure limitation

PV2 is **not independent clinical validation** and is **not historically untouched validation**. Its 499 validation studies were part of historical downstream training before PV2 was defined, and the frozen B16 encoder was aligned using reports from the broader non-gold population. PV2 exists only as a newly hidden metric surface for the predeclared matched B34 mechanism test while keeping the original 624 PV1 validation studies locked.

## Predeclared B34 mechanism test

The matched fixed-E2 experiment is frozen as:

```text
PV2-B29  = no training scaffold; B29 complementary-query inference form
PV2-B31  = local-context branch active in training and retained at inference
PV2-B34  = same local-context scaffold during training, exact context bypass at inference
```

Primary mechanism comparison:

```text
B34 - B29 weighted soft BCE
```

A training-scaffold benefit is supported only if the paired 95% interval is entirely below zero.

Inference-bypass replication:

```text
B34 - B31 weighted soft BCE
```

Metric-resolution equivalence is declared only if the entire paired 95% interval lies inside the predeclared `[-0.001,+0.001]` macro-BCE margin.

Overall B34 mechanism success requires **both** conditions. No target-specific switching, blending, scaffold masking, kernel retuning, B34.1, or post-result threshold adjustment is allowed from PV2 outcomes.
