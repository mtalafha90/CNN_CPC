# B46 five-fold GPU preflight record

**Date:** 2026-08-25

**Status:** ALL FIVE PREFLIGHTS PASSED. B46 fixed-E2 fold training is authorized under the already-frozen protocol. This record does not change the fold manifest, gold weight, architecture, optimizer, or decision rule.

## Frozen manifest

```text
manifest SHA-256
054c4ce9ab808af714cd4b86f159ef02a2b7e67de0c80e5c930d29fa5fb22e03

fold sizes
12 / 12 / 12 / 11 / 11
```

The manifest must not be regenerated after this point.

## Common runtime

Every fold preflight ran on:

```text
device         NVIDIA RTX A4500 Laptop GPU
GPU count      1
precision      bf16
workers        0
pin_memory     false
```

Every fold inherited and passed the full B42 geometry/gradient preflight:

```text
synthetic 448x448     PASS
synthetic 320x640     PASS
synthetic 640x320     PASS
synthetic 256x800     PASS
real rectangular MRI PASS
worst-case 14 + 13 series backward PASS
```

The real rectangular probe preserved ragged/native-aspect behavior and included output series geometries such as `512x416`, `480x448`, and `448x448`.

The common worst-case B42 backward values were:

```text
total       2.169435
combined    0.632212
local       1.537223
```

## Memory envelope

Across the five independent fold processes, the B42 worst-case preflight reported:

```text
RSS current       4.24--4.28 GiB
RSS peak          4.52--4.54 GiB
host available    50.48--50.85 GiB
CUDA current      ~0.25 GiB allocated
CUDA peak         ~1.45 GiB allocated
CUDA reserve peak ~1.67 GiB
```

No memory pressure or GPU-capacity warning was observed.

## Clean-gold gradient probe

Every fold then ran a held-in official expert study through the declared B46 loss and verified:

```text
official target values       exact hard 0/1
gold cell weight             exactly 4.0
sparse evidence-head gradient nonzero
trainable encoder-tail gradient nonzero
held-out fold not used by this probe
```

Observed fold probes:

| Fold | total | combined | local | result |
|---:|---:|---:|---:|---|
| 0 | 2.015580 | 0.398402 | 1.617178 | PASS |
| 1 | 1.891767 | 0.590784 | 1.300982 | PASS |
| 2 | 2.015580 | 0.398402 | 1.617178 | PASS |
| 3 | 2.015580 | 0.398402 | 1.617178 | PASS |
| 4 | 2.015580 | 0.398402 | 1.617178 | PASS |

The repeated probe values in folds 0/2/3/4 arise because the first held-in gold item selected by the deterministic dataset ordering is the same study in those folds. Fold 1 excludes that study and therefore probes a different held-in study. This is expected and does not imply cross-fold weight sharing.

## Authorization

```text
B46 fold 0 preflight PASS
B46 fold 1 preflight PASS
B46 fold 2 preflight PASS
B46 fold 3 preflight PASS
B46 fold 4 preflight PASS
ALL B46 PREFLIGHTS PASS
```

Training may now proceed for folds 0 through 4, sequentially, with exactly two epochs each. Intermediate loss values, gates, or fold-specific behavior must not be used to alter the frozen protocol or stop/select a fold checkpoint. The only valid endpoint for every fold is fixed epoch 2.
