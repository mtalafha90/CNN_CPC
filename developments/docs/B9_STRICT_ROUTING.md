# B9 — strict semantic MRI stream routing

> **Status — 2026-08-12:** **COMPLETED / REJECTED GLOBALLY.** B9 gold macro AUC was `0.5334962669`. The label-free routing audit remains valid and useful, but strict routing did not improve the global development metric. B13 is now the reused-gold champion.

## Motivation

The historical six-stream selector could populate a missing contrast slot with a same-plane acquisition from the opposite contrast class. B9 enforced exact `Fluid_Sensitive` semantics.

Full released-training metadata audit:

```text
historical selected streams  21886
strict selected streams      21334
wrong-slot substitutions       552
wrong-slot fraction            2.52%
strict semantic mismatches        0
```

Per stream, wrong-slot assignments removed:

```text
sagittal_fluid       251
sagittal_structural   28
coronal_fluid          2
coronal_structural    34
axial_fluid            0
axial_structural     237
```

This finding was label-free and remains an important data-quality observation even though the model experiment was not successful globally.

## Single scientific change versus B7.1

```text
*_fluid       -> Fluid_Sensitive == True only
*_structural  -> Fluid_Sensitive == False only
missing class -> None / masked
```

Everything else returned to the B7.1 recipe: B5 initialization, B6 v1.2.1 weak labels, six-stream KneeMILNet, 16 sampled positions/stream, four full epochs, same optimizer/augmentation/TTA, and zero gold gradients/early stopping.

## Completed result

```text
B9 macro AUC    0.5334962669
B7.1 macro AUC  0.5644802945
```

B9 was rejected as a global replacement. The result does not invalidate the metadata audit; it shows that removing the 2.52% semantic substitutions did not produce a better global ranking under this architecture/training recipe.

## Successor context through B15

```text
B10 gold  0.5523982721
B11.1     0.5506902702
B12 gold  0.5660915179
B13 gold  0.6293565948  retained champion
B14 gold  0.6197914249
B15 gold  0.6209002783
```

B15 strongly improved frozen weak-v2 teacher agreement (`0.7319060415` vs control `0.5652498118`, paired median `+0.1675245839`) without improving global expert-gold AUC. The current next diagnostic is therefore a B6 report-state audit, not further target-specific routing adjustments.

## Decision

B9 remains **completed/rejected globally**. Do not restore individual substitutions, tune routing per target, or mix B9 target winners with B13/B15 based on reused gold.

The historical selector remains in the repository for reproducibility; strict routing remains available as an engineering/data-contract option but is not the retained model path.

Current status: [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md). Validation policy: [`VALIDATION.md`](VALIDATION.md).