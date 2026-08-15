# B2 — discriminative SSL fine-tuning

> **Status — 2026-08-12:** **COMPLETED / REJECTED.** B2 remains a historical optimizer ablation. B13 is now the reused-gold development champion; completed B15 did not replace it.

B2 tested whether the strong in-domain SSL encoder was being overwritten too quickly by using a lower encoder learning rate while keeping the Stage-1 architecture and head learning rate fixed.

## Single intervention

```yaml
encoder_lr: 0.00001
lr: 0.0001
```

No encoder freezing was used.

## Result

```text
B2 macro AUC       0.4993244663
95% CI            [0.4512751879,0.5464103264]
B1 macro AUC       0.5030284974
median(B2-B1)     about -0.00395
95% paired CI     [-0.05905,+0.05269]
P(B2 > B1)         0.4506
```

Decision: **rejected**. Lowering the encoder learning rate did not produce a stable improvement over B1, and no further B2 LR search was performed on the same outer gold labels.

## Current successor context

Later representation and weak-supervision experiments progressed substantially beyond B2:

```text
B13 gold  0.6293565948  retained champion
B14 gold  0.6197914249
B15 gold  0.6209002783
```

B15 also showed a large frozen weak-v2 teacher-agreement gain (`0.7319060415` versus control `0.5652498118`) without expert-gold improvement. The current bottleneck investigation therefore focuses on supervision-state quality, not another encoder-LR sweep.

Current status: [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).