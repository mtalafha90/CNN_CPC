# B4.2 — grouped policies on frozen SSL features

> **Status — 2026-08-12:** **COMPLETED / REJECTED.** B4.2 remains a historical selector-sharing ablation. B13 is now the reused-gold development champion; B15 completed without replacing it.

B4.2 tested an intermediate policy-sharing strategy between B4's 12 target-specific selectors and B4.1's single shared selector.

Predefined groups:

1. ligament/meniscus: ACL, MCL, Medial Meniscus, Lateral Meniscus;
2. osteoarthritis: Medial OA, Lateral OA, PF OA;
3. fluid/inflammatory: Effusion, Synovitis, Baker's;
4. osseous injury: Contusion, Fracture.

The groups were fixed from anatomy before evaluation.

## Result

```text
B4.2 macro AUC       0.4901328905
95% CI              [0.4430999386,0.5385509307]
median(B4.2-B4)     -0.0237234374
95% paired CI       [-0.0551712560,+0.0080455243]
P(B4.2 > B4)         0.0724
```

Decision: **rejected**. The four-group compromise removed too much target-specific flexibility. The grouping was not retuned from outer labels.

## Current successor context

```text
B13 gold  0.6293565948  retained champion
B14 gold  0.6197914249
B15 gold  0.6209002783
```

B15 passed frozen weak-v2 strongly but did not improve expert-gold macro AUC. Current development therefore prioritizes supervision-state diagnosis rather than reopening selector/group searches on the same 58 labels.

Current status: [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).