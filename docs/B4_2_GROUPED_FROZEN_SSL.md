# B4.2 — grouped policies on frozen SSL features

B4.2 tested an intermediate policy-sharing strategy between B4's 12 target-specific selectors and B4.1's single shared selector.

> Current campaign status is summarized in [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).

## Predefined groups

The groups were fixed from anatomy before running B4.2:

1. `ligament_meniscus`: ACL, MCL, Medial Meniscus, Lateral Meniscus
2. `osteoarthritis`: Medial OA, Lateral OA, PF OA
3. `fluid_inflammatory`: Effusion, Synovitis, Baker's
4. `osseous_injury`: Contusion, Fracture

They were not learned from outer OOF results.

## Leakage contract

For each outer fold:

1. keep outer gold untouched;
2. use the configured inner fold for policy selection;
3. fit each candidate on the remaining gold selection-training fold;
4. select one policy per predefined pathology group by inner group macro AUC;
5. refit separate target classifiers on all non-outer gold using the group's selected policy;
6. predict the untouched outer fold once.

The frozen SSL encoder was trained only on non-gold competition images.

## Candidate grid

The original B4 grid was unchanged:

```text
feature mode:   all, prior
PCA components: 4, 8, 12, 16
logistic C:     0.1, 1.0
```

## Reproduction

```bash
rsna-knee-b4-grouped \
  --config configs/train_local_ssl_strong.yaml \
  --features runs/b4_frozen_ssl/gold_features.npz \
  --out-root runs/b4_2_grouped_ssl \
  --n-bootstrap 5000
```

## Final result

```text
pooled macro AUC = 0.4901328905
95% CI           = [0.4430999386, 0.5385509307]
```

Against B4, using A=B4 and B=B4.2:

```text
paired median difference = -0.0237234374
95% CI                   = [-0.0551712560, +0.0080455243]
P(B4.2 > B4)             = 0.0724
```

B4.2 also did not improve B1 (`P(B4.2 > B1)=0.3326`).

## Decision

**Rejected.** The four-group compromise still removes too much target-specific flexibility. The groups must not be retuned from these outer labels.
