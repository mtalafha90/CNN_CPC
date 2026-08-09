# B4.2 — grouped policies on frozen SSL features

B4.2 is an intermediate-variance classical downstream model built on the same
competition-only frozen SSL feature cache used by B4 and B4.1.

## Motivation

- B4 target-wise policy selection achieved the strongest frozen-feature point
  estimate so far, but selected `(feature mode, PCA dimension, C)` independently
  for all 12 targets and showed substantial fold-to-fold instability.
- B4.1 forced one shared policy across all 12 targets and degraded performance,
  indicating that pathology heterogeneity is real.
- B4.2 therefore fixes four medically motivated pathology groups *a priori* and
  selects one policy per group and outer fold.

The groups are not learned from OOF results:

1. `ligament_meniscus`: ACL, MCL, Medial Meniscus, Lateral Meniscus
2. `osteoarthritis`: Medial OA, Lateral OA, PF OA
3. `fluid_inflammatory`: Effusion, Synovitis, Baker's
4. `osseous_injury`: Contusion, Fracture

## Leakage control

For outer fold `f`:

1. the outer gold fold is held untouched;
2. the configured inner fold is used only for policy selection;
3. the remaining gold fold trains each candidate;
4. one candidate is chosen separately for each predefined pathology group by
   inner group macro ROC-AUC;
5. each target is refit independently on all non-outer gold studies using its
   group's chosen policy;
6. the untouched outer fold is predicted once.

Outer labels never select a group, feature mode, PCA dimension, or logistic C.
The frozen SSL encoder was trained only on non-gold competition training images.

## Candidate grid

B4.2 deliberately keeps the original B4 grid unchanged:

- feature mode: `all`, `prior`
- PCA components: `4`, `8`, `12`, `16`
- logistic `C`: `0.1`, `1.0`

No new hyperparameter is introduced.

## Run

The MRI feature cache from B4 is reused; no GPU extraction is required.

```bash
rsna-knee-b4-grouped \
  --config configs/train_local_ssl_strong.yaml \
  --features runs/b4_frozen_ssl/gold_features.npz \
  --out-root runs/b4_2_grouped_ssl \
  --n-bootstrap 5000
```

Outputs include per-fold `selection.json`, `oof.csv`, and `bootstrap.json`, plus
pooled `oof.csv`, `evaluation.json`, and `policy.json`.

## Interpretation

B4.2 should be compared pairwise with both:

- B1 strong SSL end-to-end baseline;
- B4 target-wise frozen-feature classifier.

A B4.2 improvement over B4 would support the hypothesis that four-group policy
sharing reduces selection variance without erasing clinically meaningful target
heterogeneity. If B4 remains superior, retain B4 as the current frozen-feature
candidate and do not tune the groups on outer OOF labels.
