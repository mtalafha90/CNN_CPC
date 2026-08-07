# Competition summary

Competition: **RSNA Knee Abnormality Detection** (Kaggle, 2026).

## Objective
Predict 12 abnormalities from a complete knee MRI study: ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, and Fracture.

## Metric
The competition uses **macro-averaged ROC AUC** across all 12 targets. Each target therefore has equal importance regardless of prevalence.

## Current public data snapshot
Public competitors examining the released files report:

- 4,407 training studies.
- 58 studies with explicit binary gold labels.
- 4,349 studies with no explicit gold labels.
- radiology reports are supplied for training studies.
- `train_series.csv` provides series-level metadata including anatomical plane and fluid-sensitive / fat-suppression flags.
- the hidden test set is evaluated through Kaggle notebook submissions.

Because the competition is active, this repository does not call any current method a winning solution and does not copy unverified leaderboard claims.

## Practical implication
The training problem is closer to semi-supervised/weakly supervised learning than a standard fully supervised image classification benchmark. A useful pipeline should exploit the radiology reports, while validation must remain anchored to the explicitly labeled studies.
