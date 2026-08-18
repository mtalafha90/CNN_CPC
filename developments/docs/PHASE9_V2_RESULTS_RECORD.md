# Phase 9 v2 — Matched Supervision Results Record

**Status:** Frozen / analysis complete; no retraining or target-specific tuning performed.

## Experimental design

Phase 9 v2 compared a matched **control** arm against a **candidate** arm.

- Studies: 3,850 training studies in each arm.
- PV2 holdout: 499 studies, held out from both arms.
- MRI series: 21,260 in each arm.
- Architecture/checkpoint family: fixed B34 / fixed E2.
- Construction seed: `40002026`.
- Loader seed: `40102026`.
- Post seed: `40202026`.
- The PV2 evaluation used the original frozen B6 labels only.
- The candidate differs from control by adding the frozen Phase-8 translated/rescued supervision; MRI exposure and the matched study population were kept identical.

## Training results

| Arm | E1 loss | E1 time | E2 loss | E2 time | Output |
|---|---:|---:|---:|---:|---|
| Control | 0.6857720218 | 37.9 min | 0.5927135918 | 40.7 min | `runs/phase9_matched_supervision_v2/control/model.pt` |
| Candidate | 0.7408268425 | 39.7 min | 0.6582265225 | 40.8 min | `runs/phase9_matched_supervision_v2/candidate/model.pt` |

## Frozen PV2 evaluation

Primary metric: original-B6-weighted soft-label BCE. Lower is better.

- Control BCE: **0.6411802086**
- Candidate BCE: **0.6313035785**
- Candidate minus control: **-0.0098766301**
- Paired bootstrap median difference: **-0.0097649364**
- 95% CI: **[-0.0199023586, +0.0000842380]**
- Probability candidate better: **0.9742**
- Valid replicates: **5000 / 5000**

Interpretation: strong directional improvement, but the frozen two-sided 95% interval narrowly includes zero; therefore the BCE primary is **inconclusive / borderline**, not a confirmed overall win.

## Competition-aligned secondary AUC

Macro ROC AUC across all 12 targets, using original-B6 weak states as truth.

- Control macro AUC: **0.7401347411**
- Candidate macro AUC: **0.7433569106**
- Point difference: **+0.0032221694**
- Paired bootstrap median difference: **+0.0029062334**
- 95% CI: **[-0.0084747230, +0.0150826454]**
- Probability candidate better: **0.6897312475**
- Valid replicates: **4986 / 5000**
- Bootstrap unit: `StudyInstanceUID`
- Strict all-12-target requirement per replicate: yes

Interpretation: the candidate has a small favorable point estimate, but the macro-AUC result is **inconclusive**.

## Per-target AUC audit

| Target | Control | Candidate | Delta | 95% CI | P(candidate better) | Added cells | Added positive | Added negative |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| ACL | 0.65464 | 0.65321 | -0.00143 | [-0.02588, 0.02214] | 0.4570 | 435 | 272 | 163 |
| MCL | 0.61541 | 0.60547 | -0.00994 | [-0.04040, 0.02041] | 0.2590 | 267 | 54 | 213 |
| Medial Meniscus | 0.72623 | 0.74059 | +0.01436 | [-0.00979, 0.03866] | 0.8776 | 699 | 585 | 114 |
| Lateral Meniscus | 0.68489 | 0.67789 | -0.00700 | [-0.03340, 0.01868] | 0.2970 | 592 | 269 | 323 |
| Medial OA | 0.83427 | 0.84955 | +0.01528 | [-0.01170, 0.04429] | 0.8600 | 206 | 199 | 7 |
| Lateral OA | 0.75871 | 0.75846 | -0.00025 | [-0.03013, 0.03093] | 0.4778 | 137 | 132 | 5 |
| PF OA | 0.75810 | 0.76208 | +0.00398 | [-0.02534, 0.03170] | 0.6012 | 276 | 253 | 23 |
| Effusion | 0.74161 | 0.71541 | **-0.02619** | **[-0.04834, -0.00522]** | **0.0082** | 597 | 433 | 164 |
| Synovitis | 0.77941 | 0.78922 | +0.00980 | [-0.04925, 0.06618] | 0.6414 | 35 | 35 | 0 |
| Baker's | 0.73718 | 0.71276 | -0.02442 | [-0.06456, 0.01342] | 0.1074 | 356 | 226 | 130 |
| Contusion | 0.80005 | 0.85548 | **+0.05544** | **[0.02061, 0.09325]** | **0.9990** | 243 | 213 | 30 |
| Fracture | 0.79111 | 0.80016 | +0.00905 | [-0.01924, 0.03844] | 0.7146 | 58 | 48 | 10 |

The per-target analysis is descriptive only and was not used for targetwise tuning, rescue filtering, or model mixing.

## Macro influence / leave-one-target-out audit

Overall macro delta: **+0.0032221694**.

The strongest influence is Contusion:

- Contusion delta: **+0.0554371002**.
- Contribution to macro target sum: **+0.0046197584**.
- Macro delta after removing Contusion: **-0.0015246425**.
- Therefore, removing Contusion **flips the overall macro-AUC sign**.

Other leave-one-target-out deltas:

- Medial OA removed: +0.0021264428.
- Medial Meniscus removed: +0.0022095173.
- Synovitis removed: +0.0026238283.
- Fracture removed: +0.0026925490.
- PF OA removed: +0.0031536816.
- Lateral OA removed: +0.0035377081.
- ACL removed: +0.0036454108.
- Lateral Meniscus removed: +0.0041512835.
- MCL removed: +0.0044189466.
- Baker's removed: +0.0057350961.
- Effusion removed: +0.0058962113.

## Key scientific interpretation

Phase 9 v2 should **not** be reported as a general model win. The aggregate BCE and macro-AUC estimates both favor the candidate, but their frozen uncertainty intervals include zero.

The stronger finding is pathology-specific:

1. **Contusion:** statistically supported positive shift on this frozen PV2 weak-label surface (+0.0554 AUC; 95% CI entirely above zero).
2. **Effusion:** statistically supported negative shift (-0.0262 AUC; 95% CI entirely below zero).
3. Other targets are mostly directional but inconclusive.
4. The positive overall macro-AUC is substantially driven by Contusion; removing it reverses the sign.

This suggests that translated/rescued supervision may affect pathology representations heterogeneously rather than providing a uniform gain. This is a hypothesis, not a causal claim.

## Governance / non-circularity

- The PV2 holdout was excluded from both Phase-9 training arms.
- Evaluation used original frozen B6 labels, not expert or hidden competition labels.
- Per-target results were inspected only descriptively after the frozen evaluation.
- No target-specific rescue filtering, reweighting, model mixing, or retraining was performed based on these results.
- The next investigation should audit the **label-generation evidence** behind the rescued cells, especially Contusion versus Effusion, without using PV2 outcomes to alter the rescue set.

## Artifact paths

```text
runs/phase9_matched_supervision_v2/control/model.pt
runs/phase9_matched_supervision_v2/candidate/model.pt
runs/phase9_matched_supervision_v2/eval/
runs/phase9_matched_supervision_v2/target_analysis/
```

The raw Phase-7 rescue evidence required for the mechanism audit was not found at the attempted local path. The available `translation_rescue_supervision_phase8.zip` did not expose the expected `translation_cache.jsonl`, `full_population_rescue_audit.csv`, or `recovered_cells.csv` in the inspection performed so far. Therefore the label-generation mechanism audit remains **pending** and no claims about the underlying translated report phrases should be made from the Phase-9 metrics alone.
