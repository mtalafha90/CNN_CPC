# B9 — strict semantic MRI stream routing

> **Status — 2026-08-10:** **IMPLEMENTED / PREDECLARED / REAL-DATA TRAINING PENDING.** B9-v1 was defined after a label-free train/test series-metadata audit and before its first 58-study gold development evaluation.

## Motivation

B7.1 remains the current best standalone development model:

```text
macro AUC = 0.5644802945
95% CI   = [0.5052432984, 0.6229422178]
```

B8 spatial-anatomy learning was rejected at `0.5300962807` and its spatial-prior branch is closed. The next experiment therefore returns to the successful B7.1 architecture and tests a data-routing inconsistency discovered from `train_series.csv` / `test_series.csv`, without using target labels.

The historical dual-stream selector attempts to fill both the fluid and structural slot of a plane whenever at least two series are present. If a study has two acquisitions of only one contrast class, one can be placed in the opposite semantic slot.

Example:

```text
two sagittal structural series
historical routing -> sagittal_fluid + sagittal_structural
B9 strict routing  -> missing + sagittal_structural
```

This means the network can receive a structural series together with the learned `sagittal_fluid` stream embedding, or the converse.

## Label-free metadata audit

The full released training metadata contains 4,407 studies and 24,371 series. Comparing the historical selector against exact `Fluid_Sensitive` semantics gives:

| Stream | Historical selected | Strict selected | Historical semantic mismatches removed |
|---|---:|---:|---:|
| sagittal_fluid | 4,401 | 4,150 | 251 |
| sagittal_structural | 4,294 | 4,266 | 28 |
| coronal_fluid | 4,250 | 4,248 | 2 |
| coronal_structural | 3,440 | 3,406 | 34 |
| axial_fluid | 4,407 | 4,407 | 0 |
| axial_structural | 1,094 | 857 | 237 |
| **Total** | **21,886** | **21,334** | **552** |

Thus 552 / 21,886 historically selected stream assignments (`2.52%`) contradict the supplied contrast metadata. Strict routing removes all of them.

Because removing a false assignment can also free a different valid same-class series to become the retained slot, 805 individual stream assignments change between historical and strict routing. This is a direct consequence of enforcing semantic slots, not an additional target-driven selection heuristic.

The provided three-study test metadata show the same issue at smaller scale:

```text
historical selected streams  14
strict selected streams      13
historical mismatches          1
strict mismatches              0
```

The affected test case contains two sagittal structural series and no sagittal fluid series. Historical routing fabricates a sagittal-fluid slot; B9 leaves that slot missing.

## Single scientific change versus B7.1

B9 changes only the six-stream routing rule.

Historical B7.1 dual routing:

```text
if multiple series exist in a plane:
    rank a fluid candidate
    rank a structural candidate
    try to make their UIDs different
```

B9 strict routing:

```text
*_fluid:
    candidates = Fluid_Sensitive == True only

*_structural:
    candidates = Fluid_Sensitive == False only

if no candidate of the required class exists:
    slot = None
    presence mask = False
```

Unknown contrast after the normal metadata-repair pass is not promoted into either semantic class.

The historical routing implementation remains untouched so B7.1 is exactly reproducible. B9 uses `src/rsna_knee/strict_routing.py` and records a label-free `routing_audit.json` for the actual weak-training study subset.

## Everything else is frozen from B7.1

B9 deliberately returns to the B7.1 model rather than inheriting B8.

Unchanged components:

- B5 competition-only image-report encoder initialization;
- six-stream 2.5D ConvNeXt encoder;
- one globally pooled token per sampled slice;
- 96 MRI memory tokens/study at full six-stream coverage;
- slice-position and stream embeddings;
- cross-sequence Transformer;
- 12 pathology queries and cross-attention;
- frozen B6 v1.2.1 weak labels;
- asymmetric positive/negative soft-target policy;
- target balancing;
- 16 sampled slices/stream;
- batch size 2;
- 4 epochs;
- 1,560 batches/epoch;
- 3,120 weak-training study draws/epoch before any all-MRI-empty filtering;
- encoder LR `1e-5`, head LR `1e-4`, cosine schedule to `1e-6`;
- augmentation policy;
- TTA offsets `[-1,0,1]`;
- 5,000 study-level bootstrap replicates;
- no gold-gradient use;
- no gold early stopping.

Configuration:

```text
configs/b9_strict_routing.yaml
```

Package / entry points:

```text
package 0.14.0
rsna-knee-b9
rsna-knee-b9-eval
```

## Frozen B6 supervision

```text
report-only studies       4349
active studies            3120
usable cells             14123
positive cells            6871
negative cells            7252
```

| B6 state | Soft target | Base weight |
|---|---:|---:|
| positive | 0.85 | 0.50 |
| negated | 0.05 | 1.00 |
| uncertain | ignored | 0.00 |
| unmentioned | ignored | 0.00 |

The B6 parser and supervision policy remain frozen.

## Test the implementation

```bash
cd /media/talafha/Disk_1/CNN_CPC
git pull --ff-only origin main
python -m pip install -e .

pytest -q \
  tests/test_b6_report_labels.py \
  tests/test_b6_gold_audit.py \
  tests/test_b7_weak_supervision.py \
  tests/test_b9_strict_routing.py
```

The B9 tests explicitly verify that:

- two structural series cannot fabricate a fluid slot;
- two fluid series cannot fabricate a structural slot;
- unknown contrast remains missing;
- strict routing has zero semantic mismatches;
- the historical substitution failure is visible in the routing audit;
- selection is deterministic.

## Train B9-v1

```bash
rsna-knee-b9 \
  --config configs/b9_strict_routing.yaml \
  --data-root "$DATA_ROOT" \
  --b5-checkpoint runs/b5_report_ssl/b5_encoder.pt \
  --b6-root runs/b6_report_labels_v121 \
  --out-root runs/b9_strict_routing
```

Outputs:

```text
runs/b9_strict_routing/
├── b9_model.pt
├── history.json
├── policy.json
├── routing_audit.json
└── supervision_plan.json
```

Before gold evaluation inspect:

```bash
cat runs/b9_strict_routing/routing_audit.json
cat runs/b9_strict_routing/history.json
cat runs/b9_strict_routing/supervision_plan.json
```

The routing audit must report:

```text
routing_policy              fluid_sensitive_exact_v1
strict_semantic_mismatches  0
```

Every completed full epoch should retain the frozen B6 supervision-cell counts unless strict routing unexpectedly removes every MRI stream for a weak-training study. Any such filtering is recorded explicitly in `supervision_plan.json`.

## Gold development evaluation

Only after the frozen training artifacts are inspected:

```bash
rsna-knee-b9-eval \
  --config configs/b9_strict_routing.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b9_strict_routing/b9_model.pt \
  --out-root runs/b9_strict_routing/gold_eval
```

Primary benchmark:

```text
B7.1 macro AUC = 0.5644802945
```

Primary statistical comparison: paired B7.1 -> B9 study-level bootstrap with 5,000 replicates.

## Decision rule

B9-v1 is a one-shot controlled routing experiment. Do not modify the strict rule, add target-specific routing, selectively restore substituted streams, tune contrast rules from per-target gold AUCs, or search blend weights on the 58-study development set and still call the result B9-v1.

The B9 hypothesis was motivated by metadata consistency rather than target outcomes, but its eventual 58-study score is still a development/model-selection estimate because the broader campaign has repeatedly used these same gold studies.
