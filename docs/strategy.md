# Modeling strategy

> **Snapshot: 2026-08-09.** Canonical measured results are in [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md). B5 image-report representation training has completed all four predefined epochs; its frozen B4 probe is pending, so it has no OOF score yet.

## Core principle

`CNN_CPC` treats the challenge as a **weakly supervised multi-sequence MRI problem with a tiny trusted gold set**. The strategy prioritizes supervision quality, leakage control, metric alignment, representation quality and runtime discipline before increasing model scale.

The verified release contains 4,407 training studies, of which only 58 are fully gold-labelled and 4,349 are report-only.

## What the completed experiments now show

| Candidate | Macro AUC | Interpretation |
|---|---:|---|
| B0 random | `0.4763` | weak baseline |
| B1 strong MRI SSL | `0.5030` | useful in-domain representation signal |
| B2 lower encoder LR | `0.4993` | catastrophic forgetting not the main bottleneck |
| B3 pathology MIL | `0.4945` | architecture simplification not sufficient |
| B4 frozen SSL + classical | `0.5138` | best clean standalone point estimate |
| B4.1 shared policy | `0.4848` | too rigid |
| B4.2 grouped policies | `0.4901` | still too rigid/noisy |
| B4.3 two-way CV selector | `0.4966` | selector stabilization did not help |
| B1+B4 rank 50:50 | `0.5167` | highest numerical score, statistically tied with B4 |
| B5 image-report SSL | pending | representation training complete; frozen probe pending |

The main strategic conclusion is that the current useful lever is **representation quality from the report-only corpus**, not more downstream policy search on the same 58 gold labels.

## 1. Reports are training supervision, not inference inputs

Reports are converted to per-target states:

```text
positive
negated
uncertain
unmentioned
```

Report silence is not a negative. `unmentioned` receives zero direct report weight by default. Finite official labels override report-derived targets cell-by-cell.

Final inference remains MRI-only.

## 2. Fold-safe calibration

Report-state calibration uses only gold studies allowed in the current phase/fold. Confidence reflects both evidence and informativeness beyond target prevalence.

The trusted/pairwise-ranking thresholds are not lowered just to increase pseudo-label counts.

## 3. Compartment-aware OA parsing

The first real audit showed the original lexicon produced no useful OA supervision. The parser was expanded to recognize compartment-specific OA/arthrosis, cartilage loss, chondrosis/chondromalacia, osteophytes and related degenerative cartilage language while avoiding generic meniscal degeneration.

Verified report states:

| Target | Positive | Negated | Unmentioned |
|---|---:|---:|---:|
| Medial OA | 492 | 339 | 3,576 |
| Lateral OA | 409 | 387 | 3,611 |
| PF OA | 695 | 379 | 3,333 |

These cells remain weak supervision rather than gold-equivalent trusted examples.

## 4. Gold validation contract

The 58 official gold studies are deterministically balanced across three folds:

| Outer fold | Gold train | Inner selection | Outer validation |
|---|---:|---:|---:|
| 0 | 20 | 20 | 18 |
| 1 | 18 | 20 | 20 |
| 2 | 20 | 18 | 20 |

Every target has positives and negatives in each outer fold.

For the original neural Stage-1 protocol:

```text
outer gold -> final OOF only
inner gold -> epoch selection
remaining gold -> selection training
Phase A discarded
fresh Phase B -> all non-outer gold
```

## 5. Macro-aligned loss and trusted pairing

The supervised neural objective uses planned-epoch target normalization so weak-label coverage differences do not let frequently mentioned targets dominate the macro metric.

The pairwise ranking auxiliary is confidence-gated. The trusted/general sampler was changed to emit trusted rows in pairs for even batch sizes, activating ranking without weakening trust thresholds.

Verified smoke diagnostics:

```text
selection ranking pairs = 63
retrain ranking pairs   = 61
all 12 targets          = active
```

## 6. Six-stream MRI contract

```text
sagittal_fluid       sagittal_structural
coronal_fluid        coronal_structural
axial_fluid          axial_structural
```

Missing streams are explicit and masked. `axial_structural` is present in only about one quarter of training studies, so missing-stream handling is essential.

Each active series is represented with distributed 2.5D triplets and encoded with ConvNeXt-Tiny.

## 7. Strong competition-only MRI SSL

The strong SSL run uses only the 4,349 non-gold competition MRI studies. Same-study views/sequences provide representation positives; plane and sequence-type prediction are auxiliary objectives.

Completed run:

```text
epochs                 8
batches                 8,000
study draws             24,000
approx corpus passes    5.52
active 2.5D examples    238,274
loss                    ~3.434 -> ~2.862
```

Checkpoint:

```text
runs/ssl_strong/ssl_encoder.pt
```

No external pretrained weights are used.

## 8. End-to-end neural findings: B1-B3

B1 moved the point estimate above B0, but paired uncertainty remained wide. B2's 0.1x encoder LR did not improve B1, and B3's pathology-aware low-capacity MIL also did not improve pooled OOF.

This argues against spending the next experiments on small optimizer/head variations.

## 9. Frozen representation probe: B4

B4 freezes the strong SSL encoder and extracts deterministic per-stream mean/std/max features. The verified cache has shape:

```text
[58, 6, 2304]
```

Each target selects between fixed `all`/`prior` feature modes plus PCA and balanced logistic regression.

B4 reached:

```text
macro AUC = 0.5137567459
95% CI   = [0.4619827141, 0.5642366629]
P(B4 > B1) = 0.6378
```

This is the best clean standalone point estimate, but not a statistically proven improvement.

## 10. Why the B4 selector branch is closed

B4 target-wise policies were visibly unstable across folds. Three controlled attempts to reduce that variance all lowered pooled OOF:

- B4.1 one shared policy: `0.4848`;
- B4.2 four predefined group policies: `0.4901`;
- B4.3 target-wise two-way-CV selector: `0.4966`.

Further B4 policy/grid redesign driven by the same outer OOF would risk meta-overfitting. The B4 probe is therefore frozen for the first B5 comparison.

## 11. Fixed ensemble finding

A fixed B1+B4 raw-probability 50:50 average scored `0.5050`. A fixed 50:50 **rank** average scored `0.5167`.

Against B4:

```text
median ensemble-B4 difference = +0.00276
95% CI                        = [-0.03513, +0.04174]
P(ensemble > B4)              = 0.5544
```

Therefore the rank ensemble is kept as a fixed candidate but is not claimed to improve B4. No weight search is permitted on these 58 labels.

## 12. B5 strategy and completed pretraining

B5 changes the representation rather than the downstream classifier.

Only the 4,349 report-only studies were used for B5 pretraining; all 58 gold studies were excluded.

Text representation:

```text
normalized competition report
-> word TF-IDF (1-2 grams)
-> TruncatedSVD (<=256 dimensions)
-> normalized semantic target
```

MRI objective:

```text
strong SSL image-image objective
+ plane/sequence metadata objective
+ image-report contrastive alignment
+ cosine report alignment
```

A report embedding queue of 256 supplies additional semantic negatives for small MRI batches. Exact duplicate normalized report hashes are masked as false negatives.

No external language model and no external image weights were used. The report branch is discarded after training; the saved artifact is an MRI encoder.

Completed B5 run:

```text
checkpoint              runs/b5_report_ssl/b5_encoder.pt
epochs                  4
batches               4000
study draws          16000
active 2.5D examples 158886
total loss    5.5204 -> 4.7049
image loss    3.0068 -> 2.8937
report NCE    4.6031 -> 3.2901
report cosine 0.8015 -> 0.5924
budget limited          false
```

All logged objectives improved monotonically. This is stable optimization evidence, not yet a gold-label performance result.

## 13. Fixed B5 evaluation rule — current next step

B5 must now be tested with the **original B4 probe unchanged**:

```text
B4 encoder -> B4 frozen probe
versus
B5 encoder -> same B4 frozen probe
```

This isolates representation quality. Do not modify B4 feature modes, PCA grid, logistic grid, target groupings, ensemble weights, or B5 epoch count based on the upcoming outer OOF and then reuse the same OOF as a pristine estimate.

## 14. Validation campaign caveat

Although each experiment has an internally leakage-aware protocol, the same 58 gold studies have supported many method decisions. The campaign-level estimate is therefore increasingly **model-selection cross-validation**.

Do not:

- select per-target winners after viewing outer OOF;
- create more B4 selector variants;
- optimize ensemble weights;
- tune B5 on gold outer results without a new predeclared experiment;
- describe the best OOF point estimate as an independent hidden-test guarantee.

## 15. Runtime and competition policy

Long GPU runs use one GPU, CPU multiprocessing for data work, an 8.5-hour software budget and a ten-minute reserve. External pretrained weights remain disabled in the conservative configuration.

See [`competition_policy.md`](competition_policy.md) and [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).
