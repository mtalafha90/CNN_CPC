# B5 — competition-only image-report representation learning

> **Status — 2026-08-12:** **COMPLETED / RETAINED REPRESENTATION BASELINE.** B5 gold macro AUC was `0.5243650851`. It later served as the encoder source for B7-family experiments. B13 is now the reused-gold champion; B15 shows that representation gains in weak-teacher agreement do not necessarily transfer to expert gold.

## Goal

B5 changes the MRI representation while keeping the downstream B4 frozen-feature probe fixed. It uses the 4,349 report-only competition studies as semantic representation supervision without converting reports into brittle 12-target hard pseudo-labels.

All 58 gold studies are excluded from B5 representation training. Final inference remains MRI-only.

## Representation design

Report branch:

```text
normalized competition report
-> word TF-IDF 1-2 grams
-> at most 20,000 features
-> TruncatedSVD <=256 dimensions
-> L2 normalization
```

MRI branch starts from the completed competition-only strong SSL ConvNeXt checkpoint. No ImageNet/external image weights or external clinical language model are used in B5.

The objective combines image contrast, metadata supervision and image-report representation alignment. The report branch is training-only.

## Completed result

B5 training completed four epochs and produced:

```text
runs/b5_report_ssl/b5_encoder.pt
```

Under the unchanged B4 target-wise nested PCA/logistic-regression probe:

```text
B5 macro AUC       0.5243650851
95% CI            [0.4728108406,0.5761619105]
B4 macro AUC       0.5137567459
paired median      +0.0105821232
95% paired CI      [-0.0408197338,+0.0622131599]
P(B5 > B4)          0.656
```

The paired interval crosses zero, so B5 was retained as a representation baseline rather than claimed as a proven improvement.

## Downstream role

B5 became the initialization source for B7/B7.1 and several subsequent competition-only weak-supervision experiments. The predeclared B5+B7.1 50:50 rank ensemble was later rejected at `0.5540141184` versus B7.1 `0.5644802945`.

B13 eventually changed to a public ImageNet ConvNeXt protocol and reached `0.6293565948`, the current reused-gold development champion.

## B15-era interpretation

B15 returned to representation learning with:

```text
ImageNet -> knee-MRI same-study contrastive SSL -> B13 hierarchy
```

It improved frozen weak-v2 B6-teacher agreement from matched-control `0.5652498118` to `0.7319060415`, with paired median `+0.1675245839`, but its one-look expert-gold result was only `0.6209002783` versus B13 `0.6293565948`.

This reinforces one of B5's original motivations: report information may be valuable as representation-level semantic context without requiring every report to be converted into a fully trustworthy target label. A future richer image-report experiment may be reasonable, but only after the current B6 state audit and under a separately frozen protocol.

## Decision discipline

B5 remains a historical representation baseline. Do not retune its report-loss weights, temperatures, epochs, target-specific B4/B5 winners or ensemble weights from the reused gold set.

Current status: [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md). Post-B15 roadmap: [`RAISING_AUC.md`](RAISING_AUC.md).