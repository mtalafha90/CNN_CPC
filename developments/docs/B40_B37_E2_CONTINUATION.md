# B40 — B37 E2 one-epoch optimizer-reset continuation

## Status

**COMPLETED / NOT PROMOTED.**

B40 completed its one predeclared optimizer-reset continuation epoch from the
immutable B37 fixed-E2 checkpoint. The continuation reduced the weak-supervision
training losses, but the reused Expert-58 diagnostic did not improve overall
macro AUC relative to B37 E2. B40 is therefore retained as a completed negative/
neutral duration experiment and is not preferred over B37.

B37 remains immutable and retains the independently observed Kaggle score of
`0.714`. No B40 E4 continuation is allowed from this result.

## Purpose

The completed B37 endpoint used exactly two epochs. Its `b37_model.pt` retains
the full trained model state but not the AdamW moment buffers. A genuine resumed
third B37 epoch was therefore impossible from the saved artifact. B40 records
this limitation explicitly rather than treating the run as an exact optimizer
resume:

```text
start point      exact immutable B37 fixed-E2 model parameters
new training     exactly one full-coverage epoch (absolute epoch 3)
optimizer        fresh AdamW with the same B37 parameter groups and rates
changed science  optimization duration, with declared optimizer-state reset
unchanged         data, labels, crop, 448 input, 32 centres, 6x6 grid,
                  top-k=8, auxiliary loss, trainable encoder tail
```

The fresh optimizer is an unavoidable, declared difference. B40 serialized its
own optimizer/scaler state in `recovery_latest.pt` for reproducibility, but the
fixed scientific endpoint remains exactly one additional epoch.

## Fixed B40 contract

- Parent: completed B37 checkpoint, immutable fixed E2 only.
- Parent checkpoint SHA-256:
  `4f208674b0fd27be21232088154bed8b338d4f26be5332ab52e0f0541b6cceb9`.
- Supervision: B37 all-target B6-preserved LLM-fill export.
- Training population: 4,349 report-only studies.
- MRI exposure: 24,035 eligible series.
- Supervision cells: 34,010.
- Image path: B37 90% native centre crop, one antialiased 448 resize, 32 centres.
- Model: B37 sparse-MIL, 6x6 grid, top-k 8, one trainable encoder-tail stage.
- Optimizer: fresh AdamW.
- Head LR: `1e-4`.
- Encoder-tail LR: `5e-6`.
- Weight decay: `1e-4`.
- Gradient clip: `1.0`.
- Duration: exactly one additional full-coverage epoch; no checkpoint selection.
- Expert labels: zero expert studies/labels in gradients, stopping, or selection.

## Completed training audit

The continuation completed normally and covered the complete declared training
surface:

```text
fixed endpoint                 true
completed absolute epochs      3
additional B40 epochs          1
batches in E3                  2,175
studies in E3                  4,349
series in E3                   24,035
supervision cells in E3        34,010
gold studies used in gradient  0
base 448 reconstruction error  0.0
optimizer                      fresh AdamW from B37 E2 weights
```

The trainable encoder tail changed from the B37 E2 fingerprint

```text
078b69f49a34a8e6acf9edd6c8aa6dc456ce1d0b70d59862d014ce8e5aa1ef20
```

to the B40 E3 fingerprint

```text
8c5d51083866e18e225bb20f114f335f4bb39f29cd47bd47425c86da514bd3c2
```

so B40 genuinely updated the representation rather than only changing the
residual head.

### Training trajectory

| Endpoint | Total loss | Combined loss | Local auxiliary loss | Mean abs(gate) | Max abs(gate) |
|---|---:|---:|---:|---:|---:|
| B37 E1 | 1.1341568780 | 0.5439505712 | 0.5902063080 | 0.020121 | 0.035931 |
| B37 E2 | 1.0514200304 | 0.5305002703 | 0.5209197600 | 0.050247 | 0.086884 |
| B40 E3 | 1.0228941822 | 0.5218335125 | 0.5010606702 | 0.081500 | 0.148869 |

From B37 E2 to B40 E3, total loss fell by about 2.7%, combined loss by about
1.6%, and local auxiliary loss by about 3.8%. Thus the negative downstream result
is not an optimization failure: the model continued fitting the report-derived
training objective.

Peak runtime memory remained healthy during E3:

```text
host RSS peak          about 8.43 GiB
CUDA allocated peak    about 5.52 GiB
CUDA reserved peak     about 10.71 GiB
```

## Expert-58 result

Expert-58 is a repeatedly reused development diagnostic, not independent test
evidence. B40 was evaluated once at its fixed E3 endpoint against the immutable
B37 E2 parent using the same three-offset `[-1, 0, +1]` 448 sparse-MIL path.

### Primary comparison

| Metric | B37 E2 | B40 E3 | B40 - B37 |
|---|---:|---:|---:|
| Global 448 macro AUC | 0.6794831901 | 0.6761350965 | -0.0033480935 |
| Combined 448 macro AUC | 0.6858177916 | 0.6847721365 | **-0.0010456551** |
| Focal-six mean AUC | 0.5841648772 | 0.5854404368 | **+0.0012755596** |

The historical 224 full-fill base replay remained exactly consistent at
`0.6686507523` macro AUC.

### Paired bootstrap: B40 combined minus B37 combined

```text
median difference      -0.0009974444
95% CI                  [-0.0128037347, +0.0096924830]
P(B40 > B37)            0.429
valid replicates        5,000
```

The interval spans zero broadly and the probability of a positive B40 difference
is below 0.5. There is no evidence that the extra epoch improves the overall
Expert-58 endpoint.

### Per-target Expert-58 AUC

| Target | B37 E2 | B40 E3 | B40 - B37 |
|---|---:|---:|---:|
| ACL | 0.479167 | 0.474265 | -0.004902 |
| MCL | 0.426304 | 0.448980 | +0.022676 |
| Medial Meniscus | 0.781250 | 0.768029 | -0.013221 |
| Lateral Meniscus | 0.678261 | 0.686957 | +0.008696 |
| Medial OA | 0.815504 | 0.812403 | -0.003101 |
| Lateral OA | 0.682785 | 0.659574 | -0.023211 |
| PF OA | 0.711712 | 0.710425 | -0.001287 |
| Effusion | 0.853416 | 0.858385 | +0.004969 |
| Synovitis | 0.817204 | 0.816010 | -0.001195 |
| Baker's | 0.844203 | 0.847826 | +0.003623 |
| Contusion | 0.533063 | 0.534413 | +0.001350 |
| Fracture | 0.606944 | 0.600000 | -0.006944 |

The changes are mixed rather than coherent. The largest positive target movement
is MCL (`+0.0227`), while Lateral OA (`-0.0232`) and Medial Meniscus (`-0.0132`)
move in the opposite direction. These reused target-level diagnostics are not a
basis for target-wise hybridization.

## Interpretation

B40 answers the duration question cleanly enough for the current development
program: one additional optimizer-reset epoch continues to reduce the weak-label
training objective but does not improve the reused Expert-58 overall endpoint.
The 448 global branch actually falls by about `0.00335` macro AUC, while the
combined sparse-MIL prediction falls by only about `0.00105`; the local residual
therefore partly compensates for the degradation in the global representation,
but not enough to produce an overall gain.

This pattern is consistent with B37 E2 already being near the useful stopping
point for this weak-supervision trajectory. It does not prove classical
overfitting on an independent validation set, because Expert-58 is reused, but
it provides no prospective justification for extending B40 to E4.

## Verdict

```text
B40 status              completed_not_promoted
B40 vs B37 macro delta  -0.0010456551
B40 focal-six delta     +0.0012755596
bootstrap 95% CI        [-0.0128037347, +0.0096924830]
P(B40 > B37)            0.429
```

Decision:

- retain B40 and all artifacts as a completed duration experiment;
- do not replace B37 with B40;
- do not run E4 as a post-hoc duration selection step;
- do not construct target-wise B37/B40 hybrids from Expert-58;
- prioritize independently motivated candidates such as B41 for further hidden
  competition evidence.

A B40 hidden submission is not the preferred use of a limited submission budget.
If ever submitted, it must remain the exact frozen E3 checkpoint and be recorded
as a separate experiment; it must not retroactively modify B37 or B39.

## Artifacts

```text
runs/075_Experiment_B40_b37_e2_optimizer_reset_continuation/
└── b40_b37_e2_continuation/
    ├── b40_model.pt
    ├── recovery_latest.pt
    ├── history.json
    ├── training_audit.json
    ├── preflight.log
    ├── expert58.log
    └── expert58/
        └── b40_vs_b37_expert58.json
```

B40 final checkpoint SHA-256:

```text
68617230aba2a7ddb831b775b007bb76ca2fdf6cfe6c298fcbab7cccce2fee89
```
