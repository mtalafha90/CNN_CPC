# B20 causal occlusion diagnostic

> **Status — 2026-08-13:** IMPLEMENTED. Diagnostic only; no training and no model-selection claim.

B20 remains the active working model. This diagnostic tests whether a spatial region is causally important to one target prediction by replacing local MRI content with a blur of the same image and measuring the target probability change.

```text
delta-p = baseline probability - perturbed probability
```

Positive `delta-p` means that removing the region lowered the target probability, so the region provides supportive model evidence. Negative `delta-p` means that removing the region increased the probability.

The method deliberately avoids black occlusion rectangles, which could create an artificial out-of-distribution feature. The default patch is `28 x 28`, stride `14`, with an odd local blur kernel of `15` pixels.

For efficiency, unchanged frozen-encoder slice features are cached and only the perturbed token(s) are re-encoded. The diagnostic requires the cached-feature baseline to reproduce direct B20 inference within a small numerical tolerance before writing results.

Two scopes are available:

- `slice`: perturb only the selected 2.5D token. Use this first for a specific displayed MRI slice.
- `series`: apply the same spatial patch across all sampled tokens in the selected MRI series. Use this as a robustness follow-up when slice-level evidence is diluted by redundancy across neighboring slices.

Outputs include a six-panel figure, patch-level CSV, signed occlusion map, coverage map, JSON summary, maximum probability drop, mean absolute perturbation effect, Grad-CAM/occlusion correlation, and top-20% overlap.

This is a perturbation-based model-dependence diagnostic. It is **not** lesion segmentation, does not use pixel-level ground truth, and is not independent validation.

Run through the module entry point:

```bash
python -m rsna_knee.b20_occlusion --help
```
