# Competition execution policy

This file records the execution and model-development policy enforced by the repository.

> **Snapshot — 2026-08-11.** Package `0.21.0`. B7.1 remains the retained benchmark at `0.5644802945`; B12 has the highest development point estimate at `0.5660915179` but is statistically tied with B7.1. B12.1 is the competition-only hierarchical series-token experiment. B13 is the clean standalone ImageNet encoder-protocol experiment.

## Conservative defaults

```yaml
requested_gpus: 1
runtime_budget_hours: 8.5
runtime_reserve_minutes: 10
pretrained: false
allow_external_pretrained: false
```

These defaults keep ordinary experiments on the competition-data-only path. External pretrained weights must be explicitly enabled by an experiment-specific config.

## Verified external-data / pretrained-model rule

The official competition rules supplied by the repository owner were reviewed on 2026-08-11. The Competition-Specific Rules section **External Data and Tools** permits data other than Competition Data when it is publicly available/equally accessible at no cost or satisfies the stated reasonableness criteria. The same section states that external data and models are acceptable unless specifically prohibited by the Host.

No competition-specific prohibition on publicly available pretrained models was present in the supplied rules. Therefore a standard publicly accessible torchvision ImageNet checkpoint is permitted under the supplied rules, subject to the remaining competition obligations on accessibility, licensing, reproducibility and winner documentation.

B13 is the first experiment that opts in:

```yaml
allow_external_pretrained: true
pretrained: true
```

The existing B0-B12.1 paths remain competition-data-only and reproducible.

## B12.1 / B13 separation

B12.1 is explicitly competition-only:

```text
trainer       rsna-knee-b12-1
initialization B5 competition-only SSL
external_pretrained false
```

Its trainer rejects `pretrained: true` or `allow_external_pretrained: true`.

B13 is explicitly external-pretrained:

```text
trainer       rsna-knee-b13
initialization torchvision ConvNeXt-Tiny IMAGENET1K_V1
normalization standard ImageNet mean/std
external_pretrained true
```

There is no B5 checkpoint argument in B13.

## Frozen report-supervision policy

B6 v1.2.1 is frozen:

```text
confidence threshold        0.75
positive target / weight    0.85 / 0.50
negative target / weight    0.05 / 1.00
uncertain/unmentioned       ignored
gold rows in weak export    0
```

Later experiments use no gold labels in gradients or early stopping. Because the B6 gold audit informed the global weak-label policy and the same 58 studies have been reused for model development, later scores are development/model-selection estimates.

## Gold-development policy

Do not:

- tune ensemble weights on the 58 gold labels;
- select target-specific model winners;
- retune B6 parser rules or weak-label weights from later gold outcomes;
- tune B12/B12.1 series caps or pooling heads from gold;
- tune B13 ImageNet variants, normalization, learning rate or epoch count from the same 58-study result;
- call a development result a leaderboard result.

## Frozen B12/B12.1/B13 series surface

```text
active training studies      3120
B6 supervised cells         14123
positive / negative       6871 / 7252
eligible real MRI series    17475
series mapping SHA-256
5c4bb1c52294e45f9e83274c5c07d198dc54811c49b96111b7c8439bd7bcd376
```

B13 must use the same series policy and refuses mapping drift.

## TTA policy

B12, B12.1 and B13 development evaluation use the frozen center offsets:

```text
[-1, 0, 1]
```

Diagnostic center-only evaluation is not permission to retune TTA after reading gold labels.

## Submission and winner obligations

Final competition inference must preserve the competition submission schema and all applicable documentation/reproducibility requirements. If an external pretrained model is part of a final winning solution, record its exact source/version and ensure the full final software/model pipeline can be reproduced as required by the competition rules.

## Reporting vocabulary

Use these terms accurately:

- **audit** — data/routing/supervision inventory;
- **training run** — optimization result;
- **gold development result** — score on the reused 58-study development surface;
- **paired bootstrap** — aligned resampling comparison between model prediction files;
- **leaderboard result** — actual competition submission score.
