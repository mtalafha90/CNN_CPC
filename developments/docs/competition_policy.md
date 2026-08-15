# Competition execution policy

This file records the execution and model-development policy enforced by the repository.

> **Snapshot — 2026-08-12.** Package `0.24.1`. B13 remains the reused-gold development champion at `0.6293565948`. B15 passed the frozen weak-v2 gate but reached `0.6209002783` on its one-look reused-gold confirmation and did not replace B13.

## Conservative defaults

```yaml
requested_gpus: 1
runtime_budget_hours: 8.5
runtime_reserve_minutes: 10
pretrained: false
allow_external_pretrained: false
```

Ordinary competition-only experiments remain on this path. External pretrained weights require explicit experiment-specific opt-in.

## External pretrained-model policy

The competition rules supplied by the repository owner were reviewed before B13. Under the supplied External Data and Tools language, publicly/equally accessible external models were permitted absent a competition-specific prohibition.

B13 and B15 therefore explicitly opt into:

```yaml
allow_external_pretrained: true
pretrained: true
```

B13 initialization:

```text
torchvision ConvNeXt-Tiny IMAGENET1K_V1
standard ImageNet mean/std normalization
```

B15 starts from the same ImageNet protocol and then adds competition knee-MRI same-study contrastive SSL.

B0-B12.1 remain historical competition-data-only paths unless their own documents say otherwise.

## Frozen B6 supervision policy

```text
version                     1.2.1
confidence threshold        0.75
positive target / weight    0.85 / 0.50
negative target / weight    0.05 / 1.00
uncertain/unmentioned       ignored
gold rows in weak export    0
```

B6 v1.2.1 remains frozen. Any new state treatment must receive a separate experiment/version.

Do **not** infer that unmentioned report states are negatives.

## Reused-gold development policy

The 58 fully labelled studies have supported repeated sequential development. They are therefore development/model-selection data, not independent validation.

Do not:

- tune ensemble weights on the 58 gold labels;
- select target-specific model winners;
- retune B6 parser rules/weak-label weights from downstream gold outcomes;
- tune B13/B14/B15 architecture, normalization, learning rate, epoch count or TTA from the same gold surface;
- call a local development result a leaderboard result.

## Frozen all-series surface

Historical B12-B14 full B6-active surface:

```text
active studies             3120
B6 supervised cells      14123
positive / negative    6871 / 7252
eligible real MRI series 17475
series mapping SHA
5c4bb1c52294e45f9e83274c5c07d198dc54811c49b96111b7c8439bd7bcd376
```

B13 refuses mapping drift.

## Frozen weak-v2 policy

B15-era model ranking uses:

```text
surface                 weak_b6_holdout_v2
weak-train studies      2497
holdout studies          623
holdout cells           2875
report-group overlap       0
manifest SHA
1a1b07bd690bae3cbb945773c4fcb1c3b0d0f6aa1dd18649d62859aeeb4603d1
```

The split was frozen before B15/control training using no gold labels and no model predictions. Do not regenerate it based on performance.

Any model evaluated on weak-v2 must exclude all holdout UIDs from downstream training. For B15 SSL, both gold images and weak-v2 holdout images were also excluded.

Weak-v2 measures **B6 teacher agreement, not expert truth**.

## Strict weak-v2 bootstrap

```text
sample studies with replacement
-> compute all 12 target AUCs
-> reject replicate if any target is undefined
-> macro = mean of exactly 12 AUCs
```

This keeps the estimand fixed despite rare classes.

## B15 predeclared gate

B15 proceeded to one reused-gold confirmation only if:

```text
raw B15-control weak macro delta > 0
paired median delta > 0
P(B15 > control) >= 0.95
```

Observed:

```text
control weak-v2      0.5652498118
B15 weak-v2          0.7319060415
raw delta           +0.1666562297
paired median       +0.1675245839
95% paired CI       [+0.1124433208,+0.2165156305]
P(B15 > control)     1.0000
gate                 PASS
```

B15 then received exactly one reused-gold evaluation:

```text
B15 gold             0.6209002783
B13 gold             0.6293565948
```

No B15 retuning from that result is permitted under the original experiment identity.

## TTA policy

B12-B15 relevant development evaluation uses frozen center offsets:

```text
[-1,0,1]
```

Diagnostic center-only evaluation is not permission to retune TTA after reading gold labels.

## Current next-stage policy

The next step is a **diagnostic B6 report-state audit**, not a B15 hyperparameter search. The audit may inspect `positive`, `negated`, `uncertain`, and `unmentioned` states against the already-reused gold truth.

If it motivates a new supervision rule, that rule must be separately named/versioned and frozen before model evaluation. B6 v1.2.1 remains unchanged.

## Submission and reporting vocabulary

Use these terms accurately:

- **audit** — data/routing/supervision inventory;
- **training run** — optimization result;
- **weak-v2 result** — B6 teacher-agreement ranking result;
- **gold development result** — score on the reused 58-study expert-labelled surface;
- **paired bootstrap** — aligned resampling comparison;
- **leaderboard result** — actual competition submission score.

If an external pretrained model is part of a final competition solution, record its exact source/version and preserve all required licensing/reproducibility documentation.

The hidden Kaggle evaluation remains the next genuinely independent performance signal.