# B50 redesign — train the part of the model that produces the score

**Date:** 2026-08-29
**Status:** PROTOCOL / NOT IMPLEMENTED / NOT RUN.

This replaces the ordered-slice-sequence design as B50's subject. That design,
even amended, refines the local sparse branch. This document argues the local
branch is the wrong target and states what should be trained instead.

## The measurement that forces the redesign

At `b37_highres_sparse_mil.py:313-315`, every B37-descended model does this
before training begins:

```python
for name, parameter in self.base.named_parameters():
    if not name.startswith("encoder."):
        parameter.requires_grad_(False)
```

So the whole B34 study hierarchy — the slice-pooling gate, the two-layer study
Transformer, the twelve pathology queries — is frozen. What actually trains is
the encoder's final stage and a sparse head. And the sparse head's output is
admitted to the score through a gate the completed runs measured at

```text
|tanh(g)| mean 0.022 (B49)   0.025 (B48)
```

Putting those together, for every submission this project has made since B37:

```text
z = z_B34 + tanh(g) * z_local
    ~98%          ~2%
```

**Roughly 98% of every 0.714 comes from a hierarchy that has not received a
gradient since B34.** It was trained at 224 pixels on globally pooled features,
in the Phase-9 era, before the resolution change, before the LLM-fill
supervision was finalised, and before every experiment from B37 onward.

B37 raised the input to 448 and fine-tuned the encoder tail so the features
changed. **The consumer of those features did not move with them.** No
experiment has ever adapted the hierarchy to the representation it is now fed.

That is also the structural reason B43, B44, B45, B48 and B49 all returned
nulls. They were refining the 2%, and their endpoints could not resolve the
result — B48's and B49's measurements were capped at 0.0015 and 0.0024 against a
predeclared `+0.010` bar. Five experiments, one frozen 98%.

## The question

> Does adapting the frozen B34 study hierarchy to the 448-pixel encoder
> representation improve ranking, when nothing else changes?

This is not another crop, tile, resolution, centre-count, top-k, plane-router,
gold-weight or query-rank variant. It is the first experiment in this line to
train the path that produces the prediction.

## Arms

Both inherit B42's complete data and optimisation contract, and both start from
the same base checkpoint.

| Arm | Hierarchy | Everything else |
|---|---|---|
| `frozen_hierarchy_control` | `requires_grad_(False)`, exactly as B37–B49 | identical |
| `adapted_hierarchy_candidate` | slice-pooling gate, study Transformer and pathology queries receive gradients at `hierarchy_lr_scale` × head LR | identical |

The control is a reproduction of the current endpoint, so a difference is
attributable to one change and nothing else.

`hierarchy_lr_scale` must be frozen before any result is seen. A principled
starting value is the encoder tail's own scale, `0.05` — the hierarchy is
pretrained and large relative to the data, so it should move slowly. **One
value. No sweep.** The reason to pick the encoder-tail scale rather than invent
one is that it is the only reduced learning rate this project has already run
successfully at 448.

## The risk, stated plainly

The hierarchy is roughly 46.7M parameters and the training population is 4,349
studies with report-derived labels whose positive precision is about 69%.
Unfreezing it may simply overfit noisy labels faster. B40 is the precedent: an
extra epoch lowered the weak-supervision objective while expert AUC fell.

Three things bound that risk, and none of them is a sweep:

- the fixed two-epoch endpoint with no checkpoint selection, unchanged;
- the reduced learning rate, frozen in advance;
- the control arm, which makes overfitting visible as a divergence between the
  weak-objective loss and the held-out ranking rather than as an unattributable
  null.

If the candidate's training loss falls below the control's while its held-out
ranking does not improve, that is B40's signature and the honest reading is that
the label surface, not the frozen hierarchy, is the binding constraint.

## Endpoints

**Primary.** Candidate minus control macro ROC AUC on the fresh B50
scanner-grouped `validation_unseen_scanners` surface, all 12 targets.

**Secondary, reported always, never a selector.** Expert-58 macro AUC and the
focal six, as a post-training audit; the seen-scanner delta as a domain-gap
safety check; the weak-supervision training loss of both arms, because the
overfitting signature above is only visible if it is recorded.

**Mandatory power check, before any verdict.** Report the discordant-pair
fraction between the two arms on the primary surface alongside the threshold.

```text
if ceiling < threshold    the endpoint could not have detected the effect;
                          record `endpoint_underpowered`, NOT `no_support`
```

This clause is carried over from the B50 amendment. B48 and B49 were both filed
as `no_support` under thresholds their measurements could not reach, and that
must not happen again. Unfreezing the hierarchy changes the base logits
directly, so unlike its predecessors this experiment is *expected* to clear the
ceiling comfortably — but it must be checked, not assumed.

**Supported** only if all hold:

```text
unseen delta >= +0.010
paired 95% CI lower bound > 0
P(candidate > control) >= 0.95
at least 7 of 12 targets improve
every leave-one-target-out macro delta > 0
seen-scanner delta >= -0.005
discordant-pair ceiling >= +0.010
```

## Why this before the other candidates

Ranked by evidence per GPU hour, with what is already known:

1. **This experiment.** Targets 98% of the score. Never attempted. One run.
2. **Class balancing.** There is no `pos_weight` and no focal loss anywhere in
   the repository, while report positives carry weight `0.50` against `1.00` for
   negatives *and* a softened `0.85` target — discounted twice, on the class that
   determines a ranking metric. The closest published system on this task uses
   per-class `pos_weight` on every branch head and focal loss on its fusion
   head. Two independent lines of evidence, one cheap change. This should be
   B51, on its own.
3. **Augmentation.** A full laterality-safe augmentation suite exists at
   `dataset.py:128-185`, is configured in the B42 YAML, and is unreachable
   because `train=False` is hard-coded twice. Horizontal flip must stay off —
   four targets are laterality-specific — but the jitter that is already written
   is free. Fold into B51 or run separately.
4. **The gate itself.** It was still nearly tripling between epoch 1 and epoch 2
   when fixed-E2 training stopped, so it has never settled. Worth its own
   protocol, and explicitly not folded into this one.

## Implementation notes

`b37_highres_sparse_mil.py` must not be modified — B42, B46, B48 and B49 are
completed frozen experiments that import it. B50 subclasses
`B42ConstantAreaAspectSparseMILResidual` and overrides the constructor's freeze
loop, exactly as B47 subclasses rather than edits.

The parameter count receiving gradients must be recorded in the training audit
for both arms, and the control's must equal B42's exactly. That is the check
that the control really is a reproduction.

Still to write: the model subclass, the training entry point, the config, and
the fresh selection gate (the existing
`developments/scripts/prepare_b50_ordered_slice_gate.sh` builds a split that is
reusable here, since the split is about scanners and not about the mechanism).
