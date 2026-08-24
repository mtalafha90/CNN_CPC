# Post-B45 plateau retrospective — where the current line stopped improving

**Date:** 2026-08-24

**Purpose:** freeze the evidence available after B45, identify which assumptions were wrong or weakly supported, and redirect the next experiments away from further small variants of the B37/B42 sparse-MIL line.

This document is a retrospective diagnosis, not a rewrite of the frozen experiment records. Earlier protocol documents remain historically correct for what was known when each experiment was declared.

## Current independent scoreboard

The currently observed Kaggle scores are:

```text
B37  direct-square 448 sparse MIL                 0.714
B41  aspect-preserving 448 square-pad sparse MIL 0.714
B42  constant-area native-aspect ragged sparse MIL 0.714
B45  plane-calibrated sparse MIL                  NOT SUBMITTED by explicit decision
```

Kaggle displays scores rounded to three decimals, so the three `0.714` values are a displayed tie, not proof of identical unrounded AUC.

The important scientific fact is that materially different in-plane geometry choices did not produce a visible hidden-test separation.

## Late-line local diagnostics

```text
                         Expert-58 macro    focal-six
B37 combined               0.685818          0.584165
B41 combined               0.677872          0.567447
B42 combined               0.683120          0.580098
B45 combined               0.679176          0.579334
```

B45 minus B42:

```text
macro point delta          -0.003944
bootstrap median           -0.003464
95% CI                     [-0.014613, +0.003548]
P(B45 > B42)                0.1886
```

B45 minus B37:

```text
macro point delta          -0.006641
bootstrap median           -0.006209
95% CI                     [-0.015937, +0.000392]
P(B45 > B37)                0.0346
```

The B45 ACL result moved in the wrong direction (`0.47549 -> 0.46201` versus B42), while Lateral Meniscus (`+0.01366`) and Contusion (`+0.00675`) improved. The router remained close to uniform, so B45 mainly tested independent per-plane top-k quotas followed by near-uniform fusion rather than strong target-specific plane specialization.

## What the project did correctly

Several controls were unusually strict and should be preserved:

- expert labels were kept out of gradients for the B20-B45 development lineage;
- fixed endpoints and fixed seeds prevented post-hoc epoch selection;
- B40 demonstrated that lower weak-label training loss does not imply higher expert AUC;
- B43/B44 separated routing/coverage questions without modifying checkpoints;
- B41/B42 isolated in-plane geometry more cleanly than B37;
- hidden-safe B41 resubmission demonstrated that the earlier hidden crash was operational, not scientific;
- paired study bootstraps and leave-one-target-out audits exposed many apparent aggregate gains as target-specific effects.

The plateau is therefore not mainly caused by careless bookkeeping. It is caused by the scientific line repeatedly probing mechanisms that are smaller than the dominant error sources.

# Where the line went wrong

## 1. The 58-study expert set became an adaptive design surface

The repository itself already recognizes this. The same 58 studies have been used repeatedly across many architecture and mechanism decisions. Keeping them out of gradients prevents direct training leakage, but it does **not** prevent adaptive overfitting of hypotheses to the same cases.

B43 -> B44 -> B45 is the clearest late example. B43 found an interpretable ACL sagittal signal and axial selection enrichment. B44 correctly ruled out simple center-density failure. B45 then encoded a plane-routing hypothesis motivated by those reused observations. The fixed B45 implementation remained formally prospective, but the scientific hypothesis was still chosen after repeated inspection of the same 58 cases. B45's negative ACL movement is exactly the kind of regression expected when an appealing small-sample mechanism does not generalize.

**Correction:** no more architecture should be invented from target-level Expert-58 patterns. Expert-58 can remain a descriptive OOF or post-training audit only.

## 2. We optimized report-derived supervision long after evidence showed objective mismatch

B40 is a key result: from B37 E2 to B40 E3, total, combined and local weak-supervision losses all decreased, the encoder fingerprint moved, yet Expert-58 macro AUC fell. This is not an optimization failure. The model became better at the training objective without becoming better at the expert target.

Phase-9 v2 made the same point from the supervision side. Added translated/rescued supervision produced heterogeneous effects: Contusion improved strongly, Effusion worsened, and removing Contusion flipped the macro sign. Earlier B25X was overwhelmingly driven by Synovitis. These are signs that the report-derived target process and the expert image-label target are not interchangeable.

The project continued to spend most later compute on architecture while holding this imperfect objective nearly fixed.

**Correction:** the next experiment must address source-label mismatch directly before another pooling or geometry variant.

## 3. We underused the only 696 official clean target cells

There are `58 x 12 = 696` official expert-labelled cells. The development line deliberately kept all 58 studies out of gradients so that they could serve as a validation surface. That was defensible early. After dozens of adaptive looks, however, their value as an untouched validation set is already largely spent, while their value as the only clean competition-target supervision remains unused.

For competition performance, permanently refusing to train on the clean labels is now a larger cost than the validation benefit they still provide.

**Correction:** move to prospective cross-fitted use of the 58 gold studies. Every gold study can still receive an out-of-fold prediction from a model that did not train on that study, while the other folds provide clean target anchoring.

## 4. B37 was a successful model but a poor causal experiment

B37 jointly changed several axes: resolution, native-order crop, slice count, local grid density, sparse local pooling, auxiliary loss, and encoder-tail adaptation. Its hidden `0.714` is valuable, but the gain cannot be assigned to one mechanism.

B38 later showed that 448 global-only tail adaptation was not enough (`0.66441` versus historical base about `0.66875`). B40 showed that longer fitting was not enough. B41/B42 showed that the exact in-plane aspect policy did not produce visible hidden separation. The late ablations helped, but they were reconstructing causal understanding after a joint intervention had already become the champion.

**Correction:** future experiments should change one high-level capability at a time and include a matched control from the start.

## 5. We kept modifying the sparse residual after it had shown itself to be a small correction

In B37 and B41 the local sparse residual added only about six thousandths of Expert-58 macro AUC over the corresponding global branch. B45 changed the routing logic substantially but still moved overall performance by only a few thousandths.

The repeated result is that this branch is a **small residual corrector**, not the dominant study representation. Spending B43-B45 primarily on how its top-k evidence is selected was therefore unlikely to create a large step.

**Correction:** close the B35/B36/B37 sparse-MIL family as the primary development direction. It can remain as a component or comparator, but B46 should not be another top-k/grid/router variant.

## 6. B45 handled plane identity too late and too statically

B45 gives each target three learned scalar router logits. Those weights are study-independent apart from missing-plane masking. The actual image features from sagittal, coronal and axial sequences do not interact before the per-plane logits are formed.

That is much weaker than the multi-sequence problem radiologists solve. A plane can be useful for one patient because it confirms, disambiguates or localizes evidence in another plane; a fixed target-level scalar cannot express that interaction.

The completed B45 router staying near one-third is therefore not evidence that plane information is unimportant. It is evidence that a static late scalar router is not enough.

**Correction:** if plane/sequence modeling is revisited, use feature-level cross-series/cross-sequence attention with study-dependent weights, not fixed target-plane priors.

## 7. We sampled depth but did not truly model volume continuity

B37/B42 use 32 deterministic 2.5D centers and local spatial tokens. B44 showed that simply doubling centers from 32 to 64 did not fix the weak targets. That rules out *coverage density* as the main explanation.

It does **not** rule out through-plane reasoning. ACL morphology, meniscal tears, fractures and contusions can require continuity or shape changes across adjacent slices. Top-k pooling treats strong locations mostly as unordered evidence. A 2.5D triplet gives short context but does not build a true sequence representation across the full acquired volume.

The older B31 depthwise `Conv1d(k=3)` is not a counterexample: its direct inference perturbation was extremely small and it operated on an earlier one-token pathway. It was not a full slice-transformer or 3D sequence model.

**Correction:** the next representation experiment should test explicit within-series slice sequence modeling, not denser sampling.

## 8. The report representation experiment was much weaker than its name suggests

B16 was called full-report semantic alignment, but its text representation was word TF-IDF -> truncated SVD. It did not use a pretrained clinical or multilingual language encoder. Later audits established that the dataset is genuinely multilingual and contains Greek/Cyrillic text.

Thus the project used rich reports mainly in two compressed forms:

1. discrete parser/LLM-derived target states;
2. a bag-of-ngrams low-rank vector for image-report alignment.

This almost certainly leaves semantic, negation, morphology and cross-lingual information unused.

**Correction:** a later representation experiment should use a frozen, publicly accessible multilingual text encoder or verified translate-then-encode teacher during training, with MRI-only inference. This should be treated as a new experiment, not as 'B16 again'.

## 9. The validation design is not explicitly site-held-out despite a multicenter task

The challenge dataset is multicenter. The prospective weak splits were frozen by UID hashing, which is good against label-based split manipulation, but does not create a scanner/site-domain holdout. A model can therefore look strong while exploiting acquisition/site regularities that remain on both sides of the split.

**Correction:** create a prospective site/scanner-grouped validation surface before the next major representation family. It will still be weak-label validation, but it will measure a different failure mode: domain generalization.

## 10. Single-seed fixed endpoints protected governance but limited resolution

The fixed-seed/fixed-E2 policy prevented p-hacking and was appropriate. However, when effects are only `0.002-0.008`, a single training realization cannot separate architecture effect from optimization variance. The project often interpreted tiny point differences mechanistically even while correctly refusing promotion.

**Correction:** for any new high-cost mechanism that passes its first gate, use a small predeclared seed replication (for example 3 seeds) on the powered non-gold development surface before spending a hidden submission. Do not seed-sweep and select the best; report the mean and dispersion.

# External evidence that changes the next direction

The 2026 RSNA challenge is explicitly built from more than 5,000 knee MRI exams from 19 sites with reports in about a dozen languages, making both domain shift and multilingual supervision first-order issues.

The 2024 CoPAS study is unusually relevant because it also predicts twelve knee abnormalities. It reports overall AUC `0.812` on 1,748 subjects and uses feature-level co-plane and cross-sequence attention rather than late scalar plane weights. Its method includes three plane branches, 3D encoding, cross-plane spatial attention, cross-sequence integration and a plane-aware abnormality integration stage. Importantly, its AUC falls to roughly `0.721-0.726` on external datasets with different sequence settings, demonstrating how large the domain/sequence shift can be.

MRNet is older but reinforces two points: different tasks can prefer different MRI series/planes, and series-level evidence should be learned before exam-level fusion. Its ACL model is strongly associated with sagittal information rather than a universal all-plane representation.

These papers do not prove that their architectures will transfer to this competition, but they point to a capability missing from the current B37/B42 line: **feature-level interaction across the acquired volume and across MRI sequences**.

References:

- Bien et al. 2018, MRNet, PLOS Medicine. DOI: `10.1371/journal.pmed.1002699`.
- Qiu et al. 2024, *Learning co-plane attention across MRI sequences for diagnosing twelve types of knee abnormalities*, Nature Communications 15, 7637. DOI: `10.1038/s41467-024-51888-4`.
- RSNA 2026 Knee MRI AI Challenge official page: `https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge`.

# Next experiments

## B46 — gold-anchored cross-fitted supervision test (highest priority)

**Question:** is the current ceiling primarily caused by report-to-expert target mismatch rather than insufficient model capacity?

Do **not** change the image architecture first. Use the strongest stable parent family and add clean official supervision in a prospectively cross-fitted way.

Proposed design:

```text
parent image/model contract     freeze before any new gold OOF result
weak studies                    4,349 report-only
clean studies                   58 official gold
folding                         5-fold multilabel-stratified, with site/scanner grouping when feasible
for each fold                   train on 4,349 weak + ~46-47 gold
                                predict only the held-out ~11-12 gold studies
clean-label weight              one fixed predeclared value; no sweep
checkpoint selection            none
architecture changes            none
output                          58-study OOF prediction, every row from a model that did not train on that row
```

The clean label weight must be chosen before OOF results. A principled starting rule is to make one gold cell contribute several times more than a weak report cell, reflecting source quality, but not enough for 58 studies to dominate the 4,349-study representation. The exact value must be frozen before running B46.

Decision logic:

- if cross-fitted B46 clearly improves the frozen parent across macro AUC and does not collapse into one target, label mismatch is a confirmed bottleneck and the final B46 model can be trained once on all 58 gold studies;
- if it does not, stop spending effort on gold anchoring and move to representation B47.

This experiment directly tests the largest unresolved assumption in the current pipeline.

## B47 — explicit slice-sequence model

Only after the B46 supervision question is answered, test a genuinely new image capability.

Recommended architecture:

```text
native-aspect B42-style series geometry
-> per-slice / 2.5D ConvNeXt features
-> ordered slice tokens with physical/normalized position
-> lightweight 1D Transformer or state-space/temporal block within each series
-> one or several series tokens retaining sequence/plane metadata
-> pathology queries over all series tokens
-> 12 logits
```

Critical difference from B44: center count stays fixed. B47 tests **ordered depth relationships**, not more samples.

Critical difference from B45: sagittal/coronal/axial information interacts through features and study content; there are no static target-plane scalar priors.

Start with a matched control that replaces only the within-series unordered pooling while keeping the downstream study hierarchy and supervision fixed.

## B48 — cross-sequence feature interaction, only if B47 justifies the representation family

If explicit slice modeling produces a real gain, add study-level cross-series/co-sequence attention. Use plane, fluid sensitivity, fat suppression and parsed sequence text as metadata tokens, but let pathology queries attend to image features dynamically.

Do not preassign 'ACL=sagittal' or similar clinical weights. The model should learn study-dependent interactions.

## Later: modern multilingual report-teacher alignment

A modern report-teacher experiment is also high-value, but it should not be mixed into B46/B47 because that would again confound supervision and representation.

A future arm should replace B16's TF-IDF/SVD teacher with a frozen public multilingual/clinical text representation (or verified translation followed by a strong text encoder), then align MRI study features with report semantics while keeping test-time inference MRI-only.

# What should stop now

Do not spend another experiment on:

```text
448 vs another nearby square resolution
32 vs 48 vs 64 deterministic centers
another top-k value
another 6x6-grid variant
another static plane-router temperature
another crop fraction near 90%
B37/B42 extra epochs
post-hoc per-target model mixing from Expert-58
another backbone swap without a task-specific representation hypothesis
```

Those axes have either been directly tested, indirectly bounded, or are smaller than the current measurement/label mismatch.

# Recommended immediate sequence

```text
1. Close and archive B45.                         DONE
2. Stop using Expert-58 to invent architectures. NOW
3. Freeze the B46 cross-fitted gold-anchor protocol.
4. Run B46 OOF before any new image architecture.
5. In parallel, audit site/scanner/sequence groups and freeze a domain-shift validation policy.
6. If B46 is positive, train one all-gold anchored final model.
7. If B46 is negative or small, implement B47 explicit slice-sequence modeling.
8. Require a materially larger effect before another hidden submission.
```

The central conclusion is that the project did not fail because it needed one more small pooling trick. It plateaued because the late experiments were optimizing second-order architecture details while the first-order problems are **target-source mismatch, adaptive validation, domain/sequence shift, and insufficient explicit volumetric/sequence reasoning**.
