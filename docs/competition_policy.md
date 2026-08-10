# Competition execution policy

`docs/competition.md` is the preserved competition-description document. This file describes the conservative execution policy enforced by the current code and experiment workflow.

> **Snapshot: 2026-08-10.** Package `0.14.0`. B7.1 is the current best standalone development model at `0.5644802945`; B8 is rejected at `0.5300962807`; B9 strict semantic routing is the active predeclared experiment. Current scores are in [`EXPERIMENT_STATUS.md`](EXPERIMENT_STATUS.md).

## Conservative defaults

```yaml
requested_gpus: 1
runtime_budget_hours: 8.5
runtime_reserve_minutes: 10
pretrained: false
allow_external_pretrained: false
```

Additional defaults:

- no DDP / no `torchrun` in the production path;
- CPU multiprocessing only for DICOM/data work;
- Internet-independent final inference;
- output exactly `submission.csv`;
- competition-data checkpoint provenance checked;
- final inference MRI-only.

## External-pretraining policy

The conservative path uses only competition data:

- strong SSL: competition MRI only;
- B5 MRI/text alignment: competition MRI + competition reports only;
- B6: competition reports only;
- B7/B7.1/B8/B9: competition MRI + frozen B6 competition-report supervision;
- no ImageNet weights;
- no external clinical language model.

## Frozen report-supervision policy

B6 v1.2.1 is frozen after its gold audit.

```text
confidence threshold        0.75
positive target / weight    0.85 / 0.50
negative target / weight    0.05 / 1.00
uncertain/unmentioned       ignored
gold rows in weak export    0
```

B7/B7.1/B8/B9 do not use gold labels in gradients or early stopping.

Because the B6 gold audit informed the global weak-label policy, later 58-study scores are development/model-selection estimates.

## Gold-development policy

Do not:

- tune ensemble weights on the 58 gold labels;
- select target-specific post-hoc winners;
- reopen B4 selector searches;
- retune B6 parser rules/weak-label weights from later gold results;
- tune B8 spatial priors after its negative result;
- tune B9 target-specific routing or selectively restore substituted streams after seeing B9 gold outcomes;
- call a development result a leaderboard result.

## Closed branches

```text
B4 selector redesign
B5/B7.1 blend-weight search
raw-vs-rank blend search
target-specific model mixtures
post-audit B6 parser tuning from gold
B8 spatial-grid/anatomy-prior tuning from its gold result
```

## B9 execution policy

B9 is motivated by a label-free acquisition-metadata inconsistency.

Historical routing audit:

```text
selected training streams       21886
wrong semantic slots              552
wrong-slot fraction               2.52%
strict selected streams         21334
strict semantic mismatches          0
```

B9 exact routing:

```text
*_fluid       -> Fluid_Sensitive == True only
*_structural  -> Fluid_Sensitive == False only
missing class -> None / presence mask False
```

Everything else remains B7.1-equivalent:

```text
B5 encoder initialization
B6 v1.2.1 supervision
KneeMILNet global-token architecture
4 epochs
1560 batches/epoch
same optimizer/augmentation
TTA [-1,0,1]
5000 bootstrap replicates
```

Before gold evaluation, inspect:

```text
runs/b9_strict_routing/routing_audit.json
runs/b9_strict_routing/history.json
runs/b9_strict_routing/supervision_plan.json
```

The routing audit must certify `strict_semantic_mismatches = 0`.

## TTA policy

Neural validation/submission TTA contracts are stored explicitly. B7.1/B9 use the predeclared center offsets `[-1,0,1]` for development evaluation. Diagnostic center-only evaluation is not permission to retune TTA after reading gold labels.

## DICOM quality policy

Historical audit:

```text
21,886 / 21,886 historically selected series decoded
732,554 / 732,556 candidate files decoded
2 partial one-file failures
0 selected series lost
```

B9 changes routing selection, not DICOM decoding. Missing semantic streams are legitimate and must remain masked rather than fabricated.

## Submission schema

The final competition file must contain exactly:

```text
StudyInstanceUID + 12 target columns
```

with the expected study set/order and finite probabilities in `[0,1]`.

Default output:

```text
/kaggle/working/submission.csv
```

## Reporting vocabulary

Use these terms accurately:

- **preflight** — technical data-path/DICOM gate;
- **audit** — full data-quality/routing/supervision inventory;
- **training run** — optimization result;
- **gold development result** — 58-study development score;
- **model-selection CV** — repeated use of those 58 studies for method decisions;
- **leaderboard result** — actual competition submission score.
