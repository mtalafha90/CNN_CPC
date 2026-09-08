# What TTA is worth, and whether an ensemble helps

**Status: implemented, unrun.** Both need only a GPU and existing checkpoints —
no training. Run them whenever the GPU is free.

Two questions, one tool, and the first gates the second.

## 1. Three TTA views cost three quarters of the inference budget

Every submission this project has made scores each study **three times**, at
centre offsets `(-1, 0, 1)`, and averages the probabilities.

```text
Kaggle ceiling                                 9.00 h
guard (B37_SUBMISSION_MAX_HOURS)               8.25 h
one model, 3 offsets, ~1,300 studies, 2x T4    ~4.5 h
one model, 1 offset                            ~1.5 h
```

**Nobody has measured what the two extra views buy.** B39 tried to widen TTA to
five offsets and failed operationally before producing a score, so the marginal
worth of offset averaging is unknown here.

That measurement decides whether an ensemble is affordable at all:

```text
1 model  x 3 offsets   ~4.5 h    what runs today
3 models x 1 offset    ~4.5 h    the same cost, three independent models
```

### Running it

```bash
cd /media/talafha/Disk_1/CNN_CPC
PYTHONPATH=developments/src python -m rsna_knee.ensemble_eval \
  --mode tta \
  --data-root /media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection \
  --labels-root runs/067_Experiment_LLM_FILL_ALL_b6_preserved_llm_fill_all_targets/b6_plus_llm_fill_all \
  --series-policy runs/020_Experiment_B12_variable_series/b12_variable_series/audit/series_policy.json \
  --base-checkpoint runs/067_Experiment_LLM_FILL_ALL_b6_preserved_llm_fill_all_targets/b6_plus_llm_fill_all_ft1/train/llm-filled/model.pt \
  --domain-split runs/083_Experiment_B50_selection_gate/b50_ordered_slice_selection_split \
  --checkpoint runs/087_Experiment_B52_full_data/b52_best_model.pt \
  --num-workers 6 \
  --out-json runs/tta_measurement.json
```

Three passes over 548 studies. Each offset is a separate single-view pass whose
probabilities are then averaged, which is exactly what the submission does —
so this reproduces it without touching the model or the dataset.

### The threshold, declared before the number

```text
delta >= 0.005    three views are earning their three-fold cost. Keep them.
delta <  0.005    they are not. Dropping to one offset frees two thirds of the
                  inference budget, which is what makes a three-model ensemble
                  affordable.
```

`TTA_WORTH_KEEPING = 0.005` is a module constant, written down before the
measurement exists.

## 2. The ensemble, and why it was blocked until now

There is no ensembling code in this archive, and until now there was nothing
worth ensembling. 086, 087 and B54 share a base checkpoint, an architecture, a
seed and a split, differing only in training population and two mechanisms
measured at zero. **Rank-averaging near-identical models lands between them,
not above.** An earlier enthusiasm for ensembling in
`WHERE_THE_SCORE_IS_ACTUALLY_LOST` did not account for that, and is corrected
here.

B53 (augmentation) and B55 (physical crop, canonical laterality, 336) are the
first genuinely different checkpoints this project has. Whether they are
different *enough* is measurable rather than arguable.

### Running it

```bash
PYTHONPATH=developments/src python -m rsna_knee.ensemble_eval \
  --mode ensemble \
  ... the same six paths as above ... \
  --checkpoint runs/087_Experiment_B52_full_data/b52_best_model.pt \
  --checkpoint runs/088_Experiment_B53_augmentation_applied/b53_best_model.pt \
  --checkpoint runs/089_Experiment_B55_physical_geometry/b52_best_model.pt \
  --num-workers 6 \
  --out-json runs/ensemble_measurement.json
```

### Reading it

The report prints the gain **and the agreement**, together, because one
explains the other:

```text
    best single member    0.8471
    the ensemble          0.8520
    gain                  +0.0049

    mean rank correlation 0.8300
    highest pair          0.8300
```

An ensemble pays for disagreement. A mean rank correlation near `1.0` means the
members already say the same thing, and a small gain there is the expected
result rather than a disappointing one. A high `highest pair` with a lower mean
means two of your members are near-twins and the third is doing the work.

### Why ranks rather than probabilities

Macro ROC AUC depends only on the ordering of studies within a target.
Averaging raw probabilities lets a member with a different calibration dominate
one it merely disagrees with in scale; averaging ranks removes calibration from
the question and leaves only the ordering the metric reads. It also fits no
weights, so there is nothing to overfit.

Ties take their average rank, so two studies a member cannot separate stay
unseparated rather than being ordered by their position in the array.

## What neither of these is

**Not a selection.** Both score checkpoints that were already chosen by their
own runs on their own surfaces. Picking the best row from either table and
calling it promoted would be the post-hoc selection this archive forbids.

**Not a substitute for the leaderboard.** The measured exchange rate is that
local movement reaches Kaggle at about a quarter, and these are local
measurements. What they can do is stop you spending a submission slot to learn
something a few GPU-hours would have told you.

## The order to run them in

```text
1. TTA on one existing checkpoint          three passes, hours
2. read the delta against 0.005
3. if TTA is cheap to drop, the ensemble becomes affordable at 3 members
4. ensemble B53 + B55 + 087 once both have finished
5. only then decide what to submit
```

Step 1 can run today on `087`. Steps 3 and 4 need B53 and B55 to exist.
