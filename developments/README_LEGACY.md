# CNN_CPC — RSNA Knee Abnormality Detection

`CNN_CPC` is a PyTorch research pipeline for the **2026 RSNA Knee Abnormality Detection** challenge. The released training surface contains 4,407 studies: 58 fully expert-labelled studies and 4,349 report-only/non-gold studies, with multiple MRI series per knee and 12 study-level targets evaluated by macro ROC AUC.

## Current project state — 2026-08-15

> **B20 remains the active working model.** Its checkpoint and preprocessing are preserved unchanged.
>
> **B23-v1 has now been run and audited.** It improved state-only macro AUC and coverage relative to B6, but failed its predeclared formal gate because specificity (`0.5678`) was below B6 (`0.6061`). No canonical B23 holdout was therefore frozen and formal B24 remains blocked/not run.
>
> A separate **B24X exploratory matched pilot** was run with no gold evaluation and no promotion path. On the frozen B6 weak-v2 holdout, the B23/Qwen-supervised arm scored `0.7116126450` versus `0.6148488366` for the matched B6-supervised control, with paired delta `+0.0967638083` and 95% CI `[+0.0612014772,+0.1316174812]`. This is teacher-agreement evidence, not expert truth.
>
> **B24X-Density** has completed training: all 3,045 B6 cells were preserved and 2,844 B23-only cells were added, for 5,889 supervised cells with zero B6 overrides. Its frozen weak-v2 evaluation is pending.

| Model / experiment | Role | Canonical result | Status |
|---|---|---:|---|
| **B17** | fixed-epoch reference | E5 `0.6425890153` gold | frozen |
| **B18** | full-FOV comparator | replay E2 `0.6655517376` gold | frozen; nested audit complete |
| **B19** | spatial vignette ablation | E3 `0.6581308356` gold | rejected: artificial border shortcut |
| **B20** | historical 90% crop-only model | E2 `0.6671593555` gold | **ACTIVE WORKING MODEL** |
| **B21-v1** | pre-resize crop correction | weak-v2 `0.7410090411`; gold `0.6573196516` | weak-v2 passed; gold acceptance failed |
| **B22** | B21 duration audit | best E2 `0.6574269018` gold | closed; longer training did not rescue |
| **B23-v1** | local Qwen report labeller | state-only AUC `0.8125164416`; specificity `0.5678` | **formal gate FAILED** |
| **B24 formal** | matched B6-vs-B23 supervision | not run | **blocked by failed B23 gate** |
| **B24X** | exploratory matched supervision pilot | weak-v2 `0.6148488366 -> 0.7116126450` | completed; no gold/no promotion |
| **B24X-Density** | B6 + B23-only missing cells | trained on `5,889` cells | training complete; evaluation pending |

The 58 expert-labelled studies have been reused repeatedly and are therefore a **development/model-selection surface, not independent validation**. Hidden competition evaluation remains the independent predictive-performance signal.

For the full up-to-date record, start with [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md).

## Quick start

### 1. Install

```bash
git clone https://github.com/mtalafha90/CNN_CPC.git
cd CNN_CPC

conda create -n rsna-knee python=3.12 -y
conda activate rsna-knee

python -m pip install --upgrade pip
python -m pip install -e .
```

The package requires Python `>=3.10` and installs PyTorch, torchvision, NumPy, pandas, scikit-learn, pydicom, PyYAML, matplotlib and Pillow.

### 2. Point to the competition data

```bash
export DATA_ROOT="/path/to/rsna-knee-abnormality-detection"
```

Expected metadata include:

```text
train.csv
train_series.csv
test.csv
test_series.csv
```

MRI DICOMs may live under either `<split>_images/<StudyInstanceUID>/<SeriesInstanceUID>/` or `<split>_series/<StudyInstanceUID>/<SeriesInstanceUID>/`.

### 3. Verify the metadata

```bash
python - <<'PY'
import os
from pathlib import Path
from rsna_knee.data import load_train_csv, load_series_csv

root = Path(os.environ["DATA_ROOT"])
train = load_train_csv(root / "train.csv")
series = load_series_csv(root / "train_series.csv")

print("studies:", len(train))
print("series :", len(series))
print("data check passed")
PY
```

### 4. Run the active B20 model

If the canonical checkpoint is available:

```bash
rsna-knee-b20-submit \
  --config configs/b20_crop_focus.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b20_crop_focus/b20_model.pt \
  --out runs/b20_crop_focus/test_predictions.csv
```

### 5. Visualize a prediction

```bash
rsna-knee-b20-visualize \
  --config configs/b20_crop_focus.yaml \
  --data-root "$DATA_ROOT" \
  --checkpoint runs/b20_crop_focus/b20_model.pt \
  --target effusion \
  --cam-layer 28x28 \
  --cam-threshold 0.65
```

For a first-time user, start with **B20 inference and visualization**. B1-B19 explain how the project arrived at B20; B21+ are controlled research experiments and are not prerequisites for using the package.

## Active B20 recipe

Historical B20 executes:

```text
native MRI
  -> percentile normalization
  -> resize 224
  -> centered 90% crop
  -> resize 224
  -> frozen historical B16 encoder
  -> hierarchical series/pathology head
  -> 12 study-level probabilities
```

Canonical record:

```text
model                  B20_crop_only_joint_focus
checkpoint             runs/b20_crop_focus/b20_model.pt
canonical epoch        2
canonical gold score   0.667159355531343
```

B20 is retained because it is the clean historical knee-focused formulation without B19's synthetic vignette boundary. Its tiny difference from B18 on the reused 58-study surface is not evidence of predictive superiority.

## B21/B22 result: crop order and duration

B21 changed the spatial ordering to:

```text
B20 historical: native -> normalization -> resize224 -> crop90% -> resize224
B21-v1:         native -> crop90% -> normalization -> resize224
```

Weak-v2 favored B21:

```text
B20-v2 control macro AUC        0.7298727911
B21-v1 macro AUC                0.7410090411
paired delta                   +0.0111362500
paired 95% CI        [+0.0001624070,+0.0226346590]
```

But the one-look reused-gold acceptance did not:

```text
B20 replay macro AUC            0.6674066371
B21 macro AUC                   0.6573196516
B21 - B20 replay               -0.0100869854
P(B21 > B20)                    0.1812
```

B22 then showed that extending the same B21 formulation to E5 does not rescue it:

```text
Epoch   training loss   expert macro AUC
E1      0.7388751291    0.6135270850
E2      0.6381611442    0.6574269018  <- best
E3      0.6087977977    0.6387456622
E4      0.5890809184    0.6136783995
E5      0.5680555741    0.6282683534
```

This shifted the campaign focus from more optimization toward supervision and model-selection quality.

## B23-v1: labeller result

The B23 pilot used local `qwen3:14b` through Ollama (`Q4_K_M`). On the reused 58-study labeller audit:

```text
                         B6                 B23
state-only macro AUC     0.7024597743       0.8125164416
sensitivity              0.9748             0.9855
specificity              0.6061             0.5678
coverage                 0.3606             0.6365
```

Paired state-only AUC difference:

```text
raw B23 - B6             +0.1100566673
paired 95% CI            [+0.0680786389,+0.1531882641]
P(B23 > B6)              1.0000
```

Despite the AUC/coverage improvement, the formal gate **failed** because specificity did not exceed B6. B23-v1 is therefore not formally adopted and no canonical B23 holdout exists.

## B24X exploratory result

Because formal B24 could not run, B24X tested the supervision hypothesis separately while preserving the failed gate and prohibiting gold evaluation.

Matched training surface:

```text
shared studies                         692
B6 usable cells                       3045
B23 usable cells                      5697
B23-only added cells                  2844
B6 cells dropped by B23                192
overlap disagreements                   70 / 2853 = 2.5%
```

Frozen B6 weak-v2 evaluation, 623 studies, zero train/holdout overlap:

```text
B6 control       0.6148488366  [0.5856757959,0.6451316589]
B23/Qwen         0.7116126450  [0.6785972089,0.7435358854]
raw delta       +0.0967638083
paired 95% CI   [+0.0612014772,+0.1316174812]
P(B23 > B6)      1.0000
```

This is strong exploratory cross-teacher evidence, but **weak-v2 is a B6 teacher-agreement surface, not expert truth**. B20 therefore remains active.

See [`docs/B24X_EXPLORATORY_SUPERVISION.md`](docs/B24X_EXPLORATORY_SUPERVISION.md).

## B24X-Density — current next experiment

Density preserves all B6 committed labels and adds B23 only where B6 was silent:

```text
B6 cells preserved            3045
B23-only cells added           2844
final supervised cells         5889
B6 cells dropped                  0
B6 labels overridden              0
```

Training completed at fixed E2:

```text
E1 loss 0.7647414911
E2 loss 0.6197285242
checkpoint runs/b24x_density/density/b24x_density_model.pt
```

Its frozen weak-v2 evaluation is pending. The intended three-arm comparison is:

```text
B6       = 0.6148488366
Density  = pending
Full B23 = 0.7116126450
```

No gold evaluation or promotion is allowed for this exploratory arm.

## Canonical records

- [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md) — **current authoritative project snapshot**.
- [`docs/WORKING_MODEL.md`](docs/WORKING_MODEL.md) — active model and scientific position.
- [`docs/EXPERIMENT_STATUS.md`](docs/EXPERIMENT_STATUS.md) — experiment ledger.
- [`docs/B18_NESTED_EPOCH_AUDIT.md`](docs/B18_NESTED_EPOCH_AUDIT.md) — B18 checkpoint-selection audit.
- [`docs/B19_JOINT_FOCUS.md`](docs/B19_JOINT_FOCUS.md) — rejected B19 vignette formulation.
- [`docs/B20_CROP_ONLY_FOCUS.md`](docs/B20_CROP_ONLY_FOCUS.md) — canonical B20 record.
- [`docs/B20_NESTED_EPOCH_AUDIT.md`](docs/B20_NESTED_EPOCH_AUDIT.md) — B20 nested audit.
- [`docs/B21_PRERESIZE_CROP.md`](docs/B21_PRERESIZE_CROP.md) — B21 crop-order experiment.
- [`docs/B21_FULL_ACCEPTANCE.md`](docs/B21_FULL_ACCEPTANCE.md) — B21 one-look acceptance.
- [`docs/B22_DURATION_AUDIT.md`](docs/B22_DURATION_AUDIT.md) — B22 duration audit.
- [`docs/B23_LLM_REPORT_LABELS.md`](docs/B23_LLM_REPORT_LABELS.md) — formal B23 protocol/background.
- [`docs/B24_SUPERVISION_SOURCE.md`](docs/B24_SUPERVISION_SOURCE.md) — formal B24 protocol.
- [`docs/B24X_EXPLORATORY_SUPERVISION.md`](docs/B24X_EXPLORATORY_SUPERVISION.md) — exploratory B24X and Density record.
- [`docs/VALIDATION.md`](docs/VALIDATION.md) — validation governance.
- [`docs/VISUALIZATION_GUIDE.md`](docs/VISUALIZATION_GUIDE.md) — visualization guide.

## Governance

```text
B17: frozen fixed-epoch reference
B18: frozen full-FOV comparator; nested audit complete
B19: rejected spatial formulation
B20: ACTIVE WORKING MODEL; preserve checkpoint/preprocessing exactly
B21: closed; weak-v2 passed but gold acceptance failed
B22: closed; E2 best, no duration rescue
B23-v1: formal labeller gate FAILED; not adopted
B24 formal: BLOCKED / NOT RUN
B24X: exploratory matched pilot only; NO GOLD / NO PROMOTION
B24X-Density: training complete; weak-v2 evaluation pending; NO GOLD / NO PROMOTION
58-study expert surface: reused development/post-hoc surface, not independent validation
weak-v2: B6 teacher-agreement diagnostic, not validated expert truth
hidden competition evaluation: independent predictive-performance signal
no target-specific epoch/model mixing from reused development results
FINAL all-data expert-label fit: deferred
```
