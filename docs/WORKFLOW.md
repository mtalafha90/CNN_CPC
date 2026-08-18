# Training, validation and testing workflow

Three ordinary stages, one model. Run everything from the repository root —
the entry points locate `developments/src` themselves, so no `PYTHONPATH`
prefix is needed.

## 1. Environment

```bash
conda activate rsna-knee
pip install -e .
```

The dataset and run artefacts are not stored in Git. Set the paths explicitly.

```bash
cd /media/talafha/Disk_1/CNN_CPC

export DATA_ROOT="/media/talafha/Disk_1/CNN_CPC/rsna-knee-abnormality-detection"
export REPORT_LABELS="/media/talafha/Disk_1/CNN_CPC/runs/b6_report_labels_v121"
export TRANSLATED_LABELS="/media/talafha/Disk_1/CNN_CPC_current/runs/translation_rescue_supervision_v1"
export SERIES_POLICY="/media/talafha/Disk_1/CNN_CPC/runs/b12_variable_series/audit/series_policy.json"
export ENCODER="/media/talafha/Disk_1/CNN_CPC/runs/b16_full_report/report_ssl/b16_report_encoder.pt"
```

### Check `TRANSLATED_LABELS` before launching

This path is recorded under two different roots in the experiment archive —
`CNN_CPC/runs/...` in the earlier documents and `CNN_CPC_current/runs/...` in
the run that most recently completed. The merged-supervision export is
fingerprinted, so confirm which one is real rather than guessing:

```bash
ls -la "$TRANSLATED_LABELS"     # expect training_targets.csv, merge_audit.json, policy.json
sha256sum "$TRANSLATED_LABELS/training_targets.csv"
# c59d78c74743112f09946fd18b64d7726947e6f75b83aabd1f585389a89d045a
```

A mismatch aborts the run rather than training on the wrong labels.

### Pre-flight

```bash
for p in "$DATA_ROOT" "$REPORT_LABELS" "$TRANSLATED_LABELS" "$SERIES_POLICY" "$ENCODER"; do
  [ -e "$p" ] && echo "OK   $p" || echo "MISS $p"
done
python -m model.architecture
```

## 2. Training

`--supervision` chooses which report-label surface enters the gradient:
`original` is the frozen rule-parser labels, `merged` adds the cells recovered
by translating the non-Latin-script reports first.

```bash
python -m training.train \
  --supervision original \
  --data-root "$DATA_ROOT" \
  --report-labels "$REPORT_LABELS" \
  --translated-labels "$TRANSLATED_LABELS" \
  --series-policy "$SERIES_POLICY" \
  --encoder "$ENCODER" \
  --out-root runs/working_model
```

```bash
python -m training.train \
  --supervision merged \
  --data-root "$DATA_ROOT" \
  --report-labels "$REPORT_LABELS" \
  --translated-labels "$TRANSLATED_LABELS" \
  --series-policy "$SERIES_POLICY" \
  --encoder "$ENCODER" \
  --out-root runs/working_model
```

Both arms take all five paths even though each trains on one label surface
only: the trainer fingerprints both to prove the arms are matched.

Training runs to a fixed epoch. Nothing is selected by score, and neither arm
should be stopped early on the basis of the other arm's loss.

Expect roughly 90 minutes per arm on an RTX A4500 Laptop GPU, about 3 hours for
both.

### Outputs

The directory names remain `control` and `candidate` because the underlying
frozen trainer writes them:

```text
runs/working_model/control/model.pt              <- --supervision original
runs/working_model/control/training_audit.json
runs/working_model/control/history.json
runs/working_model/candidate/model.pt            <- --supervision merged
runs/working_model/candidate/training_audit.json
runs/working_model/candidate/history.json
```

## 3. Validation

```bash
python -m validation.validate --data-root "$DATA_ROOT" \
  --checkpoint runs/working_model/control/model.pt \
  --out runs/working_model/control/validation.json

python -m validation.validate --data-root "$DATA_ROOT" \
  --checkpoint runs/working_model/candidate/model.pt \
  --out runs/working_model/candidate/validation.json
```

**Read the result carefully.** These 58 expert-annotated studies were reused
throughout development, and a paired difference below roughly 0.03 macro AUC is
not resolvable at that sample size. Training on the full study population also
consumes the held-out weak-label studies, so these checkpoints have no powered
local validation surface left. Treat the number as a plausibility check — is
the model behaving sensibly — not as a comparison between the two arms.

## 4. Test-set prediction

```bash
python -m testing.test --data-root "$DATA_ROOT" \
  --checkpoint runs/working_model/control/model.pt \
  --out submissions/original_labels.csv

python -m testing.test --data-root "$DATA_ROOT" \
  --checkpoint runs/working_model/candidate/model.pt \
  --out submissions/merged_labels.csv
```

Every study is scored at slice offsets `[-1, 0, 1]` and the probabilities are
averaged. Each run also writes `<name>.csv.manifest.json` recording the
checkpoint hash, submission hash, encoder fingerprint and test-split series
coverage, so a submitted file can always be traced back to the run that
produced it. Keep the manifests — they are how the two submissions are told
apart afterwards.

## 5. Submission

Submit both files. The architecture, study population, series exposure,
optimiser and seeds are identical across the two arms, so the difference
between their scores isolates the effect of the translated supervision on the
only surface that is not reused development data.

## 6. Model information

```bash
python -m model.architecture
```

Prints the twelve findings and the architecture and training contract.

## Argument names changed in the reorganisation

Earlier notes and scripts use the experiment-era names:

| Previously | Now |
|---|---|
| `-m rsna_knee.phase9_matched_supervision_training` | `-m training.train` |
| `--arm control` / `--arm candidate` | `--supervision original` / `--supervision merged` |
| `--b6-root` | `--report-labels` |
| `--phase8-root` | `--translated-labels` |
| `--report-ssl-checkpoint` | `--encoder` |
| `PYTHONPATH=developments/src` | no longer required |

`--data-root`, `--series-policy`, `--out-root` and `--config` are unchanged.

## Historical work

Nothing from the research history was deleted. The complete experiment lineage
is preserved under `developments/`, and a safety branch exists at:

```text
archive/pre-clean-structure-2026-08-15
```
