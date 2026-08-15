# Training, validation and testing workflow

The root of the repository now exposes one model and three ordinary machine-learning stages.

## 1. Environment

```bash
conda activate rsna-knee
pip install -e .
```

The dataset is not stored in Git. Set the path explicitly when running commands.

## 2. Training

```bash
python -m training.train \
  --data-root /path/to/rsna-knee-abnormality-detection \
  --b6-root runs/b6_report_labels_v121 \
  --series-policy runs/b12_variable_series/audit/series_policy.json \
  --report-ssl-checkpoint runs/b16_v2_safe_report/report_ssl/b16_v2_report_encoder.pt \
  --out-root runs/current_model
```

The clean command delegates to the preserved, verified B20 implementation in `developments/src/rsna_knee/`.

## 3. Validation

```bash
python -m validation.validate \
  --data-root /path/to/rsna-knee-abnormality-detection \
  --checkpoint runs/b20_crop_focus/b20_model.pt
```

The output is written to `runs/current_model/validation.json` by default.

**Interpretation:** the 58 expert studies are a reused development surface, not independent validation/test data.

## 4. Testing / submission inference

```bash
python -m testing.test \
  --data-root /path/to/rsna-knee-abnormality-detection \
  --checkpoint runs/b20_crop_focus/b20_model.pt \
  --out submission.csv
```

This produces `submission.csv` plus the existing B20 inference manifest.

## 5. Model information

```bash
python -m model.architecture
```

## Historical work

Nothing from the research history was deleted. The complete prior repository structure is preserved under `developments/` and a safety branch exists at:

```text
archive/pre-clean-structure-2026-08-15
```
