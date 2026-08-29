# Endpoint configuration

## Maintained B42 endpoint

[`b42_constant_area_aspect_sparse.yaml`](b42_constant_area_aspect_sparse.yaml)
is the immutable B42 model configuration. It is the configuration associated
with the maintained `0.714` Kaggle reference endpoint and must be paired with
the exact frozen B42 and B34 checkpoint artefacts described in
[`../docs/ACTIVE_ENDPOINTS.md`](../docs/ACTIVE_ENDPOINTS.md).

## Legacy compatibility configuration

`current_model.yaml` is the frozen configuration for the older top-level B34
compatibility interface. It is retained for historical reproduction; it is not
the configuration for B42 training or Kaggle submission.

The `b*` key names are retained deliberately: they are part of the recorded
provenance of each setting, and renaming them would break the frozen
implementation that reads them. They do not name separate models. Code above
the implementation bridge reaches them through `model/_implementation.py`
rather than reading the file directly.

Use the legacy configuration only through the top-level commands:

```bash
python -m training.train --config config/current_model.yaml ...
python -m validation.validate --config config/current_model.yaml ...
python -m testing.test --config config/current_model.yaml ...
```

`--config` defaults to this file, so it can usually be omitted.

Two settings are load-bearing and worth knowing about:

- `runtime_budget_hours: 8.5` — training and inference refuse to begin work
  they cannot finish inside this budget. The guard rejects any value of 9 or
  more, so raising it does not extend a session; it stops the run.
- `b7_eval_tta_offsets: [-1, 0, 1]` — the slice offsets averaged at inference.
  Test-set prediction refuses to run with any other value, so a submission
  cannot silently be produced under different inference geometry.

Historical experiment configurations are preserved under `developments/configs/`.
