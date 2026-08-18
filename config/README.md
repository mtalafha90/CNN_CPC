# Model configuration

`current_model.yaml` is the frozen configuration of the working model.

The `b*` key names are retained deliberately: they are part of the recorded
provenance of each setting, and renaming them would break the frozen
implementation that reads them. They do not name separate models. Code above
the implementation bridge reaches them through `model/_implementation.py`
rather than reading the file directly.

Use the configuration through the top-level commands:

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
