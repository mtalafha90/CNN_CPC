# Current-model configuration

`current_model.yaml` is the frozen configuration of the active B20 working model.

The `b*` key names are retained intentionally because they are part of the model's recorded provenance. They should not be interpreted as separate models in the clean interface.

Use this file through the top-level commands:

```bash
python -m training.train --config config/current_model.yaml ...
python -m validation.validate --config config/current_model.yaml ...
python -m testing.test --config config/current_model.yaml ...
```

Historical experiment configurations are preserved under `developments/configs/`.
