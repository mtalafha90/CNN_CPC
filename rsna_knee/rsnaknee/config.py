"""Experiment configuration.

Everything tunable lives in one YAML file so a run is fully described by its
config plus a git commit. Command-line overrides use dotted keys, for example
``--set model.backbone=convnext_small train.epochs=12``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PathsConfig:
    data_dir: str = "data"
    cache_dir: str = "cache"
    output_dir: str = "runs/baseline"
    train_csv: str | None = None
    sample_submission_csv: str | None = None
    reports_csv: str | None = None
    test_dicom_dir: str | None = None


@dataclass
class DataConfig:
    image_size: int = 224
    depth: int = 24
    max_series: int = 5
    n_folds: int = 5
    seed: int = 42
    augment: bool = True
    horizontal_flip: bool = False
    rotate_degrees: float = 10.0
    scale_jitter: float = 0.1
    intensity_jitter: float = 0.2
    noise_std: float = 0.01
    series_dropout: float = 0.1
    random_erase: float = 0.15
    num_workers: int = 8


@dataclass
class ModelSettings:
    backbone: str = "convnext_tiny"
    pretrained: bool = True
    embed_dim: int = 512
    slice_layers: int = 2
    slice_heads: int = 8
    dropout: float = 0.1
    drop_path: float = 0.1
    grad_checkpoint: bool = False


@dataclass
class TrainConfig:
    epochs: int = 12
    batch_size: int = 4
    accumulate: int = 2
    learning_rate: float = 3e-4
    backbone_lr_scale: float = 0.1
    weight_decay: float = 0.01
    warmup_epochs: float = 1.0
    amp_dtype: str = "bf16"  # "bf16", "fp16" or "fp32"
    ema_decay: float = 0.999
    clip_grad: float = 5.0
    focal_gamma: float = 0.0
    label_smoothing: float = 0.01
    pos_weight_max: float = 20.0
    aux_weight: float = 0.3
    distil_weight: float = 0.0
    distil_temperature: float = 2.0
    channels_last: bool = True
    compile: bool = False
    early_stop_patience: int = 4
    metric_name: str = "macro_auc"


@dataclass
class TextConfig:
    model_name: str = "xlm-roberta-base"
    max_length: int = 320
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    text_column: str | None = None


@dataclass
class InferenceConfig:
    batch_size: int = 4
    tta_hflip: bool = False
    tta_slice_shift: bool = True
    use_ema: bool = True
    half: bool = True


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelSettings = field(default_factory=ModelSettings)
    train: TrainConfig = field(default_factory=TrainConfig)
    text: TextConfig = field(default_factory=TextConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")


def _merge(target: Any, payload: dict) -> Any:
    """Recursively apply a dictionary onto a dataclass instance."""
    if not is_dataclass(target):
        return payload
    known = {f.name: f for f in fields(target)}
    for key, value in payload.items():
        if key not in known:
            raise KeyError(f"Unknown configuration key: {key}")
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(target, key, value)
    return target


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load a config from YAML and apply ``key.subkey=value`` overrides."""
    config = Config()
    if path is not None:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        _merge(config, payload)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override '{override}' is not of the form key.subkey=value")
        key, raw = override.split("=", 1)
        _apply_dotted(config, key.strip(), _parse_scalar(raw.strip()))
    return config


def _apply_dotted(config: Config, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    target: Any = config
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise KeyError(f"Unknown configuration section: {part}")
        target = getattr(target, part)
    if not hasattr(target, parts[-1]):
        raise KeyError(f"Unknown configuration key: {dotted}")
    setattr(target, parts[-1], value)


def _parse_scalar(raw: str) -> Any:
    """Turn a command-line string into a bool, number, None or string."""
    lowered = raw.lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw
