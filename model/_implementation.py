"""Bridge from the clean interface to the preserved research implementation.

This is deliberately the **only** module in the working-model interface that
refers to the historical experiment names.  Everything above it -- `model`,
`data`, `training`, `validation`, `testing` -- is written in plain language.

The research lineage under `developments/` is treated as frozen: nothing here
reimplements a trained component.  Each name below is bound once, so an
experiment rename is a single-line change rather than a repository-wide edit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import torch

from .bootstrap import ensure_developments_source

# The architectures the working-model interface is able to reconstruct.  A
# checkpoint records its own architecture string, so loading dispatches on the
# payload rather than on a caller-supplied guess.
_ARCHITECTURE_BUILDERS: dict[str, str] = {
    "hierarchical_series_token_plus_zero_gated_complementary_softmax_pool_v1": (
        "b29_complementary_series_pool:build_b29_model"
    ),
    "b29_plus_zero_init_depthwise_local_slice_context_v1": (
        "b31_local_context_complementary_pool:build_b31_model"
    ),
    "b31_training_only_local_context_scaffold_eval_bypass_v1": (
        "b34_training_only_context_scaffold:build_b34_model"
    ),
}

WORKING_ARCHITECTURE = "b31_training_only_local_context_scaffold_eval_bypass_v1"

SUPERVISION_ARMS = {
    "original": "control",
    "merged": "candidate",
}


def _resolve(path: str) -> Callable[..., Any]:
    """Import `module:attribute` from the preserved implementation."""
    ensure_developments_source()
    module_name, _, attribute = path.partition(":")
    module = __import__(f"rsna_knee.{module_name}", fromlist=[attribute])
    return getattr(module, attribute)


def read_config(path: str | Path) -> dict:
    return dict(_resolve("b7_weak_supervision:_read_config")(path))


def target_names() -> tuple[str, ...]:
    ensure_developments_source()
    from rsna_knee.constants import TARGETS

    return tuple(TARGETS)


def resolve_runtime(config: dict):
    return _resolve("runtime:resolve_runtime")(config)


def autocast(runtime):
    return _resolve("runtime:autocast")(runtime)


def runtime_budget(*, max_hours: float, reserve_minutes: float):
    return _resolve("budget:RuntimeBudget")(
        max_hours=max_hours, reserve_minutes=reserve_minutes
    )


def crop_policy(config: dict) -> dict:
    """The deterministic centred-crop contract the working model requires."""
    return _resolve("b20_crop_focus:require_b20_contract")(config)


def apply_crop(volumes, policy: dict):
    return _resolve("crop_focus:apply_crop_focus")(volumes, policy)


def network_spec(config: dict, *, normalize_input: bool = True) -> dict:
    return _resolve("b34_training_only_context_scaffold:b34_model_spec")(
        config, normalize_input=normalize_input
    )


def build_network(spec: dict, **kwargs):
    architecture = str(spec.get("architecture", ""))
    builder = _ARCHITECTURE_BUILDERS.get(architecture)
    if builder is None:
        known = ", ".join(sorted(_ARCHITECTURE_BUILDERS)) or "<none>"
        raise ValueError(
            f"unsupported architecture {architecture!r}; the working-model "
            f"interface can rebuild: {known}"
        )
    return _resolve(builder)(spec, **kwargs)


def freeze_encoder(model) -> None:
    _resolve("b17_training:freeze_encoder")(model)


def encoder_fingerprint(model) -> str:
    return _resolve("b17_training:encoder_state_sha256")(model.encoder)


def load_checkpoint(path: str | Path, *, device: str = "cpu"):
    """Rebuild a trained network from any checkpoint this interface supports.

    The architecture is taken from the checkpoint itself, and the frozen
    encoder fingerprint recorded at training time is re-verified after the
    weights are loaded.  A checkpoint whose encoder drifted is rejected rather
    than silently scored.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    for key in ("model_spec", "model_state"):
        if key not in payload:
            raise ValueError(f"checkpoint is missing {key!r}")

    model = build_network(payload["model_spec"], pretrained_weights=False)
    model.load_state_dict(payload["model_state"], strict=True)
    freeze_encoder(model)

    recorded = str(payload.get("encoder_sha256_final", ""))
    if recorded and encoder_fingerprint(model) != recorded:
        raise ValueError("reconstructed encoder fingerprint does not match the checkpoint")

    return model.to(device), payload


# --- data ------------------------------------------------------------------

def series_index(series, uids):
    return _resolve("b12_variable_series:build_variable_series_index")(series, uids)


def collate(batch):
    return _resolve("b12_variable_series:collate_variable_series")(batch)


def study_dataset(uids, index, dataset_config, *, train: bool, policy: dict):
    return _resolve("b20_crop_focus:CropFocusedVariableSeriesKneeDataset")(
        uids, index, dataset_config, train=train, crop_focus_policy=policy
    )


def inference_dataset_config(config: dict, root: Path, offsets: tuple[int, ...]):
    return _resolve("b17_submission:_test_dataset_config")(config, root, offsets)


def inference_slice_offsets(config: dict, default: tuple[int, ...]) -> tuple[int, ...]:
    """Read the configured test-time slice offsets."""
    return tuple(int(x) for x in config.get("b7_eval_tta_offsets", default))


def inference_batch_size(config: dict) -> int:
    """Read the configured number of studies scored per inference batch."""
    return max(1, int(config.get("b7_eval_batch_size", 2)))


def load_study_table(path):
    return _resolve("data:load_train_csv")(path)


def load_test_table(path):
    return _resolve("data:load_test_csv")(path)


def load_series_table(path):
    return _resolve("data:load_series_csv")(path)


def repair_series_metadata(series, root, *, split: str):
    return _resolve("data:backfill_series_metadata")(series, root, split=split)


# --- evaluation ------------------------------------------------------------

def expert_loader(config, root, studies, series, runtime, policy):
    """Loader over the 58 expert-labelled studies used for development checks."""
    return _resolve("b20_crop_focus:_expert_loader")(
        config, root, studies, series, runtime, policy
    )


def predict(model, loader, runtime):
    return _resolve("b12_1_gold_eval:predict_b12_1")(model, loader, runtime)


def macro_auc(truth, prediction):
    return _resolve("evaluation:macro_auc_from_arrays")(truth, prediction)


def validate_submission(frame, uids) -> None:
    _resolve("b17_submission:_validate_submission")(frame, uids)


def validate_against_sample(root, frame) -> dict:
    return _resolve("b17_submission:_validate_sample_submission")(root, frame)


# --- training --------------------------------------------------------------

def train_working_model(
    config: dict,
    *,
    supervision: str,
    report_labels_root: str | Path,
    translated_labels_root: str | Path,
    series_policy_path: str | Path,
    encoder_checkpoint: str | Path,
    out_root: str | Path,
):
    """Train the working model on the full report-only study population.

    `supervision` selects which label surface enters the gradient:
    `"original"` uses the frozen rule-parser labels only, `"merged"` adds the
    translated cells recovered from the non-Latin-script reports.
    """
    arm = SUPERVISION_ARMS.get(supervision)
    if arm is None:
        choices = ", ".join(sorted(SUPERVISION_ARMS))
        raise ValueError(f"supervision must be one of: {choices}")

    return _resolve("phase9_matched_supervision_training:train_phase9_arm")(
        config,
        arm=arm,
        b6_root=str(report_labels_root),
        phase8_root=str(translated_labels_root),
        series_policy_path=str(series_policy_path),
        report_ssl_checkpoint=str(encoder_checkpoint),
        out_root=str(out_root),
    )
