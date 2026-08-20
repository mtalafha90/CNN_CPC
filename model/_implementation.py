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

# The two label surfaces, named for the reports each one can actually read.
# The frozen rule parser matches Latin-script vocabulary across several
# languages -- it is not English-only -- and translation extends that to the
# Greek- and Cyrillic-script reports.  The values are the arm names the
# preserved trainer uses internally.
SUPERVISION_SURFACES = {
    "latin-script": "control",
    "all-script": "candidate",
    # The rule parser answers about a quarter of the label cells and declines
    # the rest. This keeps every answer it gives and asks a local LLM only about
    # the cells it left blank, which is the formulation measured to carry the
    # whole benefit of replacing the parser without overriding any of its calls.
    "llm-filled": "llm_fill",
}

# Which pretrained weights the frozen encoder starts from.  "report-aligned" is
# the frozen working model's encoder; "dinov3" swaps in DINOv3 self-supervised
# ConvNeXt weights of the same width, leaving everything above it untouched.
ENCODERS = ("report-aligned", "dinov3")


RUNS_ROOT = "runs"
STAGES = ("train", "validate", "test")


def run_directory(experiment: str, stage: str, *, runs_root: str | Path = RUNS_ROOT) -> Path:
    """Return `runs/<experiment>/<stage>`, creating it if needed.

    Everything one experiment produces lives under a single directory, split by
    the stage that produced it, so a run can be inspected, archived or deleted
    as one thing.
    """
    if stage not in STAGES:
        raise ValueError(f"stage must be one of: {', '.join(STAGES)}")
    name = str(experiment).strip()
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        raise ValueError("experiment must be a single directory name")
    path = Path(runs_root) / name / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def attach_dinov3(model, *, variant: str = "tiny", pretrained_weights: bool = True):
    """Replace a built model's encoder with the DINOv3 one of the same width."""
    return _resolve("dinov3_encoder:attach_dinov3_encoder")(
        model, variant=variant, pretrained_weights=pretrained_weights
    )


def freeze_encoder(model) -> None:
    _resolve("b17_training:freeze_encoder")(model)


def encoder_fingerprint(model) -> str:
    return _resolve("b17_training:encoder_state_sha256")(model.encoder)


def resolve_checkpoints(paths) -> list[Path]:
    """Check every checkpoint exists before any of them is loaded.

    Scoring several models means several paths, and a typo in the last one
    should not be discovered after the first has been built -- or, worse, after
    the scans have started being read. A missing path also names its parent's
    contents, because the usual cause is a run directory called something
    slightly different from what was typed.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    resolved = [Path(p) for p in paths]
    if not resolved:
        raise ValueError("no checkpoint given")

    for path in resolved:
        if path.is_file():
            continue
        message = f"no checkpoint at {path}"
        if path.parent.is_dir():
            siblings = sorted(p.name for p in path.parent.glob("*.pt"))
            message += (
                f"; {path.parent} holds: {', '.join(siblings)}"
                if siblings
                else f"; {path.parent} holds no .pt files"
            )
        else:
            # A wrong run name leaves several levels missing at once, so walk up
            # to the last directory that does exist and show what is in it.
            existing = next(
                (parent for parent in path.parents if parent.is_dir()), None
            )
            if existing is not None:
                nearby = sorted(p.name for p in existing.iterdir() if p.is_dir())
                message += (
                    f"; {existing} holds: {', '.join(nearby)}"
                    if nearby
                    else f"; {existing} holds no directories"
                )
        raise FileNotFoundError(message)
    return [p.resolve() for p in resolved]


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

    spec = payload["model_spec"]
    model = build_network(spec, pretrained_weights=False)

    # A checkpoint trained with a swapped encoder carries a different encoder
    # module, so it has to be reattached before the weights are loaded.  The
    # weights come from the checkpoint itself, so nothing is downloaded here.
    if str(spec.get("encoder_source", "report-aligned")) == "dinov3":
        variant = (
            payload.get("encoder", {}).get("variant")
            or spec.get("dinov3_variant")
            or "tiny"
        )
        attach_dinov3(model, variant=str(variant), pretrained_weights=False)

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


def training_dataset_config(config: dict, root: Path, *, train: bool = False):
    """Dataset settings for the training population, with augmentation optional."""
    return _resolve("b7_weak_supervision:make_b7_dataset_config")(
        config, root, train=train
    )


def report_label_supervision(studies, *, surface: str, latin_root, all_root):
    """Load the report-derived labels for one supervision surface."""
    arm = SUPERVISION_SURFACES.get(surface)
    if arm is None:
        choices = ", ".join(sorted(SUPERVISION_SURFACES))
        raise ValueError(f"surface must be one of: {choices}")
    return _resolve("phase9_supervision:load_phase9_arm_supervision")(
        studies, arm=arm, b6_root=str(latin_root), phase8_root=str(all_root)
    )


def collate_studies_fn():
    """The collate function, for callers building their own DataLoader."""
    return _resolve("b12_variable_series:collate_variable_series")


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

# How confident the labels are allowed to make the model. A report saying a
# finding is present is not proof it is present, so the training target sits
# below 1.0 and the loss stops pushing once it is reached.
#
# These are deliberately not the keys naming the frozen label export. Those
# describe what the export contains and are contract-checked; these describe
# what training aims at, which is a separate choice applied after loading.
LABEL_CONFIDENCE_KEYS = {
    "positive_target": "label_confidence_positive_target",
    "negative_target": "label_confidence_negative_target",
}


def set_label_confidence(config: dict, **targets: float | None) -> dict:
    """Return `config` with the label targets that were actually given.

    These are a claim about how often the report parser is right, so they
    should be set from a measurement rather than left at a value nobody has
    checked. `tools.label_audit` provides the measurement.
    """
    updated = dict(config)
    for name, value in targets.items():
        if value is None:
            continue
        key = LABEL_CONFIDENCE_KEYS.get(name)
        if key is None:
            choices = ", ".join(sorted(LABEL_CONFIDENCE_KEYS))
            raise ValueError(f"unknown label target {name!r}; expected one of: {choices}")
        if not 0.0 < float(value) < 1.0:
            raise ValueError(f"{name} must lie between 0 and 1, not {value}")
        # A cell is positive or negative by which side of 0.5 it falls, in the
        # loss and in the audit counts alike. A target that crosses over would
        # not just be wrong, it would recount the labels.
        if name == "positive_target" and float(value) <= 0.5:
            raise ValueError(f"positive_target must stay above 0.5, not {value}")
        if name == "negative_target" and float(value) >= 0.5:
            raise ValueError(f"negative_target must stay below 0.5, not {value}")
        updated[key] = float(value)
    return updated


# The seed every protocol-matched experiment has used. A run that varies it is
# not protocol-matched, and has to say so rather than quietly differ.
FROZEN_PROTOCOL_SEED = 2026


def set_seed(config: dict, seed: int | None) -> dict:
    """Return `config` with the seed set, declared as an ensemble member.

    Changing the seed is how a genuinely different model gets trained, which is
    the only thing that makes an ensemble worth averaging. It is also a
    departure from the matched protocol every comparison so far has relied on,
    so the run records itself as an ensemble member: the frozen contract then
    permits the change while still refusing an undeclared one.
    """
    if seed is None:
        return config
    updated = dict(config)
    updated["seed"] = int(seed)
    if int(seed) != FROZEN_PROTOCOL_SEED:
        updated["ensemble_member"] = int(seed)
    return updated


def train_working_model(
    config: dict,
    *,
    supervision: str,
    latin_script_labels_root: str | Path,
    all_script_labels_root: str | Path,
    series_policy_path: str | Path,
    encoder_checkpoint: str | Path | None,
    out_root: str | Path,
    encoder: str = "report-aligned",
    dinov3_variant: str = "tiny",
    encoder_trainable_stages: int = 0,
    encoder_lr_scale: float = 0.05,
    llm_filled_labels_root: str | Path | None = None,
):
    """Train the working model on the full report-only study population.

    `supervision` selects which label surface enters the gradient:
    `"latin-script"` uses the frozen rule-parser labels only, `"all-script"`
    adds the cells recovered by translating the Greek- and Cyrillic-script
    reports before parsing, and `"llm-filled"` keeps every parser answer and
    fills only the cells it left blank, which needs `llm_filled_labels_root`.
    """
    arm = SUPERVISION_SURFACES.get(supervision)
    if arm is None:
        choices = ", ".join(sorted(SUPERVISION_SURFACES))
        raise ValueError(f"supervision must be one of: {choices}")
    if arm == "llm_fill" and not llm_filled_labels_root:
        raise ValueError("supervision 'llm-filled' requires llm_filled_labels_root")

    return _resolve("phase9_matched_supervision_training:train_phase9_arm")(
        config,
        arm=arm,
        b6_root=str(latin_script_labels_root),
        phase8_root=str(all_script_labels_root),
        llm_fill_root=str(llm_filled_labels_root) if llm_filled_labels_root else None,
        series_policy_path=str(series_policy_path),
        report_ssl_checkpoint=str(encoder_checkpoint or ""),
        out_root=str(out_root),
        out_dirname=supervision,
        encoder_source=encoder,
        dinov3_variant=dinov3_variant,
        encoder_trainable_stages=encoder_trainable_stages,
        encoder_lr_scale=encoder_lr_scale,
    )
