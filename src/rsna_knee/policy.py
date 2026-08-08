from __future__ import annotations

from pathlib import Path

SSL_COMPETITION_SOURCE = "competition_training_data"


def _offsets(config: dict, key: str, fallback) -> tuple[int, ...]:
    values = config.get(key, fallback)
    values = [0] if values is None or len(values) == 0 else values
    return tuple(int(value) for value in values)


def validate_competition_config(config: dict, *, purpose: str) -> None:
    """Validate the conservative production contract used for competition runs."""
    if not bool(config.get("competition_mode", True)):
        return

    budget = float(config.get("runtime_budget_hours", 8.5))
    if not 0 < budget < 9.0:
        raise ValueError("competition_mode requires runtime_budget_hours strictly below 9")

    reserve = float(config.get("runtime_reserve_minutes", 10.0))
    if reserve < 0 or reserve >= budget * 60.0:
        raise ValueError("runtime_reserve_minutes must be non-negative and shorter than the run budget")

    if bool(config.get("pretrained", False)) and not bool(config.get("allow_external_pretrained", False)):
        raise ValueError(
            "competition_mode forbids external pretrained weights by default; "
            "set pretrained:false unless the competition-specific rules have been explicitly verified"
        )

    if int(config.get("requested_gpus", 1)) != 1:
        raise ValueError("production competition mode is single-GPU only")

    submission_offsets = _offsets(config, "tta_center_offsets", [-1, 0, 1])
    validation_offsets = _offsets(config, "validation_tta_offsets", submission_offsets)
    if validation_offsets != submission_offsets and not bool(
        config.get("allow_validation_submission_tta_mismatch", False)
    ):
        raise ValueError(
            "competition_mode requires validation_tta_offsets to match tta_center_offsets; "
            "validation must measure the planned submission inference policy"
        )

    stage1_root = config.get("cotrain_stage1_root")
    stage1_candidates = config.get("cotrain_stage1_candidates")
    if stage1_root and stage1_candidates:
        raise ValueError("set either cotrain_stage1_root or cotrain_stage1_candidates, not both")
    if stage1_candidates is not None:
        if not isinstance(stage1_candidates, (list, tuple)) or len(stage1_candidates) < 1:
            raise ValueError("cotrain_stage1_candidates must be a non-empty list of Stage-1 roots")

    ssl_path = config.get("ssl_encoder_checkpoint")
    if ssl_path and not bool(config.get("allow_external_pretrained", False)):
        if str(config.get("ssl_checkpoint_source", "")) != SSL_COMPETITION_SOURCE:
            raise ValueError(
                "competition_mode requires ssl_checkpoint_source=competition_training_data "
                "for attached SSL weights when external pretrained artifacts are disabled"
            )
        path = Path(str(ssl_path))
        if purpose == "train" and path.is_file():
            import torch

            payload = torch.load(path, map_location="cpu", weights_only=False)
            validate_ssl_payload(payload, config)

    if purpose == "infer":
        expected = str(config.get("submission_filename", "submission.csv"))
        if Path(expected).name != "submission.csv":
            raise ValueError("competition submission filename must be submission.csv")


def validate_ssl_payload(payload: dict, config: dict) -> None:
    """Verify that an attached SSL checkpoint matches the declared safe source."""
    if not bool(config.get("competition_mode", True)):
        return
    if bool(config.get("allow_external_pretrained", False)):
        return
    if payload.get("source") != SSL_COMPETITION_SOURCE:
        raise ValueError("attached SSL checkpoint is missing competition-training-data provenance")
    ssl_config = payload.get("config")
    if not isinstance(ssl_config, dict):
        raise ValueError("attached SSL checkpoint is missing its training config")
    validate_competition_config(ssl_config, purpose="train")


def validate_submission_path(path: str | Path, config: dict) -> None:
    if bool(config.get("competition_mode", True)) and Path(path).name != "submission.csv":
        raise ValueError("competition_mode requires the output file to be named submission.csv")
