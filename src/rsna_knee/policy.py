from __future__ import annotations

from pathlib import Path

SSL_COMPETITION_SOURCE = "competition_training_data"


def validate_competition_config(config: dict, *, purpose: str) -> None:
    """Validate the conservative production contract used for Kaggle runs."""
    if not bool(config.get("competition_mode", True)):
        return

    budget = float(config.get("runtime_budget_hours", 8.5))
    if not 0 < budget < 9.0:
        raise ValueError("competition_mode requires runtime_budget_hours strictly below 9")

    if bool(config.get("pretrained", False)) and not bool(config.get("allow_external_pretrained", False)):
        raise ValueError(
            "competition_mode forbids external pretrained weights by default; "
            "set pretrained:false unless the competition-specific rules have been explicitly verified"
        )

    if int(config.get("requested_gpus", 1)) != 1:
        raise ValueError("production competition mode is single-GPU only")

    ssl_path = config.get("ssl_encoder_checkpoint")
    if ssl_path and not bool(config.get("allow_external_pretrained", False)):
        if str(config.get("ssl_checkpoint_source", "")) != SSL_COMPETITION_SOURCE:
            raise ValueError(
                "competition_mode requires ssl_checkpoint_source=competition_training_data "
                "for attached SSL weights when external pretrained artifacts are disabled"
            )
        # During actual training the attached file is present, so verify its
        # embedded provenance. When validating a saved checkpoint later, its
        # original training path may not be mounted; the saved config declaration
        # still remains part of the checkpoint policy contract.
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
        raise ValueError(
            "attached SSL checkpoint is missing competition-training-data provenance"
        )
    ssl_config = payload.get("config")
    if not isinstance(ssl_config, dict):
        raise ValueError("attached SSL checkpoint is missing its training config")
    validate_competition_config(ssl_config, purpose="train")


def validate_submission_path(path: str | Path, config: dict) -> None:
    if bool(config.get("competition_mode", True)) and Path(path).name != "submission.csv":
        raise ValueError("competition_mode requires the output file to be named submission.csv")
